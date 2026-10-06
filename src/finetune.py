"""Fine-tuning of pretrained ECG models on the hierarchical multi-label CHD labels, with the nested cross-validation folds.

Per model: the digitized records are prepared once with the model's own preprocessing (`src.models` adapter) and cached. Per model,
repeat and outer fold (one unit): a new linear output layer on the model's embedding, the whole network trained on the fold's
train pages (AdamW, differential learning rate, BCE with per-label positive weights, mixed precision, augmentation), the epoch
with the best validation micro F1 kept (early stopping), one threshold per level chosen on validation (`src.thresholds`), the
test pages scored once. Per model (one more unit): a network trained on every labeled page for the median best epoch count of
its folds, with the median thresholds, scores the unlabeled sets in `predict` and is kept.

Writes to `out_dir/<model>/`: `inputs-<dataset>.npz` (prepared inputs), `parts/r<repeat>_f<fold>.csv` and `_settings.json`,
`oof_r<repeat>.csv`, `<dataset>_predictions.csv`, `final.pt`; to `report_dir`: `per_label.csv`, `summary.csv` (F1 by level),
`normal_vs_pjb.csv` (NORMAL = no label predicted). Every epoch's losses, thresholds and val metrics go to
`parts/r<repeat>_f<fold>_checkpoint_epochs.csv`. Long runs: one progress line with ETA over units, a log in `log_dir`; a unit
saves a checkpoint after every epoch and resumes from it, and finished units are skipped.

Usage:
    uv run python -m src.finetune --config configs/finetune.yml [--models ecgfounder]
"""
import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src import paths
from src.infer import Pages
from src.linear_probe import label_matrix, load_folds, metrics, normal_vs_pjb, write_csv, write_report
from src.models import MODELS
from src.progress import Progress, clock
from src.thresholds import choose, parents, predict

REPO = Path(__file__).resolve().parents[1]


class Net(nn.Module):
    def __init__(self, adapter, n_labels: int, dim: int):
        super().__init__()
        self.adapter, self.backbone, self.head = adapter, adapter.net, nn.Linear(dim, n_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.adapter.forward(x).float())


def build(spec: dict, n_labels: int, device: str) -> Net:
    cfg = yaml.safe_load((REPO / spec["config"]).read_text(encoding="utf-8"))
    adapter = MODELS[cfg["model"]](cfg)
    adapter.load(device)
    with torch.no_grad():
        dim = adapter.forward(torch.from_numpy(adapter.prep(np.zeros((12, 2000))))[None].to(device)).shape[1]
    for name, p in adapter.net.named_parameters():
        p.requires_grad = not any(name.startswith(f) for f in spec["frozen"])
    return Net(adapter, n_labels, dim).to(device)


def prepare(spec: dict, run: Path, out: Path, workers: int, log) -> tuple[list[str], np.ndarray]:
    """Inputs of every page through the model's own preprocessing, cached in `out` (rel paths sorted)."""
    if out.exists():
        d = np.load(out, allow_pickle=False)
        return list(d["relative_path"]), d["x"]
    cfg = yaml.safe_load((REPO / spec["config"]).read_text(encoding="utf-8"))
    adapter = MODELS[cfg["model"]](cfg)
    pages = sorted(p.parent for p in run.glob("*/record.json"))
    xs, rels = [], []
    bar = Progress(len(pages), f"prepare {out.stem}", "pages", log)
    for x, _, _, rel, _ in DataLoader(Pages(pages, adapter.prep), batch_size=32, num_workers=workers):
        xs.append(x.numpy())
        rels += list(rel)
        bar.step(len(rel))
    bar.close()
    order = np.argsort(rels)
    rels, x = [rels[i] for i in order], np.concatenate(xs)[order]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, relative_path=np.array(rels), x=x)
    return rels, x


def augment(x: torch.Tensor, a: dict) -> torch.Tensor:
    x = x * (1 + a["amplitude"] * (2 * torch.rand(x.shape[0], x.shape[1], 1, device=x.device) - 1))  # inputs are batch x leads x samples
    x = x + a["noise"] * x.std(dim=-1, keepdim=True) * torch.randn_like(x)
    shift = int(a["shift"] * x.shape[-1])
    return torch.roll(x, int(torch.randint(-shift, shift + 1, (1,))), dims=-1) if shift else x


def scores(net: Net, x: np.ndarray, bs: int, device: str) -> np.ndarray:
    net.eval()
    out = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=device == "cuda"):
        for i in range(0, len(x), bs):
            out.append(torch.sigmoid(net(torch.from_numpy(x[i:i + bs]).to(device))).float().cpu().numpy())
    return np.concatenate(out)


def train(spec, cfg, x, y, train_idx, val_idx, parent, epochs_fixed, ckpt: Path, device, note, log, on_epoch=None) -> tuple[Net, dict, int]:
    """Train on train_idx; early stopping on val micro F1 (or a fixed epoch count when val_idx is empty). Resumes from ckpt.
    `on_epoch(state, loss, best_text)` replaces the default epoch line when given."""
    torch.manual_seed(cfg["seed"])
    net = build(spec, y.shape[1], device)
    groups = [{"params": [p for p in net.backbone.parameters() if p.requires_grad], "lr": cfg["lr"]["backbone"]}, {"params": net.head.parameters(), "lr": cfg["lr"]["head"]}]
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    pos = y[train_idx].sum(0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.clip((len(train_idx) - pos) / np.maximum(pos, 1), 1, cfg["pos_weight_max"]), dtype=torch.float32, device=device))
    state = {"epoch": 0, "best_f1": -1.0, "best_epoch": 0, "stale": 0, "thresholds": {"group": 0.5, "diagnosis": 0.5}}
    if ckpt.exists():
        c = torch.load(ckpt, map_location=device, weights_only=True)
        net.load_state_dict(c["model"]), opt.load_state_dict(c["opt"]), scaler.load_state_dict(c["scaler"])
        state = c["state"]
        log.info("%s: resumed after epoch %d", note, state["epoch"])
    best_path = ckpt.with_name(ckpt.stem + "_best.pt")
    max_epochs = epochs_fixed or cfg["max_epochs"]
    xt, yt = torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx].astype(np.float32))
    while state["epoch"] < max_epochs and state["stale"] < cfg["patience"]:
        net.train()
        perm = torch.randperm(len(train_idx))
        t0, total = time.time(), 0.0
        for i in range(0, len(perm), spec["batch_size"]):
            b = perm[i:i + spec["batch_size"]]
            xb, yb = augment(xt[b].to(device), cfg["augment"]), yt[b].to(device)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = loss_fn(net(xb).float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(b)
        state["epoch"] += 1
        row = {"epoch": state["epoch"], "train_loss": total / len(train_idx), "lr_backbone": opt.param_groups[0]["lr"], "lr_head": opt.param_groups[1]["lr"]}
        if len(val_idx):
            prob = scores(net, x[val_idx], spec["batch_size"], device)
            t = choose(prob, y[val_idx], parent)
            pred = predict(prob, t, parent)
            f1 = _micro(y[val_idx], pred)
            logit = torch.logit(torch.from_numpy(prob).clamp(1e-6, 1 - 1e-6)).to(device)
            row["val_loss"] = loss_fn(logit, torch.from_numpy(y[val_idx].astype(np.float32)).to(device)).item()
            row |= {"threshold_group": t["group"], "threshold_diagnosis": t["diagnosis"]}
            names = [f"group: {j}" if parent[j] < 0 else f"diagnosis: {j}" for j in range(y.shape[1])]
            for level, m in metrics(y[val_idx], pred, names)[1].items():
                row |= {f"val_{level}_{k}": v for k, v in m.items()}
            row["is_best"] = int(f1 > state["best_f1"])
            if f1 > state["best_f1"]:
                state.update(best_f1=f1, best_epoch=state["epoch"], stale=0, thresholds=t)
                torch.save(net.state_dict(), best_path)
            else:
                state["stale"] += 1
        else:
            torch.save(net.state_dict(), best_path)
            state["best_epoch"] = state["epoch"]
        row |= {"best_epoch": state["best_epoch"], "epochs_without_improvement": state["stale"], "seconds": time.time() - t0}
        history(ckpt, row, state["epoch"])
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "state": state}, ckpt)  # latest weights; the best are in best_path
        best = f"val micro F1 best {state['best_f1']:.4f} (epoch {state['best_epoch']})" if len(val_idx) else "no validation, fixed epoch count"
        log.info("%s: epoch %d, train loss %.4f, %s, %s", note, state["epoch"], total / len(train_idx), best, clock(time.time() - t0))
        if on_epoch:
            on_epoch(state, total / len(train_idx), best)
        else:
            print(f"\r    {note}: epoch {state['epoch']}/{max_epochs}, loss {total / len(train_idx):.4f}, {best}".ljust(110), end="", flush=True)
    net.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return net, state["thresholds"], state["best_epoch"]


def history(ckpt: Path, row: dict, epoch: int) -> None:
    """Append one epoch to `<checkpoint>_epochs.csv` (train loss, val loss, thresholds, val metrics per level, best epoch, time),
    dropping rows past `epoch` first, which a stop between the history write and the checkpoint write can leave behind."""
    path = ckpt.with_name(ckpt.stem + "_epochs.csv")
    old = list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []
    rows = [r for r in old if int(r["epoch"]) < epoch] + [{k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()}]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[-1]))
        w.writeheader()
        w.writerows(rows)


def _micro(y: np.ndarray, pred: np.ndarray) -> float:
    tp, fp, fn = (y & pred).sum(), (~y & pred).sum(), (y & ~pred).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--models", nargs="+", help="run only these models of the config (default: every model)")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.models:
        cfg["models"] = {m: cfg["models"][m] for m in args.models}
    out_root, report, log_dir = paths.resolve(cfg["out_dir"]), paths.resolve(cfg["report_dir"]), paths.resolve(cfg["log_dir"])
    for d in (out_root, report, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / f"finetune-{datetime.now():%Y%m%d}.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    log = logging.getLogger("finetune")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {m: prepare(s, paths.resolve(cfg["run"]), out_root / m / f"inputs-{cfg['dataset']}.npz", cfg["workers"], log) for m, s in cfg["models"].items()}
    pages = inputs[next(iter(inputs))][0]
    labels, y = label_matrix(cfg, pages)
    parent = parents(labels, REPO / cfg["groups_config"], cfg["merge"], cfg["unknown_label"])
    folds = load_folds(cfg, pages)
    units = [(m, r, f) for m in cfg["models"] for (r, f) in folds] + [(m, 0, 0) for m in cfg["models"]]
    todo = [u for u in units if not (out_root / u[0] / "parts" / (f"r{u[1]}_f{u[2]}.csv" if u[1] else "final.done")).exists()]
    log.info("start on %s: %d models, %d labels, %d folds, %d of %d units to do, config %s", device, len(cfg["models"]), len(labels), len(folds), len(todo), len(units), cfg)
    bar = Progress(len(todo), "fine-tune", "units", log)
    for model, rep, fold in todo:
        spec, (pg, x) = cfg["models"][model], inputs[model]
        assert pg == pages, f"{model} covers other pages"
        part = out_root / model / "parts"
        part.mkdir(parents=True, exist_ok=True)
        note = f"{model} r{rep} f{fold}" if rep else f"{model} final"
        if rep:
            role = folds[(rep, fold)]
            tr, va, te = (np.flatnonzero(role == r) for r in ("train", "val", "test"))
            ckpt = part / f"r{rep}_f{fold}_checkpoint.pt"
            net, t, best_epoch = train(spec, cfg, x, y, tr, va, parent, 0, ckpt, device, note, log)
            prob = scores(net, x[te], spec["batch_size"], device)
            pred = predict(prob, t, parent)
            (part / f"r{rep}_f{fold}_settings.json").write_text(json.dumps({"thresholds": t, "best_epoch": best_epoch}), encoding="utf-8")
            write_csv(part / f"r{rep}_f{fold}.csv", ["relative_path"] + [f"{l} prob" for l in labels] + [f"{l} pred" for l in labels],
                      [[pages[i]] + [f"{v:.5f}" for v in prob[k]] + [int(v) for v in pred[k]] for k, i in enumerate(te)])
            if not cfg["keep_fold_checkpoints"]:
                for f in (ckpt, ckpt.with_name(ckpt.stem + "_best.pt")):
                    f.unlink(missing_ok=True)
        else:
            settings = [json.loads((part / f"r{r}_f{f}_settings.json").read_text(encoding="utf-8")) for (r, f) in folds]
            epochs = int(np.median([s["best_epoch"] for s in settings]))
            t = {k: float(np.median([s["thresholds"][k] for s in settings])) for k in ("group", "diagnosis")}
            ckpt = part / "final_checkpoint.pt"
            net, _, _ = train(spec, cfg, x, y, np.arange(len(pages)), np.array([], int), parent, epochs, ckpt, device, note, log)
            torch.save({"model": net.state_dict(), "labels": labels, "thresholds": t, "epochs": epochs}, out_root / model / "final.pt")
            for ds, run in cfg["predict"].items():
                rels, xs = prepare(spec, paths.resolve(run), out_root / model / f"inputs-{ds}.npz", cfg["workers"], log)
                prob = scores(net, xs, spec["batch_size"], device)
                pred = predict(prob, t, parent)
                write_csv(out_root / model / f"{ds}_predictions.csv", ["relative_path"] + [f"{l} prob" for l in labels] + [f"{l} pred" for l in labels],
                          [[p] + [f"{v:.5f}" for v in prob[i]] + [int(v) for v in pred[i]] for i, p in enumerate(rels)])
            for f in (ckpt, ckpt.with_name(ckpt.stem + "_best.pt")):
                f.unlink(missing_ok=True)
            write_csv(part / "final.done", ["key", "value"], [["epochs", epochs], ["thresholds", json.dumps(t)]])
        print("\r" + " " * 110, end="\r")
        bar.step(1, note)
    bar.close()
    per_label, summary, binary = [], [], []
    idx = {p: i for i, p in enumerate(pages)}
    for model in cfg["models"]:
        for rep in sorted({r for r, _ in folds}):
            prob, pred = np.zeros(y.shape), np.zeros(y.shape, bool)
            for (r, f) in folds:
                if r != rep:
                    continue
                with (out_root / model / "parts" / f"r{r}_f{f}.csv").open(encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        i = idx[row["relative_path"]]
                        prob[i] = [float(row[f"{l} prob"]) for l in labels]
                        pred[i] = [row[f"{l} pred"] == "1" for l in labels]
            write_csv(out_root / model / f"oof_r{rep}.csv", ["relative_path"] + [f"{l} prob" for l in labels] + [f"{l} pred" for l in labels],
                      [[p] + [f"{v:.5f}" for v in prob[i]] + [int(v) for v in pred[i]] for i, p in enumerate(pages)])
            rows, levels = metrics(y, pred, labels)
            per_label += [{"model": model, "repeat": rep, **r} for r in rows]
            summary += [{"model": model, "repeat": rep, "level": name, **v} for name, v in levels.items()]
            binary.append({"model": model, "repeat": rep, **normal_vs_pjb(y, pred)})
    write_report(report, per_label, summary, binary, list(cfg["models"]))
    log.info("report written to %s", report)


if __name__ == "__main__":
    main()
