"""One clean fine-tuning run of one model on one split of the nested folds: the first finished fine-tuned model.

Uses the training code of `src.finetune` on the split `split` (repeat and fold of `manifest_folds.csv`): train pages train, val
pages stop training and set the per-level thresholds, test pages are scored once. Keeps the best-epoch model (`model.pt`: weights,
labels, thresholds, epoch, config) and writes the test predictions, the per-label and summary F1 tables, the NORMAL vs PJB table
(NORMAL = no label predicted) and the scan predictions. A rerun of a finished run only rescores and rewrites these outputs.
One progress line with ETA over the epoch limit (early stopping may end sooner) and a log in `log_dir`; a stopped run continues from
its last epoch. Every epoch's details go to `checkpoint_epochs.csv` (train and val loss, thresholds, val F1, precision,
sensitivity, exact-set accuracy and Hamming loss per level, best epoch, time); weights are kept only for the best epoch by val micro F1
(`checkpoint_best.pt`, also in `model.pt`) and the latest epoch (`checkpoint.pt`, with optimizer state for resuming).

Usage:
    uv run python -m src.finetune_single --config configs/finetune_single.yml
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from src import paths
from src.finetune import prepare, scores, train
from src.linear_probe import label_matrix, load_folds, metrics, normal_vs_pjb, write_csv
from src.progress import Progress
from src.thresholds import parents, predict

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text(encoding="utf-8"))
    out, report, log_dir = paths.resolve(cfg["out_dir"]), paths.resolve(cfg["report_dir"]), paths.resolve(cfg["log_dir"])
    for d in (out, report, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / f"finetune_single-{datetime.now():%Y%m%d}.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    log = logging.getLogger("finetune_single")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec, name = cfg["spec"], cfg["model"]
    pages, x = prepare(spec, paths.resolve(cfg["run"]), out / f"inputs-{cfg['dataset']}.npz", cfg["workers"], log)
    labels, y = label_matrix(cfg, pages)
    parent = parents(labels, REPO / cfg["groups_config"], cfg["merge"], cfg["unknown_label"])
    role = load_folds(cfg, pages)[(cfg["split"]["repeat"], cfg["split"]["fold"])]
    tr, va, te = (np.flatnonzero(role == r) for r in ("train", "val", "test"))
    log.info("start %s on %s: %d train, %d val, %d test pages, %d labels, config %s", name, device, len(tr), len(va), len(te), len(labels), cfg)
    bar = Progress(cfg["max_epochs"], name, "epochs", log, log_every=60.0)

    def on_epoch(state: dict, loss: float, best: str) -> None:
        bar.step(state["epoch"] - bar.done, f"loss {loss:.4f}, {best}")

    net, t, best_epoch = train(spec, cfg, x, y, tr, va, parent, 0, out / "checkpoint.pt", device, name, log, on_epoch)
    bar.close()
    prob = scores(net, x[te], spec["batch_size"], device)
    pred = predict(prob, t, parent)
    rows, levels = metrics(y[te], pred, labels)
    binary = normal_vs_pjb(y[te], pred)
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else v
    write_csv(report / "per_label.csv", list(rows[0]), [[fmt(v) for v in r.values()] for r in rows])
    write_csv(report / "summary.csv", ["level"] + list(levels["overall"]), [[k] + [fmt(v) for v in d.values()] for k, d in levels.items()])
    write_csv(report / "normal_vs_pjb.csv", list(binary), [[fmt(v) for v in binary.values()]])
    head = ["relative_path"] + [f"{l} prob" for l in labels] + [f"{l} pred" for l in labels]
    write_csv(out / "test_predictions.csv", head, [[pages[i]] + [f"{v:.5f}" for v in prob[k]] + [int(v) for v in pred[k]] for k, i in enumerate(te)])
    torch.save({"model": net.state_dict(), "labels": labels, "thresholds": t, "best_epoch": best_epoch, "config": cfg}, out / "model.pt")
    for ds, run in cfg["predict"].items():
        rels, xs = prepare(spec, paths.resolve(run), out / f"inputs-{ds}.npz", cfg["workers"], log)
        p = scores(net, xs, spec["batch_size"], device)
        write_csv(out / f"{ds}_predictions.csv", head, [[r] + [f"{v:.5f}" for v in p[i]] + [int(v) for v in predict(p, t, parent)[i]] for i, r in enumerate(rels)])
    (report / "run.json").write_text(json.dumps({"best_epoch": best_epoch, "thresholds": t, "test_pages": int(len(te)), "levels": levels, "normal_vs_pjb": binary}, indent=1, default=float), encoding="utf-8")
    log.info("done: best epoch %d, thresholds %s, test micro F1 %.4f, NORMAL vs PJB macro F1 %.4f", best_epoch, t, levels["overall"]["micro_f1"], binary["macro_f1"])
    print("| Level | Micro F1 | Macro F1 | Precision | Sensitivity | Exact-set accuracy |\n|---|---|---|---|---|---|")
    for k, d in levels.items():
        print(f"| {k} | {d['micro_f1']:.4f} | {d['macro_f1']:.4f} | {d['micro_precision']:.4f} | {d['micro_sensitivity']:.4f} | {d['exact_set_accuracy']:.4f} |")
    print("\n| Class | Pages | F1 | Precision | Sensitivity |\n|---|---|---|---|---|")
    for k in ("normal", "pjb"):
        print(f"| {k.upper()} | {binary[f'{k}_pages']} | {binary[f'{k}_f1']:.4f} | {binary[f'{k}_precision']:.4f} | {binary[f'{k}_sensitivity']:.4f} |")
    print(f"| macro | - | {binary['macro_f1']:.4f} | - | - |")


if __name__ == "__main__":
    main()
