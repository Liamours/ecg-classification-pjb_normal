"""One clean fine-tuning run of one model on one split of the nested folds: the first finished fine-tuned model.

Uses the training code of `src.finetune` on the split `split` (repeat and fold of `manifest_folds.csv`): train pages train, val
pages stop training and set the per-level thresholds, test pages are scored once. Keeps the best-epoch model (`model.pt`: weights,
labels, thresholds, epoch, config) and writes the test predictions, the per-label and summary F1 tables, the NORMAL vs PJB table
(NORMAL = no label predicted) and the scan predictions. A rerun of a finished run only rescores and rewrites these outputs.
One progress line with ETA over the epoch limit (early stopping may end sooner) and a log in `log_dir`; a stopped run continues from
its last epoch. Every epoch's details go to `checkpoint_epochs.csv` (train and val loss, thresholds, val F1, precision,
sensitivity, exact-set accuracy and Hamming loss per level, val NORMAL vs PJB, best epoch, time); weights are kept only for the best
epoch by the `stop_on` score (`checkpoint_best.pt`, also in `model.pt`) and the latest epoch (`checkpoint.pt`, with optimizer state
for resuming). `src.finetune_variants` calls `run` once per training variant, without the test set.

Usage:
    uv run python -m src.finetune_single --config configs/finetune_single.yml
"""
import argparse
import csv
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


def run(cfg: dict, log: logging.Logger, score_test: bool = True, on_epoch=None) -> dict:
    """Train (or resume) one model on the configured split; returns the best epoch, thresholds and the val row of that epoch,
    plus the test levels and NORMAL vs PJB table when `score_test`."""
    out, report, inputs = paths.resolve(cfg["out_dir"]), paths.resolve(cfg["report_dir"]), paths.resolve(cfg["inputs_dir"])
    for d in (out, report, inputs):
        d.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec, name = cfg["spec"], cfg["model"]
    pages, x = prepare(spec, paths.resolve(cfg["run"]), inputs / f"inputs-{cfg['dataset']}.npz", cfg["workers"], log)
    labels, y = label_matrix(cfg, pages)
    parent = parents(labels, REPO / cfg["groups_config"], cfg["merge"], cfg["unknown_label"])
    role = load_folds(cfg, pages)[(cfg["split"]["repeat"], cfg["split"]["fold"])]
    tr, va, te = (np.flatnonzero(role == r) for r in ("train", "val", "test"))
    log.info("start %s on %s: %d train, %d val, %d test pages, %d labels, config %s", name, device, len(tr), len(va), len(te), len(labels), cfg)
    bar = None
    if on_epoch is None:
        bar = Progress(cfg["max_epochs"], name, "epochs", log, log_every=60.0)
        on_epoch = lambda state, loss, best: bar.step(state["epoch"] - bar.done, f"loss {loss:.4f}, {best}")
    net, t, best_epoch = train(spec, cfg, x, y, tr, va, parent, 0, out / "checkpoint.pt", device, name, log, on_epoch)
    if bar:
        bar.close()
    with (out / "checkpoint_epochs.csv").open(encoding="utf-8") as fh:
        val = next(r for r in csv.DictReader(fh) if int(r["epoch"]) == best_epoch)
    torch.save({"model": net.state_dict(), "labels": labels, "thresholds": t, "best_epoch": best_epoch, "config": cfg}, out / "model.pt")
    result = {"best_epoch": best_epoch, "thresholds": t, "val": val}
    if score_test:
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
        for ds, run_dir in cfg["predict"].items():
            rels, xs = prepare(spec, paths.resolve(run_dir), inputs / f"inputs-{ds}.npz", cfg["workers"], log)
            p = scores(net, xs, spec["batch_size"], device)
            write_csv(out / f"{ds}_predictions.csv", head, [[r] + [f"{v:.5f}" for v in p[i]] + [int(v) for v in predict(p, t, parent)[i]] for i, r in enumerate(rels)])
        result |= {"test_pages": int(len(te)), "levels": levels, "normal_vs_pjb": binary}
        log.info("test: best epoch %d, thresholds %s, micro F1 %.4f, NORMAL vs PJB macro F1 %.4f", best_epoch, t, levels["overall"]["micro_f1"], binary["macro_f1"])
    (report / "run.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    del net
    torch.cuda.empty_cache()
    return result


def print_test(result: dict) -> None:
    print("| Level | Micro F1 | Macro F1 | Precision | Sensitivity | Exact-set accuracy |\n|---|---|---|---|---|---|")
    for k, d in result["levels"].items():
        print(f"| {k} | {d['micro_f1']:.4f} | {d['macro_f1']:.4f} | {d['micro_precision']:.4f} | {d['micro_sensitivity']:.4f} | {d['exact_set_accuracy']:.4f} |")
    b = result["normal_vs_pjb"]
    print("\n| Class | Pages | F1 | Precision | Sensitivity |\n|---|---|---|---|---|")
    for k in ("normal", "pjb"):
        print(f"| {k.upper()} | {b[f'{k}_pages']} | {b[f'{k}_f1']:.4f} | {b[f'{k}_precision']:.4f} | {b[f'{k}_sensitivity']:.4f} |")
    print(f"| macro | - | {b['macro_f1']:.4f} | - | - |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text(encoding="utf-8"))
    log_dir = paths.resolve(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / f"finetune_single-{datetime.now():%Y%m%d}.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    print_test(run(cfg, logging.getLogger("finetune_single")))


if __name__ == "__main__":
    main()
