"""Training variants of the single fine-tuning run (`src.finetune_single`), compared on validation pages only.

Every combination of `factors` (output-layer start, early-stopping score, positive-weight cap) is one variant, trained on the base
config's split into `out_dir/<variant>/` with its own `checkpoint_epochs.csv`. The variants are ranked by `select_by` at their best
epoch; only the top variant is scored on the test pages. Writes `report_dir/variants.csv` (one row per variant: settings, best epoch,
epochs run, val scores) and the top variant's test report in `report_dir/<variant>/`. One progress line over variants with the
current epoch, a log in `log_dir`; finished variants are skipped and a stopped one resumes from its last epoch.

Usage:
    uv run python -m src.finetune_variants --config configs/finetune_variants.yml
"""
import argparse
import csv
import itertools
import json
import logging
from datetime import datetime
from pathlib import Path

import yaml

from src import paths
from src.finetune_single import print_test, run
from src.linear_probe import write_csv
from src.progress import Progress

REPO = Path(__file__).resolve().parents[1]
COLUMNS = ["val_loss", "val_overall_micro_f1", "val_overall_macro_f1", "val_group_micro_f1", "val_diagnosis_micro_f1", "val_binary_macro_f1", "val_binary_normal_sensitivity", "val_binary_pjb_sensitivity"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text(encoding="utf-8"))
    base = yaml.safe_load((REPO / cfg["base"]).read_text(encoding="utf-8"))
    out_root, report_root, log_dir = paths.resolve(cfg["out_dir"]), paths.resolve(cfg["report_dir"]), paths.resolve(cfg["log_dir"])
    for d in (out_root, report_root, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / f"finetune_variants-{datetime.now():%Y%m%d}.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    log = logging.getLogger("finetune_variants")
    keys = list(cfg["factors"])
    variants = {}
    for values in itertools.product(*cfg["factors"].values()):
        name = "_".join(f"{k}-{v}" for k, v in zip(keys, values))
        variants[name] = base | dict(zip(keys, values)) | {"probe": cfg["probe"], "out_dir": str(out_root / name), "report_dir": str(report_root / name)}
    todo = [n for n in variants if not (report_root / n / "run.json").exists()]
    log.info("start: %d variants, %d to do, factors %s", len(variants), len(todo), cfg["factors"])
    bar = Progress(len(todo), "variants", "variants", log)
    for name in todo:
        run(variants[name], log, score_test=False, on_epoch=lambda state, loss, best, n=name: bar.show(f"{n}: epoch {state['epoch']}, loss {loss:.4f}, {best}"))
        bar.step(1, name)
    bar.close()
    rows = []
    for name, v in variants.items():
        r = json.loads((report_root / name / "run.json").read_text(encoding="utf-8"))
        with (out_root / name / "checkpoint_epochs.csv").open(encoding="utf-8") as fh:
            epochs = max(int(e["epoch"]) for e in csv.DictReader(fh))
        rows.append({"variant": name, **{k: v[k] for k in keys}, "best_epoch": r["best_epoch"], "epochs_run": epochs, **{c: float(r["val"][c]) for c in COLUMNS}})
    rows.sort(key=lambda r: -r[cfg["select_by"]])
    write_csv(report_root / "variants.csv", list(rows[0]), [[f"{x:.4f}" if isinstance(x, float) else x for x in r.values()] for r in rows])
    print("| Variant | Best epoch | Epochs run | Val loss | Val micro F1 | Val macro F1 | Val NORMAL vs PJB macro F1 | Val NORMAL sensitivity |\n|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['variant']} | {r['best_epoch']} | {r['epochs_run']} | {r['val_loss']:.4f} | {r['val_overall_micro_f1']:.4f} | {r['val_overall_macro_f1']:.4f} | {r['val_binary_macro_f1']:.4f} | {r['val_binary_normal_sensitivity']:.4f} |")
    top = rows[0]["variant"]
    log.info("top variant by %s: %s; scoring its test pages", cfg["select_by"], top)
    print(f"\nTest, {top}:")
    print_test(run(variants[top], log, score_test=True, on_epoch=lambda *a: None))


if __name__ == "__main__":
    main()
