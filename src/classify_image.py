"""Image-direct NORMAL vs PJB inference: one prediction per page photo with a trained YOLO-cls model.

Reads each dataset's _labels/manifest.csv (pages with an ECG, duplicates skipped) and writes
<out_dir>/<dataset>/predictions.csv: relative_path, label, pred, p_pjb. Resumable: pages already in the file are skipped.

Usage:
    uv run python -m src.classify_image --config configs/classify_image.yml
"""
import argparse
import csv
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm
from ultralytics import YOLO

from src import paths

FIELDS = ["relative_path", "label", "pred", "p_pjb"]


def pages(dataset: str, label_column: str) -> list[dict]:
    with (paths.dataset(dataset) / "_labels" / "manifest.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if r["relative_path"] and r.get("has_ecg") == "True" and r.get("is_duplicate", "False") != "True"]


def summary(rows: list[dict]) -> str:
    pred = Counter(r["pred"] for r in rows)
    out = [f"pages {len(rows)}, predicted " + ", ".join(f"{k} {v}" for k, v in sorted(pred.items()))]
    labeled = [r for r in rows if r["label"]]
    if labeled:
        conf = Counter((r["label"], r["pred"]) for r in labeled)
        out.append("| Label | Pages | Pred NORMAL | Pred PJB | Recall |\n|---|---|---|---|---|")
        for lab in sorted({r["label"] for r in labeled}):
            n = sum(v for (l, _), v in conf.items() if l == lab)
            out.append(f"| {lab} | {n} | {conf[(lab, 'NORMAL')]} | {conf[(lab, 'PJB')]} | {conf[(lab, lab)] / n:.4f} |")
        out.append(f"accuracy {sum(conf[(l, l)] for l in ('NORMAL', 'PJB')) / len(labeled):.4f}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text(encoding="utf-8"))
    log_dir = paths.resolve(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out_root = paths.resolve(cfg["out_dir"])
    logging.basicConfig(filename=log_dir / f"{out_root.name}-{datetime.now():%Y%m%d}.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    log = logging.getLogger("classify_image")
    model = YOLO(str(paths.resolve(cfg["weights"])))
    names = model.names
    for dataset in cfg["datasets"]:
        out = out_root / dataset / "predictions.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            out.write_text(",".join(FIELDS) + "\n", encoding="utf-8")
        with out.open(encoding="utf-8") as fh:
            done = {r["relative_path"] for r in csv.DictReader(fh)}
        for page in tqdm([p for p in pages(dataset, cfg["label_column"]) if p["relative_path"] not in done], desc=dataset):
            probs = model.predict(str(paths.dataset(dataset) / page["relative_path"]), imgsz=cfg["imgsz"], device=cfg["device"], verbose=False)[0].probs
            row = {"relative_path": page["relative_path"], "label": page.get(cfg["label_column"], ""), "pred": names[int(probs.top1)],
                   "p_pjb": f"{float(probs.data[[k for k, v in names.items() if v == 'PJB'][0]]):.4f}"}
            with out.open("a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=FIELDS).writerow(row)
            log.info("%s %s: %s %s", dataset, row["relative_path"], row["pred"], row["p_pjb"])
        with out.open(encoding="utf-8") as fh:
            print(f"\n{dataset}\n{summary(list(csv.DictReader(fh)))}")


if __name__ == "__main__":
    main()
