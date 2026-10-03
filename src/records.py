"""Turn a digitize run into classification records: one 12-lead array per page, its label, and whether it is usable.

Reads <run>/<page>/page.json, panel*.json and panel*.csv as written by ecg-digitization-synthetic_realistic. Writes
<out-dir>/records/<page>.npz (signal: leads x samples in mV, NaN where there is no data; ok: lead passed the digitizer's
quality flag; present: lead was digitized at all) and <out-dir>/index.csv, one row per page with the label from the
dataset manifest. Resumable: a page already in index.csv is skipped.

Usage:
    uv run python -m src.records --config configs/records.yml --run @inferences/digitize_full/mac400 --out-dir @datasets/training/records-digitize_full-mac400
"""
import argparse
import csv
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from src import paths

INDEX_FIELDS = ["page", "dataset", "relative_path", "label", "leads_present", "leads_ok", "usable", "file"]


def read_panel(csv_path: Path) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, str]]]:
    """Time in seconds and, per lead, its mV series and quality flag."""
    with csv_path.open(encoding="utf-8") as fh:
        head, *body = csv.reader(fh)
    n = (len(head) - 1) // 2
    t = np.array([float(r[0]) for r in body])
    return t, {head[1 + i][:-3]: (np.array([float(r[1 + i]) if r[1 + i] else np.nan for r in body]), body[0][1 + n + i]) for i in range(n)}


def page_record(page_dir: Path, cfg: dict) -> dict:
    """The page's leads on one time base. A lead digitized in two panels keeps the one that passed the quality flag."""
    n = int(round(cfg["duration_s"] * cfg["fs"]))
    grid = np.arange(n) / cfg["fs"]
    signal = np.full((len(cfg["leads"]), n), np.nan, np.float32)
    ok = np.zeros(len(cfg["leads"]), bool)
    present = np.zeros(len(cfg["leads"]), bool)
    for pj in sorted(page_dir.glob("panel*.json"), key=lambda p: int(p.stem[5:])):
        if json.loads(pj.read_text(encoding="utf-8"))["error"] or not pj.with_suffix(".csv").exists():
            continue
        t, leads = read_panel(pj.with_suffix(".csv"))
        for name, (mv, flag) in leads.items():
            if name not in cfg["leads"]:
                continue
            i = cfg["leads"].index(name)
            if present[i] and (ok[i] or flag != "ok"):
                continue
            signal[i] = np.interp(grid, t - t[0], mv, right=np.nan)
            present[i], ok[i] = True, flag == "ok"
    return {"signal": signal, "ok": ok, "present": present}


def source_key(source: str) -> tuple[str, str]:
    """Dataset folder and the path inside it, from the page's recorded source path."""
    parts = source.replace("\\", "/").split("/")
    i = parts.index("datasets")
    return parts[i + 1], "/".join(parts[i + 2:])


def load_labels(dataset: str, column: str) -> dict[str, str]:
    manifest = paths.dataset(dataset) / "_labels" / "manifest.csv"
    if not manifest.exists():
        return {}
    with manifest.open(encoding="utf-8") as fh:
        return {r["relative_path"]: r.get(column, "") for r in csv.DictReader(fh)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run", type=paths.resolve, required=True)
    ap.add_argument("--out-dir", type=paths.resolve, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    log_dir = paths.resolve(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / f"records-{args.out_dir.name}-{datetime.now():%Y%m%d}.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    log = logging.getLogger("records")
    (args.out_dir / "records").mkdir(parents=True, exist_ok=True)
    index = args.out_dir / "index.csv"
    rows = []
    if index.exists():
        with index.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    else:
        index.write_text(",".join(INDEX_FIELDS) + "\n", encoding="utf-8")
    done = {r["page"] for r in rows}
    labels: dict[str, dict[str, str]] = {}
    for page_json in tqdm(sorted(args.run.glob("*/page.json")), desc="pages"):
        page = page_json.parent.name
        if page in done:
            continue
        dataset, rel = source_key(json.loads(page_json.read_text(encoding="utf-8"))["source"])
        if dataset not in labels:
            labels[dataset] = load_labels(dataset, cfg["label_column"])
        rec = page_record(page_json.parent, cfg)
        np.savez_compressed(args.out_dir / "records" / f"{page}.npz", **rec)
        row = {"page": page, "dataset": dataset, "relative_path": rel, "label": labels[dataset].get(rel, ""), "leads_present": int(rec["present"].sum()), "leads_ok": int(rec["ok"].sum()),
               "usable": int(rec["ok"].sum()) >= cfg["usable"]["min_ok_leads"], "file": f"records/{page}.npz"}
        with index.open("a", newline="", encoding="utf-8") as fh:  # the record file is written first, so a crash repeats one page and loses none
            csv.DictWriter(fh, fieldnames=INDEX_FIELDS).writerow(row)
        rows.append({k: str(v) for k, v in row.items()})
        log.info("%s: %d leads present, %d ok, label %s", page, row["leads_present"], row["leads_ok"], row["label"] or "-")
    total, usable = Counter((r["dataset"], r["label"] or "-") for r in rows), Counter((r["dataset"], r["label"] or "-") for r in rows if r["usable"] == "True")
    print("| Dataset | Label | Pages | Usable records | Mean leads ok |\n|---|---|---|---|---|")
    for key in sorted(total):
        ok = [int(r["leads_ok"]) for r in rows if (r["dataset"], r["label"] or "-") == key]
        print(f"| {key[0]} | {key[1]} | {total[key]} | {usable[key]} | {np.mean(ok):.1f} |")


if __name__ == "__main__":
    main()
