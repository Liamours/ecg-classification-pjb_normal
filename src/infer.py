"""Run one pretrained model (`src.models`) on the digitized 12-lead records of one or more digitize runs and save every page's output.

Input per page: <run>/<page>/record.csv (12 leads in mV, 500 Hz, empty where a lead is missing) and record.json.
Output per dataset: <out_dir>/<dataset>/predictions.csv (model with a label head: one column per output) or embeddings.csv
(encoder: one column per embedding dimension), each row with page, relative_path, label and leads_ok.

Long runs: a log in `log_dir` with progress, pages per second and ETA over all pages still to do; resumable (rows are
appended after each batch and pages already in the output are skipped); preprocessing runs in `workers` data-loader
processes while the model runs on the GPU when there is one.

Usage:
    uv run python -m src.infer --config configs/infer/ecgfounder.yml
"""
import argparse
import csv
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src import paths
from src.models import MODELS
from src.records import load_labels, source_key

META = ["page", "relative_path", "label", "leads_ok"]


class Pages(Dataset):
    def __init__(self, pages: list[Path], prep):
        self.pages, self.prep = pages, prep

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, i: int):
        page = self.pages[i]
        x = np.genfromtxt(page / "record.csv", delimiter=",", skip_header=1)[:, 1:].T
        image = json.loads((page / "record.json").read_text(encoding="utf-8"))["image"]
        folder, rel = source_key(image["source"])
        return torch.from_numpy(self.prep(x)), page.name, folder, rel, int(image["leads_ok"])


def done_pages(out: Path, columns: list[str]) -> set[str]:
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(META + columns)
        return set()
    with out.open(encoding="utf-8") as fh:
        return {r["page"] for r in csv.DictReader(fh)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text(encoding="utf-8"))
    out_root = paths.resolve(cfg["out_dir"])
    log_dir = paths.resolve(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / f"{out_root.name}-{datetime.now():%Y%m%d}.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    log = logging.getLogger("infer")
    device = "cuda" if cfg["device"] == "auto" and torch.cuda.is_available() else ("cpu" if cfg["device"] == "auto" else cfg["device"])
    model = MODELS[cfg["model"]](cfg)
    model.load(device)
    torch.backends.cudnn.benchmark = True
    probe_x = torch.from_numpy(model.prep(np.zeros((12, 2000))))[None].to(device)
    with torch.inference_mode():
        width = model.forward(probe_x).shape[1]
    columns = model.labels or [f"emb_{i:04d}" for i in range(width)]
    name = "predictions.csv" if model.labels else "embeddings.csv"
    todo = {}
    for dataset, run in cfg["runs"].items():
        out = out_root / dataset / name
        done = done_pages(out, columns)
        todo[dataset] = (out, [p.parent for p in sorted(paths.resolve(run).glob("*/record.json")) if p.parent.name not in done])
    total = sum(len(t[1]) for t in todo.values())
    log.info("start %s on %s, %d pages to do, %d outputs per page, config %s", cfg["model"], device, total, len(columns), cfg)
    finished, start = 0, time.time()
    for dataset, (out, pages) in todo.items():
        if not pages:
            continue
        loader = DataLoader(Pages(pages, model.prep), batch_size=cfg["batch_size"], num_workers=cfg["workers"], pin_memory=device == "cuda", persistent_workers=False)
        labels = {}
        for x, names, folders, rels, leads_ok in tqdm(loader, desc=dataset, unit="batch"):
            with torch.inference_mode():
                y = model.forward(x.to(device, non_blocking=True)).float().cpu().numpy()
            rows = []
            for i, page in enumerate(names):
                if folders[i] not in labels:
                    labels[folders[i]] = load_labels(folders[i], cfg["label_column"])
                rows.append([page, rels[i], labels[folders[i]].get(rels[i], ""), int(leads_ok[i])] + [f"{v:.5f}" for v in y[i]])
            with out.open("a", newline="", encoding="utf-8") as fh:  # the output file is the progress record: a crash repeats at most one batch
                csv.writer(fh).writerows(rows)
            finished += len(rows)
            rate = finished / (time.time() - start)
            log.info("%s: %d of %d pages, %.1f pages/s, ETA %s", dataset, finished, total, rate, timedelta(seconds=round((total - finished) / rate)))
    log.info("done %s: %d pages in %s", cfg["model"], finished, timedelta(seconds=round(time.time() - start)))
    print(f"{cfg['model']}: {finished} pages written to {out_root} in {timedelta(seconds=round(time.time() - start))}")


if __name__ == "__main__":
    main()
