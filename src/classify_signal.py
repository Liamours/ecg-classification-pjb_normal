"""Signal inference with the pretrained ECGFounder on digitized 12-lead records, no fine-tuning.

Input per page: <run>/<page>/record.csv (12 leads in mV, 500 Hz, 4 s, empty where a lead is missing) and record.json.
Preprocessing repeats the upstream ptbxl_eval.py: leads I, II, III, aVR, aVL, aVF, V1 to V6, missing values set to 0,
one z-score over the whole array; the 4 s record is zero-padded to the model's 10 s. Writes
<out_dir>/<dataset>/predictions.csv: page, relative_path, label, leads_ok, then one sigmoid output per ECGFounder label.
Resumable per page. For labeled pages the summary ranks labels by how well their output separates PJB from NORMAL (AUROC).

Usage:
    uv run python -m src.classify_signal --config configs/classify_signal.yml
"""
import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src import paths
from src.records import load_labels, source_key

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "ecgfounder"))
from net1d import Net1D  # noqa: E402


def load_model(weights: Path, n_classes: int, device: str) -> Net1D:
    """The 12-lead ECGFounder exactly as built in the upstream ptbxl_eval.py."""
    model = Net1D(in_channels=12, base_filters=64, ratio=1, filter_list=[64, 160, 160, 400, 400, 1024, 1024], m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
                  kernel_size=16, stride=2, groups_width=16, verbose=False, use_bn=False, use_do=False, n_classes=n_classes)
    checkpoint = torch.load(weights, map_location=device, weights_only=False)  # the upstream file also pickles its scheduler, which weights_only=True refuses
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def read_record(page_dir: Path, n_samples: int, fill: str) -> np.ndarray:
    """fill `zero`: missing values 0 and the record zero-padded to n_samples; `tile`: each lead's digitized stretch repeated to n_samples."""
    x = np.genfromtxt(page_dir / "record.csv", delimiter=",", skip_header=1)[:, 1:].T
    if fill == "tile":
        x = np.stack([np.resize(lead[~np.isnan(lead)], n_samples) if (~np.isnan(lead)).any() else np.zeros(n_samples) for lead in x])
    x = np.pad(np.nan_to_num(x, nan=0.0), ((0, 0), (0, n_samples - x.shape[1])))
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    d = score[positive][:, None] - score[~positive][None, :]  # ponytail: all pairs, fine for a few thousand pages
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def summary(rows: list[dict], tasks: list[str], k: int) -> str:
    probs = np.array([[float(r[t]) for t in tasks] for r in rows])
    out = [f"pages {len(rows)}; highest mean output:"]
    out += [f"  {tasks[i]}: {probs[:, i].mean():.4f}" for i in np.argsort(-probs.mean(0))[:k]]
    labels = np.array([r["label"] for r in rows])
    for name, keep in [("all pages", np.ones(len(rows), bool)), ("pages with 12 leads ok", np.array([r["leads_ok"] == "12" for r in rows]))]:
        pjb = labels[keep] == "PJB"
        if pjb.all() or not pjb.any():
            continue
        a = np.array([auroc(probs[keep, i], pjb) for i in range(len(tasks))])
        out.append(f"\nPJB vs NORMAL, {name} ({pjb.sum()} PJB, {(~pjb).sum()} NORMAL); labels ranked by |AUROC - 0.5|, AUROC above 0.5 = higher output on PJB")
        out.append("| ECGFounder label | AUROC | Mean on PJB | Mean on NORMAL |\n|---|---|---|---|")
        out += [f"| {tasks[i]} | {a[i]:.4f} | {probs[keep][pjb, i].mean():.4f} | {probs[keep][~pjb, i].mean():.4f} |" for i in np.argsort(-np.abs(a - 0.5))[:k]]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text(encoding="utf-8"))
    out_root = paths.resolve(cfg["out_dir"])
    log_dir = paths.resolve(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / f"{out_root.name}-{datetime.now():%Y%m%d}.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    log = logging.getLogger("classify_signal")
    tasks = [t.strip() for t in (REPO / cfg["tasks"]).read_text(encoding="utf-8").splitlines() if t.strip()]
    fields = ["page", "relative_path", "label", "leads_ok"] + tasks
    model = load_model(paths.resolve(cfg["weights"]), len(tasks), cfg["device"])
    for dataset, run in cfg["runs"].items():
        out = out_root / dataset / "predictions.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            with out.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(fields)
        with out.open(encoding="utf-8") as fh:
            done = {r["page"] for r in csv.DictReader(fh)}
        todo = [p.parent for p in sorted(paths.resolve(run).glob("*/record.json")) if p.parent.name not in done]
        labels = {}
        for start in tqdm(range(0, len(todo), cfg["batch_size"]), desc=dataset, unit="batch"):
            batch = todo[start:start + cfg["batch_size"]]
            with torch.no_grad():
                probs = torch.sigmoid(model(torch.from_numpy(np.stack([read_record(p, cfg["n_samples"], cfg["fill"]) for p in batch])).to(cfg["device"]))).cpu().numpy()
            rows = []
            for page_dir, p in zip(batch, probs):
                image = json.loads((page_dir / "record.json").read_text(encoding="utf-8"))["image"]
                folder, rel = source_key(image["source"])
                if folder not in labels:
                    labels[folder] = load_labels(folder, cfg["label_column"])
                rows.append([page_dir.name, rel, labels[folder].get(rel, ""), image["leads_ok"]] + [f"{v:.5f}" for v in p])
            with out.open("a", newline="", encoding="utf-8") as fh:  # written after the batch is scored, so a crash repeats at most one batch
                csv.writer(fh).writerows(rows)
            log.info("%s: %d pages scored", dataset, len(rows))
        with out.open(encoding="utf-8") as fh:
            print(f"\n{dataset}\n{summary(list(csv.DictReader(fh)), tasks, cfg['top_k'])}")


if __name__ == "__main__":
    main()
