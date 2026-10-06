"""Linear probe on frozen embeddings of a pretrained ECG encoder that has no label head.

1. Embeds every digitized page (<run>/<page>/record.csv) with the frozen encoder and saves
   <out_dir>/<dataset>/embeddings.npz (pages, relative paths, embedding matrix). Resumable: an existing file is reused.
2. For the labeled dataset, fits an L2 logistic regression on standardized embeddings for PJB vs NORMAL and for every
   detailed defect and ACC-CHD group with at least `min_pages` pages against NORMAL. Folds hold out one collection batch
   at a time, so a page is never scored by a probe that saw its batch; the AUROC is computed on the pooled out-of-fold scores.
   Saves the out-of-fold score of every page (probe_scores.csv), the AUROC table (probe_auroc.csv), and for every other
   dataset the scores of a probe fit on all labeled pages (<out_dir>/<dataset>/probe_scores.csv).

Encoders, each with its upstream preprocessing:
- `ecg_jepa` (ECG-JEPA ecg_data.py and models.load_encoder): raw mV, leads I, II, V1 to V6, resampled to 2500 samples
  (10 s at 250 Hz), mean of the final token features.

Usage:
    uv run python -m src.probe --config configs/probe_ecg_jepa.yml
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
from scipy.signal import resample
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src import paths
from src.classify_signal import FS, fit_length, read_record
from src.records import source_key

REPO = Path(__file__).resolve().parents[1]


class ECGJEPA:
    def __init__(self, cfg: dict):
        sys.path.insert(0, str(REPO / "third_party" / "ecg_jepa"))
        from ecg_jepa import ecg_jepa
        params = {"encoder_embed_dim": 768, "encoder_depth": 12, "encoder_num_heads": 16, "predictor_embed_dim": 384, "predictor_depth": 6,
                  "predictor_num_heads": 12, "c": 8, "pos_type": "sincos", "mask_scale": (0, 0), "leads": list(range(8))}  # upstream models.load_encoder
        self.model = ecg_jepa(**params).encoder
        self.model.load_state_dict(torch.load(paths.resolve(cfg["weights"]), map_location="cpu", weights_only=True)["encoder"], strict=True)

    def prepare(self, x: np.ndarray, fill: str) -> np.ndarray:
        x = np.nan_to_num(fit_length(x, 10 * FS, fill), nan=0.0)[[0, 1, 6, 7, 8, 9, 10, 11]]
        return resample(x, 2500, axis=1).astype(np.float32)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model.representation(batch)


MODELS = {"ecg_jepa": ECGJEPA}


def embed(cfg: dict, model, run: Path, out: Path) -> dict:
    if out.exists():
        return dict(np.load(out, allow_pickle=False))
    pages = sorted(p.parent for p in run.glob("*/record.json"))
    feats = []
    for start in tqdm(range(0, len(pages), cfg["batch_size"]), desc=out.parent.name, unit="batch"):
        x = torch.from_numpy(np.stack([model.prepare(read_record(p), cfg["fill"]) for p in pages[start:start + cfg["batch_size"]]])).to(cfg["device"])
        with torch.no_grad():
            feats.append(model(x).cpu().numpy())
    rel = [source_key(json.loads((p / "record.json").read_text(encoding="utf-8"))["image"]["source"])[1] for p in pages]
    data = {"page": np.array([p.name for p in pages]), "relative_path": np.array(rel), "x": np.concatenate(feats)}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **data)
    return data


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    d = score[positive][:, None] - score[~positive][None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def probe(embedded: dict, cfg: dict, out_root: Path) -> None:
    """Writes <out_root>/<labeled dataset>/probe_scores.csv (out-of-fold score per page and target, empty where the page is in
    neither class of that target), probe_auroc.csv, and for every other dataset probe_scores.csv from a probe fit on all labeled pages."""
    p = cfg["probe"]
    data = embedded[p["dataset"]]
    with (paths.dataset(p["dataset"]) / "_labels" / "manifest.csv").open(encoding="utf-8") as fh:
        manifest = {r["relative_path"]: r for r in csv.DictReader(fh)}
    groups = yaml.safe_load((REPO / p["groups_config"]).read_text(encoding="utf-8"))["groups"]
    rows = [manifest[r] for r in data["relative_path"]]
    detailed = [set(json.loads(r[p["detailed_column"]] or "[]")) for r in rows]
    batch = np.array([p["batches"][r.split("/")[0]] for r in data["relative_path"]])
    normal = np.array([r[p["label_column"]] == "NORMAL" for r in rows])
    targets = {"PJB": ~normal}
    targets |= {d: np.array([d in s for s in detailed]) for d in sorted({d for s in detailed for d in s} - {"NORMAL"})}
    targets |= {f"group {g}": np.array([any(groups[d] == g for d in s) for s in detailed]) for g in sorted(set(groups.values()) - {"normal"})}
    x = data["x"]
    print(f"{len(rows)} pages, {normal.sum()} NORMAL, embedding size {x.shape[1]}, folds hold out one of {sorted(set(batch))}\n")
    print("| Target | Pages | Out-of-fold AUROC against NORMAL |\n|---|---|---|")
    scored = [n for n, has in sorted(targets.items(), key=lambda t: -t[1].sum()) if has.sum() >= p["min_pages"]]
    oof = {n: np.full(len(rows), np.nan) for n in scored}
    other = {d: {} for d in embedded if d != p["dataset"]}
    metrics = []
    for name in scored:
        has = targets[name]
        keep = has | normal
        xs, ys, bs = x[keep], has[keep], batch[keep]
        score = np.full(len(ys), np.nan)
        for b in sorted(set(bs)):
            train = bs != b
            if len(set(ys[train])) < 2:
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=5000))
            score[~train] = clf.fit(xs[train], ys[train]).predict_proba(xs[~train])[:, 1]
        oof[name][keep] = score
        ok = ~np.isnan(score)
        metrics.append({"target": name, "pages": int(has.sum()), "normal_pages": int(normal.sum()), "oof_auroc": round(auroc(score[ok], ys[ok]), 4)})
        print(f"| {name} | {has.sum()} | {metrics[-1]['oof_auroc']:.4f} |")
        full = make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=5000)).fit(xs, ys)
        for d in other:
            other[d][name] = full.predict_proba(embedded[d]["x"])[:, 1]
    out = out_root / p["dataset"]
    with (out / "probe_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["page", "relative_path", "label"] + scored)
        for i, r in enumerate(rows):
            w.writerow([data["page"][i], data["relative_path"][i], r[p["label_column"]]] + ["" if np.isnan(oof[n][i]) else f"{oof[n][i]:.5f}" for n in scored])
    with (out / "probe_auroc.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(metrics[0]))
        w.writeheader()
        w.writerows(metrics)
    for d, scores in other.items():
        with (out_root / d / "probe_scores.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["page", "relative_path"] + scored)
            for i in range(len(embedded[d]["page"])):
                w.writerow([embedded[d]["page"][i], embedded[d]["relative_path"][i]] + [f"{scores[n][i]:.5f}" for n in scored])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text(encoding="utf-8"))
    out_root = paths.resolve(cfg["out_dir"])
    log_dir = paths.resolve(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / f"{out_root.name}-{datetime.now():%Y%m%d}.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    model = MODELS[cfg["model"]](cfg)
    model.model.to(cfg["device"]).eval()
    embedded = {d: embed(cfg, model, paths.resolve(run), out_root / d / "embeddings.npz") for d, run in cfg["runs"].items()}
    logging.getLogger("probe").info("embedded %s", {d: len(e["page"]) for d, e in embedded.items()})
    probe(embedded, cfg, out_root)


if __name__ == "__main__":
    main()
