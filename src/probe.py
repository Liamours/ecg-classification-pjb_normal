"""Linear probe on the saved embeddings of a pretrained ECG encoder that has no label head (written by `src.infer`).

Fits an L2 logistic regression on standardized embeddings for PJB vs NORMAL and for every detailed defect and ACC-CHD group
with at least `min_pages` pages against NORMAL. Folds hold out one collection batch at a time, so a page is never scored by a
probe that saw its batch; the AUROC is computed on the pooled out-of-fold scores. Saves to <out_dir>/<dataset>/:
probe_scores.csv (out-of-fold score per page and target on the labeled set, empty where the page is in neither class; on
other sets the score of a probe fit on all labeled pages) and probe_auroc.csv.

Usage:
    uv run python -m src.probe --config configs/probe/ecg_jepa_multiblock.yml
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import paths

REPO = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    cols = [c for c in rows[0] if c.startswith("emb_")]
    return {"page": [r["page"] for r in rows], "relative_path": [r["relative_path"] for r in rows], "x": np.array([[float(r[c]) for c in cols] for r in rows])}


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    d = score[positive][:, None] - score[~positive][None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def fit(x: np.ndarray, y: np.ndarray, c: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)).fit(x, y)


def write(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text(encoding="utf-8"))
    emb_root, out_root = paths.resolve(cfg["embeddings"]), paths.resolve(cfg["out_dir"])
    embedded = {d.name: load(d / "embeddings.csv") for d in sorted(emb_root.iterdir()) if (d / "embeddings.csv").exists()}
    data = embedded[cfg["dataset"]]
    with (paths.dataset(cfg["dataset"]) / "_labels" / "manifest.csv").open(encoding="utf-8") as fh:
        manifest = {r["relative_path"]: r for r in csv.DictReader(fh)}
    groups = yaml.safe_load((REPO / cfg["groups_config"]).read_text(encoding="utf-8"))["groups"]
    rows = [manifest[r] for r in data["relative_path"]]
    detailed = [set(json.loads(r[cfg["detailed_column"]] or "[]")) for r in rows]
    batch = np.array([cfg["batches"][r.split("/")[0]] for r in data["relative_path"]])
    normal = np.array([r[cfg["label_column"]] == "NORMAL" for r in rows])
    targets = {"PJB": ~normal}
    targets |= {d: np.array([d in s for s in detailed]) for d in sorted({d for s in detailed for d in s} - {"NORMAL"})}
    targets |= {f"group {g}": np.array([any(groups[d] == g for d in s) for s in detailed]) for g in sorted(set(groups.values()) - {"normal"})}
    scored = [n for n, has in sorted(targets.items(), key=lambda t: -t[1].sum()) if has.sum() >= cfg["min_pages"]]
    x = data["x"]
    print(f"{len(rows)} pages, {normal.sum()} NORMAL, embedding size {x.shape[1]}, folds hold out one of {sorted(set(batch))}\n")
    print("| Target | Pages | Out-of-fold AUROC against NORMAL |\n|---|---|---|")
    oof = {n: np.full(len(rows), np.nan) for n in scored}
    other = {d: {} for d in embedded if d != cfg["dataset"]}
    metrics = []
    for name in scored:
        keep = targets[name] | normal
        xs, ys, bs = x[keep], targets[name][keep], batch[keep]
        score = np.full(len(ys), np.nan)
        for b in sorted(set(bs)):
            train = bs != b
            if len(set(ys[train])) == 2:
                score[~train] = fit(xs[train], ys[train], cfg["C"]).predict_proba(xs[~train])[:, 1]
        oof[name][keep] = score
        ok = ~np.isnan(score)
        metrics.append([name, int(targets[name].sum()), int(normal.sum()), f"{auroc(score[ok], ys[ok]):.4f}"])
        print(f"| {name} | {metrics[-1][1]} | {metrics[-1][3]} |")
        full = fit(xs, ys, cfg["C"])
        for d in other:
            other[d][name] = full.predict_proba(embedded[d]["x"])[:, 1]
    write(out_root / cfg["dataset"] / "probe_auroc.csv", ["target", "pages", "normal_pages", "oof_auroc"], metrics)
    write(out_root / cfg["dataset"] / "probe_scores.csv", ["page", "relative_path", "label"] + scored,
          [[data["page"][i], data["relative_path"][i], r[cfg["label_column"]]] + ["" if np.isnan(oof[n][i]) else f"{oof[n][i]:.5f}" for n in scored] for i, r in enumerate(rows)])
    for d, scores in other.items():
        write(out_root / d / "probe_scores.csv", ["page", "relative_path"] + scored,
              [[embedded[d]["page"][i], embedded[d]["relative_path"][i]] + [f"{scores[n][i]:.5f}" for n in scored] for i in range(len(embedded[d]["page"]))])


if __name__ == "__main__":
    main()
