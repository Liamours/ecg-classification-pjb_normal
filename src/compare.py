"""Early comparison of pretrained models on our multi-label CHD labels, one protocol for every model.

Features: each model run's saved per-page outputs (`predictions.csv` label outputs or `embeddings.csv`). Labels: two
multi-label schemas on the labeled phone photos, `detailed` (every defect with at least `min_pages` pages, plus CHD = any
defect) and `acc_chd` (ACC-CHD groups, plus CHD); NORMAL pages carry no label. Each label is scored against all other pages.
Classifier: one L2 logistic regression per label on standardized features (balanced class weights, threshold 0.5); folds hold
out one collection batch at a time, so every score is out of fold. Writes to `out_dir`:
- `<schema>/scores-<model>.csv`: out-of-fold probability and 0/1 prediction per page and label
- `<schema>/per_label.csv`: AUROC, AUPRC, F1, sensitivity, specificity, precision, balanced accuracy per model and label
- `<schema>/summary.csv`: micro and macro F1, macro AUROC and AUPRC, Hamming loss, exact-set accuracy, mean Jaccard per model
- `<schema>/disagreement.csv`: per model pair, mean Cohen's kappa over labels, share of pages whose predicted label sets differ
- `label_fit.csv`: per model, native outputs and how many name a pediatric ECG finding tied to congenital heart defects

Usage:
    uv run python -m src.compare --config configs/compare.yml
"""
import argparse
import csv
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, cohen_kappa_score, f1_score, hamming_loss, jaccard_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src import paths

REPO = Path(__file__).resolve().parents[1]
META = {"page", "relative_path", "label", "leads_ok"}


def load_features(run_dir: Path) -> tuple[list[str], list[str], np.ndarray, bool]:
    path = run_dir / "predictions.csv"
    native = path.exists()
    with (path if native else run_dir / "embeddings.csv").open(encoding="utf-8") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: r["relative_path"])
    cols = [c for c in rows[0] if c not in META]
    return [r["relative_path"] for r in rows], cols, np.array([[float(r[c]) for c in cols] for r in rows]), native


def label_matrix(cfg: dict, pages: list[str]) -> dict[str, tuple[list[str], np.ndarray]]:
    with (paths.dataset(cfg["dataset"]) / "_labels" / "manifest.csv").open(encoding="utf-8") as fh:
        manifest = {r["relative_path"]: r for r in csv.DictReader(fh)}
    groups = yaml.safe_load((REPO / cfg["groups_config"]).read_text(encoding="utf-8"))["groups"]
    sets = [set(json.loads(manifest[p][cfg["detailed_column"]] or "[]")) - {"NORMAL"} for p in pages]
    chd = np.array([manifest[p][cfg["label_column"]] == "PJB" for p in pages])
    schemas = {}
    for name, mapper in [("detailed", lambda d: d), ("acc_chd", lambda d: groups[d])]:
        mapped = [{mapper(d) for d in s} - {"other", "normal"} for s in sets]
        labels = sorted({m for s in mapped for m in s}, key=lambda m: -sum(m in s for s in mapped))
        labels = [m for m in labels if sum(m in s for s in mapped) >= cfg["min_pages"]]
        y = np.column_stack([chd] + [[m in s for s in mapped] for m in labels])
        schemas[name] = (["CHD"] + labels, y)
    return schemas


def out_of_fold(x: np.ndarray, y: np.ndarray, batch: np.ndarray, cfg: dict) -> np.ndarray:
    prob = np.zeros(y.shape)
    for b in sorted(set(batch)):
        test = batch == b
        for j in range(y.shape[1]):
            if len(set(y[~test, j])) < 2:
                prob[test, j] = y[~test, j].mean()
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=cfg["C"], class_weight=cfg["class_weight"], max_iter=5000))
            prob[test, j] = clf.fit(x[~test], y[~test, j]).predict_proba(x[test])[:, 1]
    return prob


def per_label(y: np.ndarray, prob: np.ndarray, pred: np.ndarray, j: int) -> dict:
    t, p, s = y[:, j], pred[:, j], prob[:, j]
    tp, fp, fn, tn = (t & p).sum(), (~t & p).sum(), (t & ~p).sum(), (~t & ~p).sum()
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return {"support": int(t.sum()), "auroc": roc_auc_score(t, s), "auprc": average_precision_score(t, s), "prevalence": t.mean(),
            "f1": f1_score(t, p, zero_division=0), "sensitivity": sens, "specificity": spec, "precision": tp / max(tp + fp, 1), "balanced_accuracy": (sens + spec) / 2}


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows([{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()} for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text(encoding="utf-8"))
    out = paths.resolve(cfg["out_dir"])
    feats, fit = {}, []
    for model, run in cfg["runs"].items():
        pages, cols, x, native = load_features(paths.resolve("@inferences") / run / cfg["dataset"])
        feats[model] = (pages, x)
        matched = {f: [c for c in cols if re.search(rx, c, re.I)] for f, rx in cfg["chd_findings"].items()} if native else {}
        fit.append({"model": model, "native_outputs": len(cols) if native else 0, "embedding_size": 0 if native else len(cols),
                    "outputs_naming_a_chd_finding": sum(len(v) for v in matched.values()), "findings_covered": sum(bool(v) for v in matched.values()),
                    "outputs": "; ".join(f"{f}: {', '.join(v)}" for f, v in matched.items() if v)})
    write(out / "label_fit.csv", fit)
    pages = feats[next(iter(feats))][0]
    assert all(f[0] == pages for f in feats.values()), "runs cover different pages"
    batch = np.array([cfg["batches"][p.split("/")[0]] for p in pages])
    for schema, (labels, y) in label_matrix(cfg, pages).items():
        preds, rows, summary = {}, [], []
        for model, (_, x) in tqdm(feats.items(), desc=schema):
            prob = out_of_fold(x, y, batch, cfg)
            pred = prob >= cfg["threshold"]
            preds[model] = pred
            write(out / schema / f"scores-{model}.csv", [{"relative_path": p, **{f"{l} prob": float(prob[i, j]) for j, l in enumerate(labels)},
                                                         **{f"{l} pred": int(pred[i, j]) for j, l in enumerate(labels)}} for i, p in enumerate(pages)])
            label_rows = [{"model": model, "label": l, **per_label(y, prob, pred, j)} for j, l in enumerate(labels)]
            rows += label_rows
            summary.append({"model": model, "labels": len(labels), "macro_auroc": float(np.mean([r["auroc"] for r in label_rows])),
                            "macro_auprc": float(np.mean([r["auprc"] for r in label_rows])), "micro_f1": f1_score(y, pred, average="micro", zero_division=0),
                            "macro_f1": f1_score(y, pred, average="macro", zero_division=0), "hamming_loss": hamming_loss(y, pred),
                            "exact_set_accuracy": float((pred == y).all(1).mean()), "mean_jaccard": jaccard_score(y, pred, average="samples", zero_division=1)})
        write(out / schema / "per_label.csv", rows)
        write(out / schema / "summary.csv", summary)
        dis = []
        for a, b in combinations(preds, 2):
            kappas = [cohen_kappa_score(preds[a][:, j], preds[b][:, j]) for j in range(len(labels))]
            dis.append({"model_a": a, "model_b": b, "mean_kappa": float(np.nanmean(kappas)), "pages_with_different_label_sets": float((preds[a] != preds[b]).any(1).mean()),
                        "label_disagreement_rate": float((preds[a] != preds[b]).mean())})
        allp = np.stack(list(preds.values()))
        dis.append({"model_a": "all", "model_b": "all", "mean_kappa": float("nan"), "pages_with_different_label_sets": float((allp != allp[0]).any((0, 2)).mean()),
                    "label_disagreement_rate": float((allp != allp[0]).any(0).mean())})
        write(out / schema / "disagreement.csv", dis)
        print(f"\n{schema}: {len(pages)} pages, labels {labels}")
        print("| Model | Macro AUROC | Macro AUPRC | Micro F1 | Macro F1 | Hamming loss | Exact-set accuracy | Mean Jaccard |\n|---|---|---|---|---|---|---|---|")
        for s in summary:
            print(f"| {s['model']} | {s['macro_auroc']:.4f} | {s['macro_auprc']:.4f} | {s['micro_f1']:.4f} | {s['macro_f1']:.4f} | {s['hamming_loss']:.4f} | {s['exact_set_accuracy']:.4f} | {s['mean_jaccard']:.4f} |")


if __name__ == "__main__":
    main()
