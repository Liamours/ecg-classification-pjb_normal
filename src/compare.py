"""Comparison of pretrained models on the chosen label schema, one protocol for every model.

Schema (user, 2026-10-06): hierarchical multi-label, two levels: the ACC-CHD group and the diagnosis under it (variant names
merged into their parent diagnosis, `merge`). NORMAL is no label: a page with no output above `threshold` is NORMAL; there is no
separate CHD or NORMAL label. A PJB page with no diagnosis in its file name gets `unknown_label` at both levels.
Features: each model run's saved per-page outputs (`predictions.csv` label outputs or `embeddings.csv`). Classifier: one L2
logistic regression per label on standardized features, balanced class weights, fit on the train and val pages of a fold and
scored on its test pages; folds from `folds` (nested cross-validation manifest of `src.split`), every repeat. Writes to `out_dir`:
- `scores-<model>-r<repeat>.csv`: out-of-fold probability and 0/1 prediction per page and label
- `per_label.csv`: per model, repeat and label: AUROC, AUPRC, F1, sensitivity, specificity, precision, balanced accuracy
- `summary.csv`: per model and repeat, per level and overall: macro AUROC and AUPRC over labels with 10+ pages, micro and macro F1,
  Hamming loss, exact-set accuracy, mean Jaccard, share of pages whose predicted diagnosis lacks its predicted group
- `disagreement.csv`: per model pair (repeat 1), mean Cohen's kappa over labels, share of pages whose predicted label sets differ
- `label_fit.csv`: per model, native outputs and how many name a pediatric ECG finding tied to congenital heart defects

Usage:
    uv run python -m src.compare --config configs/compare.yml
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, cohen_kappa_score, f1_score, hamming_loss, roc_auc_score
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


def label_matrix(cfg: dict, pages: list[str]) -> tuple[list[str], list[str], np.ndarray, dict[str, str]]:
    """Labels, their level (group or diagnosis), pages x labels, and the group of each diagnosis label."""
    with (paths.dataset(cfg["dataset"]) / "_labels" / "manifest.csv").open(encoding="utf-8") as fh:
        manifest = {r["relative_path"]: r for r in csv.DictReader(fh)}
    groups = yaml.safe_load((REPO / cfg["groups_config"]).read_text(encoding="utf-8"))["groups"]
    parent, sets = {}, []
    for p in pages:
        ds = [d for d in json.loads(manifest[p][cfg["detailed_column"]] or "[]") if d != "NORMAL"]
        chd = manifest[p][cfg["label_column"]] == "PJB"
        if chd and not ds:
            sets.append({f"group: {cfg['unknown_label']}", f"diagnosis: {cfg['unknown_label']}"})
            parent[f"diagnosis: {cfg['unknown_label']}"] = f"group: {cfg['unknown_label']}"
            continue
        s = set()
        for d in ds:
            dx, gr = f"diagnosis: {cfg['merge'].get(d, d)}", f"group: {groups[d]}"
            s |= {dx, gr}
            parent[dx] = gr
        sets.append(s)
    labels = sorted({m for s in sets for m in s}, key=lambda m: (m.startswith("diagnosis"), -sum(m in s for s in sets), m))
    y = np.column_stack([[m in s for s in sets] for m in labels])
    normal = np.array([manifest[p][cfg["label_column"]] == "NORMAL" for p in pages])
    assert (y.any(1) == ~normal).all(), "a NORMAL page has a label or a PJB page has none"
    return labels, [m.split(":")[0] for m in labels], y, parent


def folds(cfg: dict, pages: list[str]) -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
    """Per repeat, per outer fold: (train mask = train and val pages, test mask)."""
    idx = {p: i for i, p in enumerate(pages)}
    role = defaultdict(lambda: np.array([""] * len(pages), dtype=object))
    with paths.resolve(cfg["folds"]).open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["relative_path"] in idx:
                role[(int(r["repeat"]), int(r["fold"]))][idx[r["relative_path"]]] = r["role"]
    out = defaultdict(list)
    for (rep, fold), roles in sorted(role.items()):
        out[rep].append((np.isin(roles, ["train", "val"]), roles == "test"))
    return out


def out_of_fold(x: np.ndarray, y: np.ndarray, splits: list, cfg: dict) -> np.ndarray:
    prob = np.full(y.shape, np.nan)
    for train, test in splits:
        for j in range(y.shape[1]):
            if len(set(y[train, j])) < 2:
                prob[test, j] = y[train, j].mean()
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=cfg["C"], class_weight=cfg["class_weight"], max_iter=5000))
            prob[test, j] = clf.fit(x[train], y[train, j]).predict_proba(x[test])[:, 1]
    return prob


def per_label(y: np.ndarray, prob: np.ndarray, pred: np.ndarray, j: int) -> dict:
    t, p, s = y[:, j], pred[:, j], prob[:, j]
    tp, fp, fn, tn = (t & p).sum(), (~t & p).sum(), (t & ~p).sum(), (~t & ~p).sum()
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return {"support": int(t.sum()), "auroc": roc_auc_score(t, s), "auprc": average_precision_score(t, s), "prevalence": t.mean(),
            "f1": f1_score(t, p, zero_division=0), "sensitivity": sens, "specificity": spec, "precision": tp / max(tp + fp, 1), "balanced_accuracy": (sens + spec) / 2}


def set_metrics(y: np.ndarray, pred: np.ndarray, rows: list[dict], min_pages: int) -> dict:
    big = [r for r in rows if r["support"] >= min_pages]  # a label with fewer pages often has none in a training fold, which then scores it with a constant
    return {"labels": len(rows), "labels_10plus": len(big), "macro_auroc_10plus": float(np.mean([r["auroc"] for r in big])),
            "macro_auprc_10plus": float(np.mean([r["auprc"] for r in big])), "micro_f1": f1_score(y, pred, average="micro", zero_division=0),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0), "hamming_loss": hamming_loss(y, pred), "exact_set_accuracy": float((pred == y).all(1).mean()),
            "mean_jaccard": float(np.mean([(t & q).sum() / (t | q).sum() if (t | q).any() else 1.0 for t, q in zip(y, pred)]))}  # empty true and predicted sets (NORMAL kept NORMAL) score 1


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
    labels, level, y, parent = label_matrix(cfg, pages)
    level = np.array(level)
    child = [j for j, l in enumerate(labels) if l in parent]
    splits = folds(cfg, pages)
    print(f"{len(pages)} pages; {len(labels)} labels ({(level == 'group').sum()} groups, {(level == 'diagnosis').sum()} diagnoses); repeats {sorted(splits)}")
    rows, summary, preds = [], [], {}
    for model, (_, x) in tqdm(feats.items(), desc="models"):
        for rep, sp in splits.items():
            prob = out_of_fold(x, y, sp, cfg)
            pred = prob >= cfg["threshold"]
            if rep == 1:
                preds[model] = pred
            write(out / f"scores-{model}-r{rep}.csv", [{"relative_path": p, **{f"{l} prob": float(prob[i, j]) for j, l in enumerate(labels)},
                                                       **{f"{l} pred": int(pred[i, j]) for j, l in enumerate(labels)}} for i, p in enumerate(pages)])
            label_rows = [{"model": model, "repeat": rep, "label": l, "level": level[j], **per_label(y, prob, pred, j)} for j, l in enumerate(labels)]
            rows += label_rows
            orphan = float(np.mean([any(pred[i, j] and not pred[i, labels.index(parent[labels[j]])] for j in child) for i in range(len(pages))]))
            for name, cols in (("overall", np.arange(len(labels))), ("group", np.flatnonzero(level == "group")), ("diagnosis", np.flatnonzero(level == "diagnosis"))):
                summary.append({"model": model, "repeat": rep, "level": name, **set_metrics(y[:, cols], pred[:, cols], [label_rows[c] for c in cols], cfg["min_pages"]),
                                "pages_with_diagnosis_without_its_group": orphan})
    write(out / "per_label.csv", rows)
    write(out / "summary.csv", summary)
    dis = [{"model_a": a, "model_b": b, "mean_kappa": float(np.nanmean([cohen_kappa_score(preds[a][:, j], preds[b][:, j]) for j in range(len(labels))])),
            "pages_with_different_label_sets": float((preds[a] != preds[b]).any(1).mean()), "label_disagreement_rate": float((preds[a] != preds[b]).mean())}
           for a, b in combinations(preds, 2)]
    write(out / "disagreement.csv", dis)
    keys = ["micro_f1", "macro_f1", "exact_set_accuracy", "hamming_loss", "mean_jaccard"]  # F1 leads the report (user, 2026-10-06); AUROC stays in the CSV files
    print("| Model | Level | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 2))
    for model in feats:
        for name in ("overall", "group", "diagnosis"):
            r = [s for s in summary if s["model"] == model and s["level"] == name]
            print(f"| {model} | {name} | " + " | ".join(f"{np.mean([s[k] for s in r]):.4f} ± {np.std([s[k] for s in r]):.4f}" for k in keys) + " |")


if __name__ == "__main__":
    main()
