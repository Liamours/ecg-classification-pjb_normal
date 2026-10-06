"""Linear probe of frozen ECG model embeddings on the hierarchical multi-label CHD labels, with the nested cross-validation folds.

Labels (user, 2026-10-06): ACC-CHD group and diagnosis (variant names merged), NORMAL = no label above its threshold.
Per model, repeat and outer fold (one resumable unit): standardize on the train pages; per label fit an L2 logistic regression
on train for every value of `C_grid` (warm-started from the previous one), keep the value with the best average precision on
val; one threshold per level chosen on val by micro F1, a diagnosis kept only when its group is predicted (`src.thresholds`);
then score the test pages once with the train-only fit. Per model (one more unit): a probe on every labeled page, with each
label's most often chosen C and the median thresholds per level, scores the unlabeled sets in `predict`.

Writes to `out_dir/<model>/`: `parts/r<repeat>_f<fold>.csv` (test-page probabilities and 0/1 predictions) and
`parts/r<repeat>_f<fold>_settings.json` (thresholds per level, C per label), `oof_r<repeat>.csv` (every labeled page's
out-of-fold predictions), `<dataset>_predictions.csv` per unlabeled set; to `report_dir`: `per_label.csv`, `summary.csv`
(per model and repeat, F1 by level: micro and macro F1, precision, sensitivity, exact-set accuracy, Hamming loss) and
`normal_vs_pjb.csv` (NORMAL = no label predicted: F1, precision, sensitivity per class, macro F1, counts).
One progress line with ETA, a log in `log_dir`; finished units are skipped on a rerun.

Usage:
    uv run python -m src.linear_probe --config configs/linear_probe.yml
"""
import argparse
import csv
import json
import logging
import os
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, hamming_loss
from sklearn.preprocessing import StandardScaler

from src import paths
from src.progress import Progress
from src.thresholds import choose, parents, predict

REPO = Path(__file__).resolve().parents[1]
META = {"page", "relative_path", "label", "leads_ok"}


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(encoding="utf-8") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: r["relative_path"])
    cols = [c for c in rows[0] if c not in META]
    return [r["relative_path"] for r in rows], np.array([[float(r[c]) for c in cols] for r in rows], dtype=np.float32)


def label_matrix(cfg: dict, pages: list[str]) -> tuple[list[str], np.ndarray]:
    with (paths.dataset(cfg["dataset"]) / "_labels" / "manifest.csv").open(encoding="utf-8") as fh:
        manifest = {r["relative_path"]: r for r in csv.DictReader(fh)}
    groups = yaml.safe_load((REPO / cfg["groups_config"]).read_text(encoding="utf-8"))["groups"]
    sets = []
    for p in pages:
        ds = [d for d in json.loads(manifest[p][cfg["detailed_column"]] or "[]") if d != "NORMAL"]
        if manifest[p][cfg["label_column"]] == "PJB" and not ds:
            sets.append({f"group: {cfg['unknown_label']}", f"diagnosis: {cfg['unknown_label']}"})
        else:
            sets.append({f"diagnosis: {cfg['merge'].get(d, d)}" for d in ds} | {f"group: {groups[d]}" for d in ds})
    labels = sorted({m for s in sets for m in s}, key=lambda m: (m.startswith("diagnosis"), -sum(m in s for s in sets), m))
    y = np.column_stack([[m in s for s in sets] for m in labels])
    assert (y.any(1) == np.array([manifest[p][cfg["label_column"]] != "NORMAL" for p in pages])).all(), "NORMAL must be the empty label set"
    return labels, y


def load_folds(cfg: dict, pages: list[str]) -> dict[tuple[int, int], np.ndarray]:
    idx = {p: i for i, p in enumerate(pages)}
    roles = defaultdict(lambda: np.full(len(pages), "", dtype=object))
    with paths.resolve(cfg["folds"]).open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["relative_path"] in idx:
                roles[(int(r["repeat"]), int(r["fold"]))][idx[r["relative_path"]]] = r["role"]
    return dict(sorted(roles.items()))


def fit_label(xtr, ytr, xva, yva, cfg: dict) -> tuple[float, LogisticRegression | None]:
    """Best C on val by average precision; (C, train-only model)."""
    if ytr.sum() == 0 or ytr.all():
        return float("nan"), None
    warnings.simplefilter("ignore", ConvergenceWarning)
    clf = LogisticRegression(class_weight=cfg["class_weight"], max_iter=cfg["max_iter"], warm_start=True)
    best = (-1.0, None, None)
    for c in cfg["C_grid"]:
        clf.set_params(C=c)
        clf.fit(xtr, ytr)
        score = average_precision_score(yva, clf.predict_proba(xva)[:, 1]) if yva.any() else -1.0
        if score > best[0]:
            best = (score, c, (clf.coef_.copy(), clf.intercept_.copy()))
    c = best[1] if best[1] is not None else 1.0
    clf.set_params(C=c)
    if best[2] is not None:
        clf.coef_, clf.intercept_ = best[2]
    else:
        clf.fit(xtr, ytr)
    return c, clf


def run_fold(x, y, role, parent, cfg) -> tuple[np.ndarray, list[float], dict]:
    """Per label C on val; thresholds per level on val (`src.thresholds`); (test probabilities, C per label, thresholds)."""
    tr, va, te = role == "train", role == "val", role == "test"
    scaler = StandardScaler().fit(x[tr])
    xtr, xva, xte = scaler.transform(x[tr]), scaler.transform(x[va]), scaler.transform(x[te])
    fits = Parallel(n_jobs=cfg["jobs"])(delayed(fit_label)(xtr, y[tr, j], xva, y[va, j], cfg) for j in range(y.shape[1]))
    score = lambda xs: np.column_stack([f[1].predict_proba(xs)[:, 1] if f[1] is not None else np.zeros(len(xs)) for f in fits])
    return score(xte), [f[0] for f in fits], choose(score(xva), y[va], parent)


def final_probe(x, y, cs: list[list[float]], cfg) -> tuple[StandardScaler, list]:
    """One probe on every labeled page per label, with the most often chosen C over all folds."""
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    models = []
    for j in range(y.shape[1]):
        chosen = [c[j] for c in cs if not np.isnan(c[j])]
        if not chosen or y[:, j].sum() == 0:
            models.append(None)
            continue
        models.append(LogisticRegression(C=Counter(chosen).most_common(1)[0][0], class_weight=cfg["class_weight"], max_iter=cfg["max_iter"]).fit(xs, y[:, j]))
    return scaler, models


def write_csv(path: Path, header: list[str], rows) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)  # the finished file appears in one step, so its existence marks the unit done


def read_part(path: Path, labels: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r["relative_path"] for r in rows], np.array([[float(r[f"{l} prob"]) for l in labels] for r in rows]), np.array([[int(r[f"{l} pred"]) for l in labels] for r in rows], bool)


def metrics(y: np.ndarray, pred: np.ndarray, labels: list[str]) -> tuple[list[dict], dict]:
    rows = []
    for j, l in enumerate(labels):
        t, p = y[:, j], pred[:, j]
        tp, fp, fn, tn = (t & p).sum(), (~t & p).sum(), (t & ~p).sum(), (~t & ~p).sum()
        rows.append({"label": l, "level": l.split(":")[0], "support": int(t.sum()), "f1": f1_score(t, p, zero_division=0), "precision": tp / max(tp + fp, 1),
                     "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1)})
    out = {}
    for name, cols in (("overall", range(len(labels))), ("group", [j for j, l in enumerate(labels) if l.startswith("group")]), ("diagnosis", [j for j, l in enumerate(labels) if l.startswith("diagnosis")])):
        cols = list(cols)
        yt, yp = y[:, cols], pred[:, cols]
        tp, fp, fn = (yt & yp).sum(), (~yt & yp).sum(), (yt & ~yp).sum()
        out[name] = {"micro_f1": f1_score(yt, yp, average="micro", zero_division=0), "macro_f1": f1_score(yt, yp, average="macro", zero_division=0),
                     "micro_precision": tp / max(tp + fp, 1), "micro_sensitivity": tp / max(tp + fn, 1),
                     "exact_set_accuracy": float((yt == yp).all(1).mean()), "hamming_loss": hamming_loss(yt, yp)}
    return rows, out


def normal_vs_pjb(y: np.ndarray, pred: np.ndarray) -> dict:
    """NORMAL vs PJB read from the multi-label output: a page is PJB when it has any label (truth) or any predicted label."""
    t, p = y.any(1), pred.any(1)
    out = {"normal_pages": int((~t).sum()), "pjb_pages": int(t.sum()), "normal_as_normal": int((~t & ~p).sum()), "normal_as_pjb": int((~t & p).sum()),
           "pjb_as_normal": int((t & ~p).sum()), "pjb_as_pjb": int((t & p).sum())}
    for name, tc, pc in (("normal", ~t, ~p), ("pjb", t, p)):
        tp, fp, fn = (tc & pc).sum(), (~tc & pc).sum(), (tc & ~pc).sum()
        out |= {f"{name}_f1": float(2 * tp / max(2 * tp + fp + fn, 1)), f"{name}_precision": float(tp / max(tp + fp, 1)), f"{name}_sensitivity": float(tp / max(tp + fn, 1))}
    out["macro_f1"] = (out["normal_f1"] + out["pjb_f1"]) / 2
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text(encoding="utf-8"))
    out_root, report = paths.resolve(cfg["out_dir"]), paths.resolve(cfg["report_dir"])
    log_dir = paths.resolve(cfg["log_dir"])
    for d in (out_root, report, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / f"linear_probe-{datetime.now():%Y%m%d}.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    log = logging.getLogger("linear_probe")
    data = {}
    for model, run in cfg["models"].items():
        path = paths.resolve("@inferences") / run / cfg["dataset"] / "embeddings.csv"
        if path.exists():
            data[model] = load_embeddings(path)
        else:
            log.info("skip %s: no embeddings at %s", model, path)
            print(f"skip {model}: no embeddings at {path}")
    pages = next(iter(data.values()))[0]
    labels, y = label_matrix(cfg, pages)
    parent = parents(labels, REPO / cfg["groups_config"], cfg["merge"], cfg["unknown_label"])
    folds = load_folds(cfg, pages)
    units = [(m, r, f) for m in data for (r, f) in folds] + [(m, 0, 0) for m in data]
    todo = [u for u in units if not (out_root / u[0] / "parts" / (f"r{u[1]}_f{u[2]}.csv" if u[1] else "final.done")).exists()]
    log.info("start: %d models, %d labels, %d folds, %d of %d units to do, config %s", len(data), len(labels), len(folds), len(todo), len(units), cfg)
    bar = Progress(len(todo), "linear probe", "units", log)
    for model, rep, fold in todo:
        pg, x = data[model]
        assert pg == pages, f"{model} covers other pages"
        part = out_root / model / "parts"
        part.mkdir(parents=True, exist_ok=True)
        if rep:
            role = folds[(rep, fold)]
            prob, cs, t = run_fold(x, y, role, parent, cfg)
            pred = predict(prob, t, parent)
            test = np.flatnonzero(role == "test")
            (part / f"r{rep}_f{fold}_settings.json").write_text(json.dumps({"thresholds": t, "C": dict(zip(labels, cs))}), encoding="utf-8")
            write_csv(part / f"r{rep}_f{fold}.csv", ["relative_path"] + [f"{l} prob" for l in labels] + [f"{l} pred" for l in labels],
                      [[pages[i]] + [f"{v:.5f}" for v in prob[k]] + [int(v) for v in pred[k]] for k, i in enumerate(test)])
        else:
            settings = [json.loads((part / f"r{r}_f{f}_settings.json").read_text(encoding="utf-8")) for (r, f) in folds]
            t = {k: float(np.median([s["thresholds"][k] for s in settings])) for k in ("group", "diagnosis")}
            scaler, models = final_probe(x, y, [[s["C"][l] for l in labels] for s in settings], cfg)
            for ds in cfg["predict"]:
                path = paths.resolve("@inferences") / cfg["models"][model] / ds / "embeddings.csv"
                if not path.exists():
                    continue
                pgs, xs = load_embeddings(path)
                xs = scaler.transform(xs)
                prob = np.column_stack([m.predict_proba(xs)[:, 1] if m is not None else np.zeros(len(pgs)) for m in models])
                pred = predict(prob, t, parent)
                write_csv(out_root / model / f"{ds}_predictions.csv", ["relative_path"] + [f"{l} prob" for l in labels] + [f"{l} pred" for l in labels],
                          [[p] + [f"{v:.5f}" for v in prob[i]] + [int(v) for v in pred[i]] for i, p in enumerate(pgs)])
            write_csv(part / "final.done", ["level", "threshold"], [[k, f"{v:.2f}"] for k, v in t.items()])
        bar.step(1, f"{model} r{rep} f{fold}" if rep else f"{model} final")
    bar.close()
    per_label, summary, binary = [], [], []
    for model in data:
        for rep in sorted({r for r, _ in folds}):
            prob, pred = np.zeros(y.shape), np.zeros(y.shape, bool)
            idx = {p: i for i, p in enumerate(pages)}
            for (r, f) in folds:
                if r == rep:
                    pg, pr, pd = read_part(out_root / model / "parts" / f"r{r}_f{f}.csv", labels)
                    rows_i = [idx[p] for p in pg]
                    prob[rows_i], pred[rows_i] = pr, pd
            write_csv(out_root / model / f"oof_r{rep}.csv", ["relative_path"] + [f"{l} prob" for l in labels] + [f"{l} pred" for l in labels],
                      [[p] + [f"{v:.5f}" for v in prob[i]] + [int(v) for v in pred[i]] for i, p in enumerate(pages)])
            rows, levels = metrics(y, pred, labels)
            per_label += [{"model": model, "repeat": rep, **r} for r in rows]
            summary += [{"model": model, "repeat": rep, "level": name, **v} for name, v in levels.items()]
            binary.append({"model": model, "repeat": rep, **normal_vs_pjb(y, pred)})
    write_report(report, per_label, summary, binary, list(data))
    log.info("report written to %s", report)


def write_report(report: Path, per_label: list[dict], summary: list[dict], binary: list[dict], models: list[str]) -> None:
    """Per-label, per-level and NORMAL vs PJB tables (one row per model and repeat); prints the means and SDs over repeats."""
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else v
    for name, rows in (("per_label", per_label), ("summary", summary), ("normal_vs_pjb", binary)):
        write_csv(report / f"{name}.csv", list(rows[0]), [[fmt(v) for v in r.values()] for r in rows])
    ms = lambda rows, k: f"{np.mean([r[k] for r in rows]):.4f} (sd {np.std([r[k] for r in rows]):.4f})"
    print("| Model | Level | Micro F1 | Macro F1 | Exact-set accuracy |\n|---|---|---|---|---|")
    for model in models:
        for name in ("overall", "group", "diagnosis"):
            r = [s for s in summary if s["model"] == model and s["level"] == name]
            print(f"| {model} | {name} | {ms(r, 'micro_f1')} | {ms(r, 'macro_f1')} | {ms(r, 'exact_set_accuracy')} |")
    print("\n| Model | NORMAL F1 | PJB F1 | Macro F1 | NORMAL sensitivity | PJB sensitivity |\n|---|---|---|---|---|---|")
    for model in models:
        r = [b for b in binary if b["model"] == model]
        print(f"| {model} | {ms(r, 'normal_f1')} | {ms(r, 'pjb_f1')} | {ms(r, 'macro_f1')} | {ms(r, 'normal_sensitivity')} | {ms(r, 'pjb_sensitivity')} |")


if __name__ == "__main__":
    main()
