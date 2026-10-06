"""Which defects and defect groups the pretrained models' outputs separate from NORMAL, to choose the label schema.

For every detailed defect and every ACC-CHD group with at least `min_pages` pages, takes the pages carrying it against the
NORMAL pages and finds, over all outputs of all listed model runs, the output with the highest |AUROC - 0.5|. The chance
level is the same maximum after shuffling which pages carry the defect (`shuffles` times, seed 42), since the best of
hundreds of outputs is above 0.5 even on noise. No model is trained. The table is printed and written to `out`.

Usage:
    uv run python -m src.label_schema --config configs/label_schema.yml
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from src import paths


def best_auroc(x: np.ndarray, pos: np.ndarray) -> tuple[float, int]:
    """Largest |AUROC - 0.5| over the columns of x (pages x outputs), by ranks; returns the AUROC and its column."""
    r = x.argsort(0).argsort(0) + 1.0  # ponytail: ties broken by order, outputs are 5-decimal floats
    n1, n0 = pos.sum(), (~pos).sum()
    a = (r[pos].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    i = int(np.abs(a - 0.5).argmax())
    return float(a[i]), i


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text(encoding="utf-8"))
    with (paths.dataset(cfg["dataset"]) / "_labels" / "manifest.csv").open(encoding="utf-8") as fh:
        detailed = {r["relative_path"]: set(json.loads(r[cfg["detailed_column"]] or "[]")) for r in csv.DictReader(fh)}
    names, columns, pages = [], [], None
    for run in cfg["runs"]:
        with (paths.resolve("@inferences") / run / cfg["dataset"] / "predictions.csv").open(encoding="utf-8") as fh:
            rows = sorted(csv.DictReader(fh), key=lambda r: r["relative_path"])
        if pages is None:
            pages = [r["relative_path"] for r in rows]
        assert pages == [r["relative_path"] for r in rows], f"{run} covers other pages"
        outputs = list(rows[0])[4:]
        names += [f"{run}: {o}" for o in outputs]
        columns.append(np.array([[float(r[o]) for o in outputs] for r in rows]))
    x = np.hstack(columns)
    labels = [detailed[p] for p in pages]
    normal = np.array([cfg["normal_label"] in s for s in labels])
    targets = {d: np.array([d in s for s in labels]) for d in sorted({d for s in labels for d in s} - {cfg["normal_label"]})}
    for g in sorted(set(cfg["groups"].values()) - {"normal"}):
        targets[f"group {g}"] = np.array([any(cfg["groups"][d] == g for d in s) for s in labels])
    rng = np.random.default_rng(42)
    print(f"{len(pages)} pages, {normal.sum()} NORMAL, {x.shape[1]} outputs from {len(cfg['runs'])} runs\n")
    print("| Target | Pages | Best AUROC | Chance level (mean, 95th percentile) | Best output |\n|---|---|---|---|---|")
    table = []
    for name, has in sorted(targets.items(), key=lambda t: -t[1].sum()):
        row = {"target": name, "pages": int(has.sum()), "best_auroc": "", "chance_mean": "", "chance_p95": "", "best_output": ""}
        if has.sum() >= cfg["min_pages"]:
            keep = has | normal
            pos = has[keep]
            a, i = best_auroc(x[keep], pos)
            null = np.array([abs(best_auroc(x[keep], rng.permutation(pos))[0] - 0.5) for _ in range(cfg["shuffles"])]) + 0.5
            row |= {"best_auroc": f"{a:.4f}", "chance_mean": f"{null.mean():.4f}", "chance_p95": f"{np.percentile(null, 95):.4f}", "best_output": names[i]}
        table.append(row)
        print(f"| {name} | {row['pages']} | {row['best_auroc'] or '-'} | {row['chance_mean'] or '-'}, {row['chance_p95'] or '-'} | {row['best_output'] or '-'} |")
    out = paths.resolve(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)


if __name__ == "__main__":
    main()
