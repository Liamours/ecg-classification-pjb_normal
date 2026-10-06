"""2-D maps of a model's frozen page embeddings, drawn as focus panels (after ECG-JEPA, arXiv 2410.08559, Figure 10).

For each run: the standardized `embeddings.csv` of the labeled dataset is reduced to 2-D by UMAP and by t-SNE (fixed seed). One
shared map per method, drawn as side-by-side panels that differ only in which pages are opaque: in `<method>-classes.png` one
panel per class in `focus` (NORMAL = pages with no label; a diagnosis = pages carrying it, variant names merged), in
`<method>-batches.png` one panel per collection batch. Writes to `<out_dir>/<run>/`: the PNGs and `coordinates.csv`.

Usage:
    uv run python -m src.embedding_map --config configs/embedding_map.yml
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import umap  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src import paths  # noqa: E402


def panels(xy: np.ndarray, masks: dict[str, np.ndarray], cfg: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(masks), figsize=(3.2 * len(masks), 3.5))
    for ax, (name, m) in zip(np.atleast_1d(axes), masks.items()):
        ax.scatter(*xy[~m].T, s=6, c=cfg["colors"]["rest"], alpha=cfg["alpha"]["rest"], linewidths=0, label="other pages")
        ax.scatter(*xy[m].T, s=8, c=cfg["colors"]["focus"], alpha=cfg["alpha"]["focus"], linewidths=0, label=f"{name} ({m.sum()})")
        ax.set_xticks([]), ax.set_yticks([])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=7, frameon=False, markerscale=1.5)  # below the panel, off the points
        for side in ax.spines.values():
            side.set_linewidth(0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text(encoding="utf-8"))
    with (paths.dataset(cfg["dataset"]) / "_labels" / "manifest.csv").open(encoding="utf-8") as fh:
        manifest = {r["relative_path"]: r for r in csv.DictReader(fh)}
    for run_name, run in cfg["runs"].items():
        with (paths.resolve("@inferences") / run / cfg["dataset"] / "embeddings.csv").open(encoding="utf-8") as fh:
            rows = sorted(csv.DictReader(fh), key=lambda r: r["relative_path"])
        cols = [c for c in rows[0] if c.startswith("emb_")]
        x = StandardScaler().fit_transform(np.array([[float(r[c]) for c in cols] for r in rows]))
        rel = [r["relative_path"] for r in rows]
        dx = [{cfg["merge"].get(d, d) for d in json.loads(manifest[p][cfg["detailed_column"]] or "[]")} - {"NORMAL"} for p in rel]
        normal = np.array([manifest[p][cfg["label_column"]] == "NORMAL" for p in rel])
        classes = {f: normal if f == "NORMAL" else np.array([f in s for s in dx]) for f in cfg["focus"]}
        batch = np.array([cfg["batches"][p.split("/")[0]] for p in rel])
        batches = {f"batch {b}": batch == b for b in sorted(set(batch))}
        maps = {"umap": umap.UMAP(random_state=cfg["seed"], **cfg["umap"]).fit_transform(x),
                "tsne": TSNE(random_state=cfg["seed"], init="pca", **cfg["tsne"]).fit_transform(x)}
        out = paths.resolve(cfg["out_dir"]) / run_name
        out.mkdir(parents=True, exist_ok=True)
        for method, xy in maps.items():
            panels(xy, classes, cfg, out / f"{method}-classes.png")
            panels(xy, batches, cfg, out / f"{method}-batches.png")
        with (out / "coordinates.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["relative_path", "batch", "normal", "umap_x", "umap_y", "tsne_x", "tsne_y"])
            for i, p in enumerate(rel):
                w.writerow([p, batch[i], int(normal[i]), *(f"{v:.4f}" for v in maps["umap"][i]), *(f"{v:.4f}" for v in maps["tsne"][i])])
        print(f"{run_name}: {len(rows)} pages, {len(cols)}-dimension embedding, maps in {out}")


if __name__ == "__main__":
    main()
