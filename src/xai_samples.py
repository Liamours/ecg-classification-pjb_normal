"""A static, shareable page of example explanations: per page the true and predicted labels, the 12 digitized leads with shaded
blocks where replacing a stretch lowers the explained output (occlusion), and the leads whose removal lowers it most tinted
(`samples` in the config). It carries no photo, so no handwriting from the page reaches it.

Usage:
    uv run python -m src.xai_samples --config configs/xai.yml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src import paths
from src.explain import render, top_leads
from src.models import LEADS
from src.xai import page_folder
from src.xai_viewer import read_csv

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text(encoding="utf-8"))
    sm = cfg["samples"]
    from src.xai import load_model
    _, labels, thresholds, parent, _ = load_model(cfg, "cpu")
    manifest = {r["relative_path"]: r["diagnosis_labels_detailed"] for r in read_csv(paths.dataset(cfg["dataset"]) / "_labels" / "manifest.csv")}
    out = paths.resolve(cfg["out_dir"]) / "pages"
    cards = []
    for name in sm["pages"]:
        rel = cfg["pages"][name]
        z = np.load(out / page_folder(rel) / f"xai_{sm['baseline']}.npz")
        p, t = z["prob"], int(z["target"])
        drop = z["lead_drop"]
        order = top_leads(drop, sm["top_leads"])
        is_group = parent < 0
        pred = np.zeros(len(p), bool)
        pred[is_group] = p[is_group] >= thresholds["group"]
        pred[~is_group] = (p[~is_group] >= thresholds["diagnosis"]) & pred[parent[~is_group]]
        truth = json.loads(manifest[rel] or "[]")
        predicted = [labels[j][11:] for j in np.flatnonzero(pred) if labels[j].startswith("diagnosis")] or ["NORMAL"]
        cards.append(f'''<section class="card"><div class="head"><h2>{name}</h2><span class="file">{rel}</span></div>
<dl><div><dt>True labels</dt><dd>{", ".join(truth)}</dd></div><div><dt>Predicted</dt><dd>{", ".join(predicted)}</dd></div>
<div><dt>Explained output</dt><dd>{labels[t].replace("group: ", "group ")} ({p[t]:.4f})</dd></div>
<div><dt>Most influential leads (probability drop when replaced)</dt><dd>{", ".join(f"{LEADS[i]} {drop[i]:.3f}" for i in order)}</dd></div></dl>
<div class="svgwrap">{render(z["record"], z["attr_occlusion"], drop, order, sm["levels"], sm["percentile"])}</div></section>''')
    html = (REPO / "src" / "xai_samples.html").read_text(encoding="utf-8").replace("__TOP__", str(sm["top_leads"])).replace("__CARDS__", "\n".join(cards))
    dest = paths.resolve(cfg["report_dir"]) / "xai_samples.html"
    dest.write_text(html, encoding="utf-8")
    print(f"{dest}: {len(html) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
