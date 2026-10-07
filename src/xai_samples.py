"""A static, shareable page of example explanations: per page the true and predicted labels and the 12 digitized leads coloured
by attribution (`samples` in the config). It carries no photo, so no handwriting from the page reaches it.

Usage:
    uv run python -m src.xai_samples --config configs/xai.yml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src import paths
from src.models import LEADS
from src.xai import page_folder
from src.xai_viewer import NAMES, read_csv

REPO = Path(__file__).resolve().parents[1]
W, ROW, GAP = 330, 46, 24      # px: lead width, lead row height, gap between the two lead columns


def runs(level: np.ndarray) -> list[tuple[int, int, int]]:
    """Consecutive stretches of one colour level: (start, end, level)."""
    out, s = [], 0
    for i in range(1, len(level) + 1):
        if i == len(level) or level[i] != level[s]:
            out.append((s, i, int(level[s])))
            s = i
    return out


def leads_svg(rec: np.ndarray, attr: np.ndarray, sm: dict, fs: int, top: list[str]) -> str:
    """The 12 leads, each stretch coloured by its attribution: smoothed over `smooth_s` for display, divided by the given
    percentile of its absolute value, cut into `levels` steps per sign."""
    k = max(1, int(sm["smooth_s"] * fs))
    attr = np.array([np.convolve(np.nan_to_num(a), np.ones(k) / k, mode="same") for a in attr])
    finite = np.abs(attr[np.isfinite(rec)])
    scale = np.percentile(finite, sm["percentile"]) if finite.size else 1.0
    levels = sm["levels"]
    n = rec.shape[1]
    parts = []
    for li, lead in enumerate(LEADS):
        x0, y0 = (li // 6) * (W + GAP), (li % 6) * ROW
        parts.append(f'<text x="{x0}" y="{y0 + ROW / 2 + 4}"{" class=\"top\"" if lead in top else ""}>{lead}</text>')
        v = rec[li]
        ok = np.isfinite(v)
        if not ok.any():
            parts.append(f'<text x="{x0 + 34}" y="{y0 + ROW / 2 + 4}" class="miss">no data</text>')
            continue
        lo, hi = np.nanmin(v), np.nanmax(v)
        span, mid = max(hi - lo, 0.5), (hi + lo) / 2
        xs = x0 + 30 + np.arange(n) / n * (W - 30)
        ys = y0 + ROW / 2 - (v - mid) / span * (ROW - 8)
        a = np.nan_to_num(np.clip(attr[li] / (scale or 1), -1, 1))
        lvl = np.where(ok, np.round(a * levels), 99).astype(int)
        for s, e, k in runs(lvl):
            if k == 99:
                continue
            pts = " ".join(f"{xs[i]:.1f},{ys[i]:.1f}" for i in range(max(s - 1, 0), e) if ok[i])
            parts.append(f'<polyline class="l{k}" points="{pts}"/>')
    return f'<svg viewBox="-2 -2 {2 * W + GAP + 4} {6 * ROW + 4}" role="img" aria-label="12 digitized leads coloured by attribution">{"".join(parts)}</svg>'


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
        order = np.argsort(-np.nan_to_num(drop, nan=-np.inf))[:sm["top_leads"]]
        is_group = parent < 0
        pred = np.zeros(len(p), bool)
        pred[is_group] = p[is_group] >= thresholds["group"]
        pred[~is_group] = (p[~is_group] >= thresholds["diagnosis"]) & pred[parent[~is_group]]
        truth = [d for d in json.loads(manifest[rel] or "[]")]
        predicted = [labels[j][11:] for j in np.flatnonzero(pred) if labels[j].startswith("diagnosis")] or ["NORMAL"]
        cards.append(f'''<section class="card"><div class="head"><h2>{name}</h2><span class="file">{rel}</span></div>
<dl><div><dt>True labels</dt><dd>{", ".join(truth)}</dd></div><div><dt>Predicted</dt><dd>{", ".join(predicted)}</dd></div>
<div><dt>Explained output</dt><dd>{labels[t].replace("group: ", "group ")} ({p[t]:.4f})</dd></div>
<div><dt>Most influential leads (probability drop when replaced)</dt><dd>{", ".join(f"{LEADS[i]} {drop[i]:.3f}" for i in order)}</dd></div></dl>
<div class="svgwrap">{leads_svg(z["record"], z[f"attr_{sm['method']}"], sm, cfg["fs"], [LEADS[i] for i in order])}</div></section>''')
    lv = sm["levels"]
    classes = "\n".join(f".l{k} {{ stroke: color-mix(in srgb, var({'--pos' if k > 0 else '--neg'}) {abs(k) / lv * 100:.0f}%, var(--neutral)); }}" for k in range(-lv, lv + 1) if k)
    html = (REPO / "src" / "xai_samples.html").read_text(encoding="utf-8").replace("__CLASSES__", classes).replace("__METHOD__", NAMES[sm["method"]]).replace("__CARDS__", "\n".join(cards))
    dest = paths.resolve(cfg["report_dir"]) / "xai_samples.html"
    dest.write_text(html, encoding="utf-8")
    print(f"{dest}: {len(html) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
