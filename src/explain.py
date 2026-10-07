"""Explanation step of the classification pipeline, for one digitized record: which leads and which stretches drive the output.

Occlusion against a blurred baseline, forward passes only, so it runs on the PyTorch checkpoint and on the ONNX export alike
(chosen 2026-10-07: occlusion had the best deletion area of eight methods on the test pages, wiki/TODO-classification.md D5).
- Lead significance: the drop of the explained output's probability when a whole lead is replaced by its baseline.
- Blocks: the drop when a `window_s` stretch of one lead is replaced, spread over that stretch (record samples).
The explained output is the group label with the highest probability, the one that decides NORMAL (no group above its threshold)
against PJB. The model reads each lead's digitized stretch repeated to 10 s (`tile`), so a record stretch is replaced in every
repeat. `render` draws the 12 leads in boxes, blocks shaded by drop, the top leads tinted.

Usage (writes <out_dir>/<page>/explanation.json and explanation.svg):
    uv run python -m src.explain --config configs/explain.yml <page folder with record.csv and record.json> [...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.ndimage import gaussian_filter1d

from src import paths
from src.finetune import build
from src.models import LEADS, ECGFounderPrep
from src.thresholds import parents, predict

REPO = Path(__file__).resolve().parents[1]
W, ROW, GAP = 330, 46, 24      # px: lead box width, lead row height, gap between the two lead columns


def load_model(path, device: str):
    """The fine-tuned checkpoint (`src.finetune_single`): network in eval mode, labels, thresholds, group index per label, config."""
    m = torch.load(paths.resolve(path), map_location="cpu", weights_only=True)
    tc = m["config"]
    net = build(tc["spec"], len(m["labels"]), device)
    net.load_state_dict(m["model"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, m["labels"], m["thresholds"], parents(m["labels"], REPO / tc["groups_config"], tc["merge"], tc["unknown_label"]), tc


class TorchScorer:
    """Probabilities (N x labels) of a batch of model inputs (N x 12 x 5000) from the PyTorch checkpoint."""
    def __init__(self, net, device: str, batch_size: int):
        self.net, self.device, self.bs = net, device, batch_size

    def __call__(self, xs: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(xs), self.bs):
                out.append(torch.sigmoid(self.net(torch.from_numpy(xs[i:i + self.bs]).to(self.device))).cpu().numpy())
        return np.concatenate(out)


class OnnxScorer:
    """The same from the ONNX export (input `ecg` 1 x 12 x 5000, output `scores`, raw), one input per run."""
    def __init__(self, path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(paths.resolve(path)), providers=["CPUExecutionProvider"])

    def __call__(self, xs: np.ndarray) -> np.ndarray:
        return np.concatenate([1 / (1 + np.exp(-self.session.run(["scores"], {"ecg": x[None]})[0])) for x in xs.astype(np.float32)])


def tile_index(rec: np.ndarray, n_in: int) -> list[np.ndarray]:
    """Per lead, the record sample each input sample copies (`fit_length` with `tile`); empty for a lead with no data."""
    out = []
    for lead in rec:
        v = np.flatnonzero(~np.isnan(lead))
        out.append(np.resize(v, n_in) if len(v) else np.array([], int))
    return out


def blur(x: np.ndarray, sigma_s: float, fs: int) -> np.ndarray:
    """The blurred baseline (Sturmfels et al. 2020): the input smoothed along time by a Gaussian of `sigma_s`."""
    return gaussian_filter1d(x, sigma_s * fs, axis=-1, mode="nearest")


def occlude(score, x: np.ndarray, base: np.ndarray, idx: list[np.ndarray], target: int, window: int) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """Record-space block attribution (leads x record samples, drop per sample, NaN where no data), the whole-lead drops and the
    windows (lead, first record sample, end, drop), for one input `x` (12 x n) against `base`."""
    n_rec = max((ix.max() + 1 for ix in idx if len(ix)), default=0)
    cuts, xs = [], [x]
    for l, ix in enumerate(idx):
        if not len(ix):
            continue
        v = np.unique(ix)
        for s in range(v.min(), v.max() + 1, window):
            sel = v[(v >= s) & (v < s + window)]
            if len(sel):
                xo, m = x.copy(), np.isin(ix, sel)
                xo[l, m] = base[l, m]
                cuts.append((l, sel))
                xs.append(xo)
        xo = x.copy()
        xo[l] = base[l]
        cuts.append((l, None))
        xs.append(xo)
    p = score(np.stack(xs).astype(np.float32))[:, target]
    drop = p[0] - p[1:]
    blocks, lead_drop, windows = np.full((len(idx), n_rec), np.nan), np.full(len(idx), np.nan), []
    for (l, sel), d in zip(cuts, drop):
        if sel is None:
            lead_drop[l] = d
        else:
            blocks[l, sel] = d / len(sel)
            windows.append((l, int(sel[0]), int(sel[-1]) + 1, float(d)))
    return blocks, lead_drop, windows


def lead_drops(score, rec: np.ndarray, cfg: dict) -> np.ndarray:
    """Whole-lead drop of every output (leads x outputs, NaN for a lead with no data): one run with each lead replaced by its
    blurred baseline, against the page's own probabilities."""
    x = ECGFounderPrep("tile")(rec)
    base = blur(x, cfg["sigma_s"], cfg["fs"])
    has = np.isfinite(rec).any(axis=1)
    xs = [x] + [np.where(np.arange(len(x))[:, None] == l, base, x) for l in np.flatnonzero(has)]
    p = score(np.stack(xs).astype(np.float32))
    out = np.full((len(x), p.shape[1]), np.nan)
    out[has] = p[0] - p[1:]
    return out


def explain(score, rec: np.ndarray, groups: np.ndarray, cfg: dict) -> dict:
    """Probabilities, the explained output, block attribution and whole-lead drops for one record (12 x record samples, mV)."""
    x = ECGFounderPrep("tile")(rec)
    base = blur(x, cfg["sigma_s"], cfg["fs"])
    prob = score(x[None])[0]
    target = int(groups[np.argmax(prob[groups])])
    blocks, lead_drop, windows = occlude(score, x, base, tile_index(rec, x.shape[-1]), target, int(cfg["window_s"] * cfg["fs"]))
    full = np.full(rec.shape, np.nan)
    full[:, :blocks.shape[1]] = blocks
    return {"prob": prob, "target": target, "blocks": full, "lead_drop": lead_drop, "windows": windows}


def top_leads(lead_drop: np.ndarray, k: int) -> list[int]:
    return [int(i) for i in np.argsort(-np.nan_to_num(lead_drop, nan=-np.inf))[:k] if lead_drop[i] > 0]


def runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    """Consecutive stretches of one value: (start, end, value)."""
    out, s = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[s]:
            out.append((s, i, int(values[s])))
            s = i
    return out


def render(rec: np.ndarray, blocks: np.ndarray, lead_drop: np.ndarray, top: list[int], levels: int, percentile: float) -> str:
    """SVG of the 12 leads in boxes: blocks shaded where replacing a stretch lowers the output (opacity by drop, divided by the
    given percentile over the page), the `top` leads' boxes tinted by their whole-lead drop. Classes: box, lead, block."""
    pos = np.where(np.isfinite(blocks), np.clip(blocks, 0, None), np.nan)
    finite = pos[np.isfinite(pos) & (pos > 0)]
    scale = np.percentile(finite, percentile) if finite.size else 1.0
    big = max(float(np.nanmax(lead_drop[top])), 1e-9) if top else 1.0
    n, parts = rec.shape[1], []
    for li, lead in enumerate(LEADS):
        x0, y0 = (li // 6) * (W + GAP), (li % 6) * ROW
        box = f'x="{x0}" y="{y0 + 2}" width="{W}" height="{ROW - 4}"'
        parts.append(f'<rect class="box" {box}/>')
        if li in top:
            parts.append(f'<rect class="lead" {box} fill-opacity="{0.25 + 0.5 * lead_drop[li] / big:.2f}"/>')
        parts.append(f'<text x="{x0 + 6}" y="{y0 + 15}"' + (' class="top"' if li in top else "") + f">{lead}</text>")
        v = rec[li]
        ok = np.isfinite(v)
        if not ok.any():
            parts.append(f'<text x="{x0 + 40}" y="{y0 + ROW / 2 + 4}" class="miss">no data</text>')
            continue
        x = lambda i: x0 + 4 + i / n * (W - 8)
        level = np.where(ok, np.round(np.nan_to_num(np.clip(pos[li] / (scale or 1), 0, 1)) * levels), 0).astype(int)
        for a, b, k in runs(level):
            if k:
                parts.append(f'<rect class="block" x="{x(a):.1f}" y="{y0 + 2}" width="{x(b) - x(a):.1f}" height="{ROW - 4}" fill-opacity="{0.85 * k / levels:.2f}"/>')
        lo, hi = np.nanmin(v), np.nanmax(v)
        span, mid = max(hi - lo, 0.5), (hi + lo) / 2
        for a, b, k in runs(ok.astype(int)):
            if k:
                pts = " ".join(f"{x(i):.1f},{y0 + ROW / 2 - (v[i] - mid) / span * (ROW - 12):.1f}" for i in range(a, b, 2))
                parts.append(f'<polyline points="{pts}"/>')
    return f'<svg viewBox="-2 -2 {2 * W + GAP + 4} {6 * ROW + 4}" role="img" aria-label="12 digitized leads with occlusion blocks">{"".join(parts)}</svg>'


STYLE = ("<style>rect.box{fill:#fff;stroke:#dcdfe3}rect.lead{fill:#e8c84a}rect.block{fill:#7b5aa6}polyline{fill:none;stroke:#2b2f34;stroke-width:1.2}"
         "text{font-family:monospace;font-size:11px;fill:#6a7078}text.top{fill:#1f2226;font-weight:600}</style>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--backend", choices=["torch", "onnx"], help="overrides `backend` of the config")
    ap.add_argument("pages", type=Path, nargs="+", help="digitized page folders holding record.csv and record.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg["backend"] = args.backend or cfg["backend"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, labels, thresholds, parent, _ = load_model(cfg["model"], device)
    score = OnnxScorer(cfg["onnx"]) if cfg["backend"] == "onnx" else TorchScorer(net, device, cfg["batch_size"])
    groups = np.flatnonzero(parent < 0)
    out_root = paths.resolve(cfg["out_dir"]) / cfg["backend"]
    for page in args.pages:
        rec = np.genfromtxt(page / "record.csv", delimiter=",", skip_header=1)[:, 1:].T
        e = explain(score, rec, groups, cfg)
        top = top_leads(e["lead_drop"], cfg["top_leads"])
        pred = predict(e["prob"][None], thresholds, parent)[0]
        out = out_root / page.name
        out.mkdir(parents=True, exist_ok=True)
        blocks = [{"lead": LEADS[l], "start_s": round(a / cfg["fs"], 3), "end_s": round(b / cfg["fs"], 3), "drop": round(d, 4)} for l, a, b, d in e["windows"] if d > 0]
        result = {"predicted": [labels[j] for j in np.flatnonzero(pred)] or ["NORMAL"], "explained_output": labels[e["target"]],
                  "probability": round(float(e["prob"][e["target"]]), 4), "lead_drop": {LEADS[i]: None if not np.isfinite(d) else round(float(d), 4) for i, d in enumerate(e["lead_drop"])},
                  "top_leads": [LEADS[i] for i in top], "blocks": blocks}
        (out / "explanation.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        svg = render(rec, e["blocks"], e["lead_drop"], top, cfg["levels"], cfg["percentile"])
        (out / "explanation.svg").write_text(svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1).replace(">", ">" + STYLE, 1), encoding="utf-8")
        print(f"{page.name}: {result['explained_output']} {result['probability']}, top leads {', '.join(result['top_leads']) or '-'} -> {out}")


if __name__ == "__main__":
    main()
