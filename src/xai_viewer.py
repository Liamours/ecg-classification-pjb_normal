"""One HTML page for the XAI results of `src.xai`: the example pages (photo with the digitized trace coloured by attribution,
the 12 leads, the drop per zeroed lead) under every method, then the test-page results (deletion and insertion curves, their
areas, the label-by-lead map). The data is embedded in the page (`src/xai_viewer.html` is the template).

Usage:
    uv run python -m src.xai_viewer --config configs/xai.yml
"""
import argparse
import base64
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from src import paths
from src.models import LEADS
from src.xai import page_folder, upright

REPO = Path(__file__).resolve().parents[1]
NAMES = {"saliency": "Saliency", "input_x_gradient": "Gradient x input", "integrated_gradients": "Integrated Gradients", "smoothgrad": "SmoothGrad",
         "gradient_shap": "GradientSHAP", "deeplift": "DeepLIFT", "grad_cam": "Grad-CAM", "occlusion": "Occlusion", "random": "Random order"}
BIN = 5          # record samples per attribution bin (10 ms at 500 Hz)
STEP = 2         # record samples per drawn point


def scaled(r: np.ndarray) -> list[list[float | None]]:
    """Attribution in bins of BIN samples, divided by the 99.5th percentile of its absolute value and clipped to [-1, 1]."""
    n = r.shape[1] // BIN
    with np.errstate(all="ignore"):
        b = np.nanmean(r[:, :n * BIN].reshape(r.shape[0], n, BIN), axis=2)
    s = np.nanpercentile(np.abs(b), 99.5) if np.isfinite(b).any() else 1.0
    b = np.clip(b / (s or 1.0), -1, 1)
    return [[None if not np.isfinite(v) else round(float(v), 3) for v in lead] for lead in b]


def photo(img: np.ndarray, side: int) -> tuple[str, float]:
    s = min(1.0, side / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(), s


def example(name: str, rel: str, d: Path, labels: list[str], thresholds: dict, parent: np.ndarray, methods: list[str], kinds: list[str], manifest: dict, side: int) -> dict:
    zs = {k: np.load(d / f"xai_{k}.npz") for k in kinds}
    z = zs[kinds[0]]
    page = json.loads((d / "page.json").read_text(encoding="utf-8"))
    uri, s = photo(upright(Path(page["source"]), page), side)
    p = z["prob"]
    pred = np.zeros(len(p), bool)
    is_group = parent < 0
    pred[is_group] = p[is_group] >= thresholds["group"]
    pred[~is_group] = (p[~is_group] >= thresholds["diagnosis"]) & pred[parent[~is_group]]
    top = np.argsort(-p)[:6]
    rec = z["record"]
    traces = {}
    for i, lead in enumerate(LEADS):
        if f"px_{lead}" not in z.files:
            continue
        px, ri = z[f"px_{lead}"][::STEP] * s, z[f"ri_{lead}"][::STEP]
        traces[lead] = {"xy": [round(float(v), 1) for v in px.reshape(-1)], "bin": [int(v) // BIN for v in ri]}
    return {"name": name, "file": rel, "truth": json.loads(manifest[rel] or "[]"), "pred": [labels[j] for j in np.flatnonzero(pred)],
            "target": labels[int(z["target"])], "target_prob": round(float(p[int(z["target"])]), 4),
            "top": [[labels[j], round(float(p[j]), 4)] for j in top], "photo": uri, "traces": traces,
            "record": [[None if not np.isfinite(v) else round(float(v), 3) for v in lead[::STEP]] for lead in rec],
            "by_baseline": {k: {"attr": {m: scaled(zk[f"attr_{m}"]) for m in methods}, "lead_drop": [None if not np.isfinite(v) else round(float(v), 4) for v in zk["lead_drop"]],
                                "base_prob": round(float(zk["base_prob"]), 4)} for k, zk in zs.items()}}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text(encoding="utf-8"))
    from src.xai import load_model
    _, labels, thresholds, parent, _ = load_model(cfg, "cpu")
    report, out = paths.resolve(cfg["report_dir"]), paths.resolve(cfg["out_dir"])
    methods = list(cfg["methods"])
    manifest = {r["relative_path"]: r["diagnosis_labels_detailed"] for r in read_csv(paths.dataset(cfg["dataset"]) / "_labels" / "manifest.csv")}
    kinds = list(cfg["baselines"])
    pages = [example(n, rel, out / "pages" / page_folder(rel), labels, thresholds, parent, methods, kinds, manifest, cfg["viewer_side"]) for n, rel in cfg["pages"].items()]
    fr = cfg["faithfulness"]["fractions"]
    test = {}
    for k in kinds:
        rows = read_csv(report / k / "faithfulness.csv")
        curve = lambda m, c: [float(np.mean([float(r[c]) for r in rows if r["method"] == m and abs(float(r["fraction"]) - f) < 1e-6])) for f in fr]
        test[k] = {"fractions": fr, "pages": len({r["page"] for r in rows}),
                   "deletion": {m: curve(m, "deletion_prob") for m in methods + ["random"]}, "insertion": {m: curve(m, "insertion_prob") for m in methods + ["random"]},
                   "auc": read_csv(report / k / "faithfulness_summary.csv"), "lead_map": read_csv(report / k / "lead_map.csv")}
    ls = cfg["label_stats"]
    by_label = {"baseline": ls["baseline"], "method": NAMES[ls["method"]], "fdr": ls["fdr"], "expected": ls["expected"],
                "stats": read_csv(report / ls["baseline"] / "label_stats.csv"), "summary": read_csv(report / ls["baseline"] / "label_summary.csv")}
    data = {"methods": methods, "names": NAMES, "leads": LEADS, "baselines": kinds, "fs": cfg["fs"] / STEP, "bin_s": BIN / cfg["fs"], "thresholds": thresholds, "pages": pages,
            "test": test, "by_label": by_label}
    html = (REPO / "src" / "xai_viewer.html").read_text(encoding="utf-8").replace("__DATA__", json.dumps(data, separators=(",", ":")))
    (report / "xai_viewer.html").write_text(html, encoding="utf-8")
    print(f"{report / 'xai_viewer.html'}: {len(html) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
