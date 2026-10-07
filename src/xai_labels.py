"""Per diagnosis: where the final model's attribution differs from NORMAL pages, tested on every labeled page.

For each diagnosis with at least `min_pages` pages, Integrated Gradients (blurred baseline, `src.xai`) explains that diagnosis'
output on its own pages and on every NORMAL page. Each explanation becomes 24 shares: the fraction of absolute attribution in each
lead's QRS complexes and in the rest of that lead (R peaks found per panel, since the leads of a panel are simultaneous). A
Mann-Whitney test compares the diagnosis' pages with the NORMAL pages per share, Benjamini-Hochberg over every diagnosis, lead and
segment. The diagnosis passes the expected-sign check when a significant increase falls in one of its textbook leads (`expected`).
This describes what the trained model uses, so it runs on all pages, including its training pages.

Writes to `out_dir/<baseline>/labels/<page>.npz` (resumable), `report_dir/<baseline>/label_stats.csv` (one row per diagnosis, lead and
segment) and `label_summary.csv` (one row per diagnosis: pages, QRS share, significant increases, expected leads, check).

Usage:
    uv run python -m src.xai_labels --config configs/xai.yml
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import false_discovery_control, mannwhitneyu

from src import paths
from src.linear_probe import label_matrix, write_csv
from src.models import LEADS
from src.progress import Progress
from src.xai import attribute, baseline, fold, load_model, page_folder, read_record, tile_index

SEGMENTS = ["QRS", "rest"]


def qrs_mask(rec: np.ndarray, panels: list[str | None], fs: int, ls: dict) -> np.ndarray:
    """True on record samples inside a QRS window; R peaks from the summed QRS-band energy of each panel's leads."""
    b, a = butter(2, ls["qrs_band_hz"], btype="band", fs=fs)
    mask = np.zeros(rec.shape, bool)
    lo, hi = (int(v * fs / 1000) for v in ls["qrs_ms"])
    for panel in {p for p in panels if p}:
        rows = [i for i, p in enumerate(panels) if p == panel and np.isfinite(rec[i]).sum() > fs]
        if not rows:
            continue
        ok = np.isfinite(rec[rows]).all(axis=0)
        if ok.sum() < fs:
            continue
        idx = np.flatnonzero(ok)
        energy = np.abs(np.diff(filtfilt(b, a, rec[rows][:, idx], axis=1), axis=1, prepend=0)).sum(axis=0)
        energy = np.convolve(energy, np.ones(int(0.05 * fs)) / int(0.05 * fs), mode="same")
        peaks, _ = find_peaks(energy, distance=int(ls["rr_min_s"] * fs), height=0.35 * np.percentile(energy, 99))
        for p in idx[peaks]:
            mask[rows, max(0, p + lo):p + hi] = True
    return mask


def shares(r: np.ndarray, qrs: np.ndarray) -> np.ndarray:
    """Fraction of absolute attribution per lead and segment (leads x 2), NaN for a lead with no data."""
    a = np.abs(r)
    total = np.nansum(a)
    out = np.full((r.shape[0], 2), np.nan)
    for l in range(r.shape[0]):
        if np.isfinite(r[l]).any():
            out[l] = [np.nansum(a[l][qrs[l]]) / total, np.nansum(a[l][~qrs[l]]) / total]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text(encoding="utf-8"))
    ls = cfg["label_stats"]
    kind = ls["baseline"]
    log_dir = paths.resolve(cfg["log_dir"])
    logging.basicConfig(filename=log_dir / f"xai_labels-{datetime.now():%Y%m%d}.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    log = logging.getLogger("xai_labels")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, labels, _, parent, tc = load_model(cfg, device)
    out, report = paths.resolve(cfg["out_dir"]) / kind / "labels", paths.resolve(cfg["report_dir"]) / kind
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    pages = list(np.load(paths.resolve(cfg["inputs"]), allow_pickle=False)["relative_path"])
    lab_cfg = {k: tc[k] for k in ("dataset", "groups_config", "label_column", "detailed_column", "merge", "unknown_label", "folds")}
    _, y = label_matrix(lab_cfg, pages)
    chosen = [j for j, l in enumerate(labels) if l.startswith("diagnosis: ") and l[11:] in ls["expected"] and y[:, j].sum() >= ls["min_pages"]]
    normal = ~y.any(axis=1)
    run_dir = paths.resolve(cfg["run"])
    todo = [i for i in range(len(pages)) if not (out / f"{page_folder(pages[i])}.npz").exists()]
    log.info("start: %d diagnoses %s, %d of %d pages to do", len(chosen), [labels[j] for j in chosen], len(todo), len(pages))
    bar = Progress(len(todo), f"label stats, {kind} baseline", "pages", log)
    for i in todo:
        d = run_dir / page_folder(pages[i])
        rec = read_record(d)
        targets = [j for j in chosen if normal[i] or y[i, j]]
        save = {"targets": np.array(targets, int)}
        if targets and not np.isnan(rec).all():
            panels = [l.get("panel") for l in json.loads((d / "record.json").read_text(encoding="utf-8"))["leads"]]
            qrs = qrs_mask(rec, panels, cfg["fs"], ls)
            x = torch.from_numpy(net.adapter.prep(rec))[None].to(device)
            base = baseline(x, kind, cfg["baselines"][kind], cfg["fs"])
            idx = tile_index(rec, x.shape[-1])
            save["qrs_fraction"] = qrs.sum() / max(np.isfinite(rec).sum(), 1)
            for j in targets:
                save[f"shares_{j}"] = shares(fold(attribute(net, x, base, j, ls["method"], cfg["methods"][ls["method"]], cfg["batch_size"]), idx, rec.shape[1]), qrs)
        np.savez_compressed(out / f"{page_folder(pages[i])}.npz", **save)
        bar.step(1, pages[i])
    bar.close()

    data = {j: {"pos": [], "normal": []} for j in chosen}
    for i, p in enumerate(pages):
        z = np.load(out / f"{page_folder(p)}.npz")
        for j in chosen:
            if f"shares_{j}" in z.files:
                data[j]["normal" if normal[i] else "pos"].append(z[f"shares_{j}"])
    rows = []
    for j in chosen:
        pos, nor = np.array(data[j]["pos"]), np.array(data[j]["normal"])
        for l, lead in enumerate(LEADS):
            for s, seg in enumerate(SEGMENTS):
                a, b = pos[:, l, s], nor[:, l, s]
                a, b = a[np.isfinite(a)], b[np.isfinite(b)]
                u, p = mannwhitneyu(a, b, alternative="two-sided")
                rows.append({"diagnosis": labels[j][11:], "lead": lead, "segment": seg, "pages": len(a), "normal_pages": len(b), "median": float(np.median(a)),
                             "median_normal": float(np.median(b)), "rank_biserial": float(2 * u / (len(a) * len(b)) - 1), "p": float(p)})
    q = false_discovery_control([r["p"] for r in rows])
    for r, qv in zip(rows, q):
        r["q"], r["significant"] = float(qv), int(qv < ls["fdr"])
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else v
    write_csv(report / "label_stats.csv", list(rows[0]), [[fmt(v) for v in r.values()] for r in rows])
    summary = []
    for j in chosen:
        name = labels[j][11:]
        mine = [r for r in rows if r["diagnosis"] == name]
        up = sorted([r for r in mine if r["significant"] and r["rank_biserial"] > 0], key=lambda r: -r["rank_biserial"])
        down = sorted([r for r in mine if r["significant"] and r["rank_biserial"] < 0], key=lambda r: r["rank_biserial"])
        pos = np.array(data[j]["pos"])
        summary.append({"diagnosis": name, "pages": len(pos), "normal_pages": len(data[j]["normal"]), "qrs_share": f"{np.nanmean(np.nansum(pos[:, :, 0], axis=1)):.3f}",
                        "higher_than_normal": "; ".join(f"{r['lead']} {r['segment']} ({r['rank_biserial']:+.2f})" for r in up) or "-",
                        "lower_than_normal": "; ".join(f"{r['lead']} {r['segment']} ({r['rank_biserial']:+.2f})" for r in down) or "-",
                        "expected_leads": ", ".join(ls["expected"][name]), "expected_found": "yes" if any(r["lead"] in ls["expected"][name] for r in up) else "no"})
    write_csv(report / "label_summary.csv", list(summary[0]), [list(s.values()) for s in summary])
    print("| Diagnosis | Pages | Higher than on NORMAL pages (rank-biserial) | Expected leads | Found |\n|---|---|---|---|---|")
    for s in summary:
        print(f"| {s['diagnosis']} | {s['pages']} | {s['higher_than_normal']} | {s['expected_leads']} | {s['expected_found']} |")


if __name__ == "__main__":
    main()
