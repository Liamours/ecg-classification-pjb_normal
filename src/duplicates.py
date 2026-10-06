"""Which phone photos show the same recording: the evidence a study-level split needs.

Three signals per page pair:
- text: the recording date and time printed in the panel header, read from the page's docling JSON (`<dataset>/_text/`);
- signal: correlation of the digitized leads (`record.csv`), downsampled to `signal_fs`, over the leads both pages have;
- image: Hamming distance of a difference hash of the page photo, smallest over the four rotations.
Pairs with the same printed date and time are taken as known same-recording pairs, so the signal and image scores can be read
against them. Writes to `out_dir`: `pages.csv` (per page: date, time, md5, image hash) and `pairs.csv` (every pair with a shared
date and time, plus every pair whose signal correlation or image hash is among the closest), then prints the comparison.

Usage:
    uv run python -m src.duplicates --config configs/duplicates.yml
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.signal import decimate
from tqdm import tqdm

from src import paths
from src.records import source_key

DATE = re.compile(r"(\d{1,2})\s*[.\-/ ]\s*([A-Za-z]{3})\s*[.\-/ ]\s*(\d{2})\b")
TIME = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")  # colon only: "0.08-150Hz", the filter band printed on every page, is not a time


def header_key(path: Path) -> tuple[str, str]:
    """Most frequent date and time in the page's docling text ('' when not read)."""
    if not path.exists():
        return "", ""
    text = " | ".join(t.get("text", "") for t in json.loads(path.read_text(encoding="utf-8")).get("texts", []))
    dates = Counter(f"{a.zfill(2)}.{b.title()}.{c}" for a, b, c in DATE.findall(text))
    times = Counter(f"{int(a):02d}:{b}" for a, b in TIME.findall(text))
    return (dates.most_common(1)[0][0] if dates else ""), (times.most_common(1)[0][0] if times else "")


def dhash(path: Path, size: int) -> list[int]:
    """Difference hash of the photo in each of the four rotations, as integers."""
    img = Image.open(path).convert("L")
    out = []
    for k in range(4):
        a = np.asarray(img.rotate(90 * k, expand=True).resize((size + 1, size)), dtype=np.int16)
        out.append(int("".join("1" if v else "0" for v in (a[:, 1:] > a[:, :-1]).ravel()), 2))
    return out


def signal(page: Path, fs: int) -> np.ndarray:
    x = np.genfromtxt(page / "record.csv", delimiter=",", skip_header=1)[:, 1:].T
    out = np.full((12, x.shape[1] * fs // 500), np.nan)
    for i, lead in enumerate(x):
        ok = ~np.isnan(lead)
        if ok.sum() > 500:  # a lead under 1 s is too short to compare
            seg = decimate(np.where(ok, lead, np.nanmean(lead)), 500 // fs)
            seg[~ok[:: 500 // fs][: len(seg)]] = np.nan
            out[i, : len(seg)] = (seg - np.nanmean(seg)) / (np.nanstd(seg) + 1e-8)
    return out


def correlation(a: np.ndarray, b: np.ndarray, min_leads: int) -> float:
    both = ~np.isnan(a) & ~np.isnan(b)
    leads = [i for i in range(12) if both[i].sum() > 25]
    if len(leads) < min_leads:
        return np.nan
    return float(np.mean([np.corrcoef(a[i, both[i]], b[i, both[i]])[0, 1] for i in leads]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text(encoding="utf-8"))
    out = paths.resolve(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    root = paths.dataset(cfg["dataset"])
    pages = sorted(p.parent for p in paths.resolve(cfg["run"]).glob("*/record.json"))
    rows, sigs, hashes = [], [], []
    for page in tqdm(pages, desc="pages"):
        rel = source_key(json.loads((page / "record.json").read_text(encoding="utf-8"))["image"]["source"])[1]
        date, time = header_key((root / cfg["text_dir"] / rel).with_suffix(".json"))
        h = dhash(root / rel, cfg["hash_size"])
        rows.append({"page": page.name, "relative_path": rel, "date": date, "time": time, "md5": hashlib.md5((root / rel).read_bytes()).hexdigest(), "dhash": h[0]})
        sigs.append(signal(page, cfg["signal_fs"]))
        hashes.append(h)
    with (out / "pages.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    ham = np.array([[min(bin(a ^ b).count("1") for b in hashes[j]) for j in range(n)] for a in (h[0] for h in hashes)])
    corr = np.full((n, n), np.nan)
    for i in tqdm(range(n), desc="signal pairs"):
        for j in range(i + 1, n):
            corr[i, j] = corr[j, i] = correlation(sigs[i], sigs[j], cfg["min_leads"])
    iu = np.triu_indices(n, 1)
    same = np.array([rows[i]["date"] and rows[i]["time"] and (rows[i]["date"], rows[i]["time"]) == (rows[j]["date"], rows[j]["time"]) for i, j in zip(*iu)], bool)
    c, h = corr[iu], ham[iu]
    keep = same | (np.nan_to_num(c, nan=-1) >= np.nanpercentile(c, 99.9)) | (h <= np.percentile(h, 0.1))
    with (out / "pairs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["page_a", "page_b", "same_printed_date_time", "signal_correlation", "image_hash_distance", "same_md5"])
        for k in np.flatnonzero(keep):
            i, j = iu[0][k], iu[1][k]
            w.writerow([rows[i]["page"], rows[j]["page"], int(same[k]), "" if np.isnan(c[k]) else f"{c[k]:.4f}", int(h[k]), int(rows[i]["md5"] == rows[j]["md5"])])
    q = lambda v: ", ".join(f"{np.nanpercentile(v, p):.3f}" for p in (5, 50, 95)) if np.isfinite(v).any() else "-"
    print(f"{n} pages, {len(c)} pairs; pages with a printed date and time {sum(bool(r['date'] and r['time']) for r in rows)}; identical files (md5) pairs {sum(rows[i]['md5'] == rows[j]['md5'] for i, j in zip(*iu))}")
    print("| Pairs | Count | Signal correlation 5th, 50th, 95th percentile | Image hash distance 5th, 50th, 95th percentile |\n|---|---|---|---|")
    print(f"| same printed date and time | {same.sum()} | {q(c[same])} | {q(h[same].astype(float))} |")
    print(f"| all other pairs | {(~same).sum()} | {q(c[~same])} | {q(h[~same].astype(float))} |")


if __name__ == "__main__":
    main()
