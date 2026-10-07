"""Per diagnosis: which leads the final model needs for that diagnosis, with the pipeline's own lead significance (`src.explain`).

For every labeled page, each lead is replaced by its blurred baseline once and the drop of every output is kept (13 model runs
per page). For each diagnosis with at least `min_pages` pages, the drop of that diagnosis' output on its own pages is compared
with the drop of the same output on the NORMAL pages, per lead (Mann-Whitney, Benjamini-Hochberg over every diagnosis and lead).
A lead is significant for a diagnosis when removing it lowers that output more on the diagnosis' pages than on NORMAL pages. The
diagnosis passes the expected-sign check when a significant lead is one of its textbook leads (`expected`). This describes what
the trained model uses, so it runs on all pages, including its training pages.

Writes `out_dir/<baseline>/lead_drops/<page>.npy` (resumable), `report_dir/<baseline>/lead_stats.csv` (one row per diagnosis and
lead) and `lead_summary.csv` (one row per diagnosis).

Usage:
    uv run python -m src.xai_leads --config configs/xai.yml
"""
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import false_discovery_control, mannwhitneyu

from src import paths
from src.explain import TorchScorer, lead_drops, load_model
from src.linear_probe import label_matrix, write_csv
from src.models import LEADS
from src.progress import Progress
from src.xai import page_folder, read_record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text(encoding="utf-8"))
    ls, kind = cfg["label_stats"], cfg["label_stats"]["baseline"]
    log_dir = paths.resolve(cfg["log_dir"])
    logging.basicConfig(filename=log_dir / f"xai_leads-{datetime.now():%Y%m%d}.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    log = logging.getLogger("xai_leads")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, labels, _, _, tc = load_model(cfg["model"], device)
    score = TorchScorer(net, device, cfg["batch_size"])
    occ = {"sigma_s": cfg["baselines"][kind]["sigma_s"], "fs": cfg["fs"]}
    out, report = paths.resolve(cfg["out_dir"]) / kind / "lead_drops", paths.resolve(cfg["report_dir"]) / kind
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    pages = list(np.load(paths.resolve(cfg["inputs"]), allow_pickle=False)["relative_path"])
    lab_cfg = {k: tc[k] for k in ("dataset", "groups_config", "label_column", "detailed_column", "merge", "unknown_label", "folds")}
    _, y = label_matrix(lab_cfg, pages)
    chosen = [j for j, l in enumerate(labels) if l.startswith("diagnosis: ") and l[11:] in ls["expected"] and y[:, j].sum() >= ls["min_pages"]]
    normal = ~y.any(axis=1)
    run_dir = paths.resolve(cfg["run"])
    todo = [i for i in range(len(pages)) if not (out / f"{page_folder(pages[i])}.npy").exists()]
    log.info("start: %d diagnoses, %d of %d pages to do", len(chosen), len(todo), len(pages))
    bar = Progress(len(todo), f"lead drops, {kind} baseline", "pages", log)
    for i in todo:
        rec = read_record(run_dir / page_folder(pages[i]))
        d = np.full((len(LEADS), len(labels)), np.nan) if np.isnan(rec).all() else lead_drops(score, rec, occ)
        np.save(out / f"{page_folder(pages[i])}.npy", d)
        bar.step(1, pages[i])
    bar.close()

    drops = np.stack([np.load(out / f"{page_folder(p)}.npy") for p in pages])   # pages x leads x outputs
    rows = []
    for j in chosen:
        for l, lead in enumerate(LEADS):
            a, b = drops[y[:, j], l, j], drops[normal, l, j]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            rows.append({"diagnosis": labels[j][11:], "lead": lead, "pages": len(a), "normal_pages": len(b), "median_drop": float(np.median(a)),
                         "median_drop_normal": float(np.median(b)), "rank_biserial": float(2 * u / (len(a) * len(b)) - 1), "p": float(p)})
    for r, qv in zip(rows, false_discovery_control([r["p"] for r in rows])):
        r["q"], r["significant"] = float(qv), int(qv < ls["fdr"] and r["rank_biserial"] > 0)
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else v
    write_csv(report / "lead_stats.csv", list(rows[0]), [[fmt(v) for v in r.values()] for r in rows])
    summary = []
    for j in chosen:
        name = labels[j][11:]
        up = sorted([r for r in rows if r["diagnosis"] == name and r["significant"]], key=lambda r: -r["rank_biserial"])
        summary.append({"diagnosis": name, "pages": int(y[:, j].sum()), "significant_leads": "; ".join(f"{r['lead']} ({r['rank_biserial']:+.2f})" for r in up) or "-",
                        "expected_leads": ", ".join(ls["expected"][name]), "expected_found": "yes" if any(r["lead"] in ls["expected"][name] for r in up) else "no"})
    write_csv(report / "lead_summary.csv", list(summary[0]), [list(s.values()) for s in summary])
    print("| Diagnosis | Pages | Leads it needs more than NORMAL pages (rank-biserial) | Textbook leads | Found |\n|---|---|---|---|---|")
    for s in summary:
        print(f"| {s['diagnosis']} | {s['pages']} | {s['significant_leads']} | {s['expected_leads']} | {s['expected_found']} |")


if __name__ == "__main__":
    main()
