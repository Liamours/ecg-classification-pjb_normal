"""Reference outputs of the phone classifier and explanation for the pages of a digitization reference run
(ecg-digitization-synthetic_realistic, `src.digitize.reference`). Written into each page folder, next to its record.csv:
  classifier_input.csv   the 12 x 5000 array the model reads: each lead's digitized stretch repeated to 10 s, NaN to 0, one z-score over the array
  classifier.json        labels, raw scores and probabilities of classifier.ort in that order, the predicted labels (NORMAL when none)
  explanation.json       the same file under src.explain: explained output, whole-lead drops, top leads and 0.2 s blocks
Resumable: a page with explanation.json is skipped.

Usage (digitization repo's GPU environment, from this repo's root):
    PYTHONPATH=. ../../.venvs/digitize_gpu/Scripts/python.exe -m src.mobile_reference --config configs/mobile_reference.yml
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import psutil
import yaml
from tqdm import tqdm

from src import paths
from src.explain import OnnxScorer, describe, explain
from src.models import ECGFounderPrep
from src.thresholds import parents

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--reference", help="run folder; overrides the config")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg["reference"] = args.reference or cfg["reference"]
    free = psutil.virtual_memory().available / 2**30
    if free < cfg["min_free_gb"]:
        sys.exit(f"{free:.1f} GB of RAM free, {cfg['min_free_gb']} needed: close other programs and run again")
    ex = yaml.safe_load((REPO / cfg["explain"]).read_text(encoding="utf-8"))
    train = yaml.safe_load((REPO / cfg["train_config"]).read_text(encoding="utf-8"))
    meta = json.loads(paths.resolve(cfg["labels"]).read_text(encoding="utf-8"))
    labels, thresholds = meta["labels"], meta["thresholds"]
    parent = parents(labels, REPO / train["groups_config"], train["merge"], train["unknown_label"])
    score = OnnxScorer(ex["onnx"])
    prep = ECGFounderPrep("tile")
    for record in tqdm(sorted(paths.resolve(cfg["reference"]).glob("*/record.csv")), desc="pages"):
        page = record.parent
        if (page / "explanation.json").exists():
            continue
        rec = np.genfromtxt(record, delimiter=",", skip_header=1)[:, 1:].T
        x = prep(rec)
        np.savetxt(page / "classifier_input.csv", x, delimiter=",", fmt="%.6g")
        raw = score.session.run(["scores"], {"ecg": x[None]})[0][0]
        prob = 1 / (1 + np.exp(-raw))
        result = describe(explain(score, rec, np.flatnonzero(parent < 0), ex), labels, thresholds, parent, ex)
        (page / "classifier.json").write_text(json.dumps({"labels": labels, "scores": [round(float(v), 5) for v in raw], "probabilities": [round(float(v), 5) for v in prob],
                                                          "predicted": result["predicted"]}, indent=1), encoding="utf-8")
        (page / "explanation.json").write_text(json.dumps(result, indent=1), encoding="utf-8")   # last: the done marker


if __name__ == "__main__":
    main()
