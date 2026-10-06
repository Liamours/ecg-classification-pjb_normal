"""Signal inference with pretrained open-weight ECG models on digitized 12-lead records, no fine-tuning.

Input per page: <run>/<page>/record.csv (12 leads in mV, 500 Hz, 4 s, empty where a lead is missing) and record.json.
Each model's preprocessing repeats its upstream code:
- `ecgfounder` (ECGFounder ptbxl_eval.py): leads I to V6 at 500 Hz, missing values 0, one z-score over the whole array, 10 s input.
- `hubert_ecg` (HuBERT-ECG utils.ecg_preprocessing and dataset.py): FIR band-pass 0.05 to 47 Hz, each lead min-max scaled to
  [-1, 1], first 5 s at 500 Hz, missing values set to the mean, the 12 leads flattened and decimated by 5 to 100 Hz.
- `ecg_fm` (ECG-FM infer_quickstart.ipynb with ecg-transform 0.1.3): leads I to V6 at 500 Hz, each lead z-scored, a constant
  or missing lead set to 0, one 5 s window.
`fill` sets how a lead shorter than the input reaches its length: `zero` leaves it missing (filled as above), `tile` repeats
the lead's digitized stretch. Writes <out_dir>/<dataset>/predictions.csv: page, relative_path, label, leads_ok, then one
sigmoid output per model label. Resumable per page. For labeled pages the summary ranks outputs by AUROC between PJB and NORMAL.

Usage:
    uv run python -m src.classify_signal --config configs/classify_signal.yml
"""
import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import signal as ss
from tqdm import tqdm

from src import paths
from src.records import load_labels, source_key

REPO = Path(__file__).resolve().parents[1]
FS = 500


def read_record(page_dir: Path) -> np.ndarray:
    """12 x samples in mV, NaN where a lead has no data."""
    return np.genfromtxt(page_dir / "record.csv", delimiter=",", skip_header=1)[:, 1:].T


def fit_length(x: np.ndarray, n: int, fill: str) -> np.ndarray:
    """Each lead to n samples: `tile` repeats its digitized stretch, `zero` pads with NaN."""
    if fill == "tile":
        return np.stack([np.resize(lead[~np.isnan(lead)], n) if (~np.isnan(lead)).any() else np.full(n, np.nan) for lead in x])
    return np.pad(x, ((0, 0), (0, max(0, n - x.shape[1]))), constant_values=np.nan)[:, :n]


class ECGFounder:
    def __init__(self, cfg: dict):
        sys.path.insert(0, str(REPO / "third_party" / "ecgfounder"))
        from net1d import Net1D
        self.labels = [t.strip() for t in (REPO / "third_party/ecgfounder/tasks.txt").read_text(encoding="utf-8").splitlines() if t.strip()]
        self.model = Net1D(in_channels=12, base_filters=64, ratio=1, filter_list=[64, 160, 160, 400, 400, 1024, 1024], m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
                           kernel_size=16, stride=2, groups_width=16, verbose=False, use_bn=False, use_do=False, n_classes=len(self.labels))
        checkpoint = torch.load(paths.resolve(cfg["weights"]), map_location="cpu", weights_only=False)  # the upstream file also pickles its scheduler, which weights_only=True refuses
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)

    def prepare(self, x: np.ndarray, fill: str) -> np.ndarray:
        x = np.nan_to_num(fit_length(x, 10 * FS, fill), nan=0.0)
        return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)


class HuBERTECG:
    def __init__(self, cfg: dict):
        sys.path.insert(0, str(REPO / "third_party"))
        import hubert_ecg  # noqa: F401  registers the model type with transformers
        from transformers import AutoModel
        self.labels = (REPO / "third_party/hubert_ecg/cardio_learning_labels.txt").read_text(encoding="utf-8").split()
        self.model = AutoModel.from_pretrained(str(paths.resolve(cfg["weights"])), num_labels=len(self.labels))
        self.band = ss.firwin(numtaps=int(0.3 * FS), cutoff=[0.05, 47], pass_zero=False, fs=FS)  # biosppy filter_signal(ftype="FIR", order=0.3*fs) as in upstream apply_filter

    def prepare(self, x: np.ndarray, fill: str) -> np.ndarray:
        out = np.full((12, 5 * FS), np.nan)
        for i, lead in enumerate(fit_length(x, 5 * FS, fill)):
            ok = ~np.isnan(lead)
            if ok.sum() <= 3 * len(self.band):
                continue
            seg = ss.filtfilt(self.band, [1.0], lead[ok])
            out[i, ok] = 2 * (seg - seg.min()) / (seg.max() - seg.min() + 1e-8) - 1
        out = np.where(np.isnan(out), np.nanmean(out) if np.isfinite(out).any() else 0.0, out)
        return ss.decimate(out.reshape(-1), 5).astype(np.float32)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch, attention_mask=None).logits


class ECGFM:
    def __init__(self, cfg: dict):
        sys.path.insert(0, str(REPO / "third_party"))
        from fairseq_signals.models import build_model_from_checkpoint
        with (REPO / "third_party/ecg_fm/label_def.csv").open(encoding="utf-8") as fh:
            self.labels = [r["name"] for r in csv.DictReader(fh)]
        self.model = build_model_from_checkpoint(checkpoint_path=str(paths.resolve(cfg["weights"])))

    def prepare(self, x: np.ndarray, fill: str) -> np.ndarray:
        out = np.zeros((12, 5 * FS))
        for i, lead in enumerate(x):
            ok = ~np.isnan(lead)
            if ok.sum() < 2 or np.ptp(lead[ok]) == 0:
                continue
            z = lead.copy()
            z[ok] = (lead[ok] - lead[ok].mean()) / (lead[ok].std() + 1e-8)
            out[i] = np.nan_to_num(fit_length(z[None], 5 * FS, fill)[0], nan=0.0)
        return out.astype(np.float32)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(source=batch)["out"]


MODELS = {"ecgfounder": ECGFounder, "hubert_ecg": HuBERTECG, "ecg_fm": ECGFM}


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    d = score[positive][:, None] - score[~positive][None, :]  # ponytail: all pairs, fine for a few thousand pages
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def summary(rows: list[dict], tasks: list[str], k: int) -> str:
    probs = np.array([[float(r[t]) for t in tasks] for r in rows])
    out = [f"pages {len(rows)}; highest mean output:"]
    out += [f"  {tasks[i]}: {probs[:, i].mean():.4f}" for i in np.argsort(-probs.mean(0))[:k]]
    labels = np.array([r["label"] for r in rows])
    for name, keep in [("all pages", np.ones(len(rows), bool)), ("pages with 12 leads ok", np.array([r["leads_ok"] == "12" for r in rows]))]:
        pjb = labels[keep] == "PJB"
        if pjb.all() or not pjb.any():
            continue
        a = np.array([auroc(probs[keep, i], pjb) for i in range(len(tasks))])
        out.append(f"\nPJB vs NORMAL, {name} ({pjb.sum()} PJB, {(~pjb).sum()} NORMAL); outputs ranked by |AUROC - 0.5|, AUROC above 0.5 = higher output on PJB")
        out.append("| Model label | AUROC | Mean on PJB | Mean on NORMAL |\n|---|---|---|---|")
        out += [f"| {tasks[i]} | {a[i]:.4f} | {probs[keep][pjb, i].mean():.4f} | {probs[keep][~pjb, i].mean():.4f} |" for i in np.argsort(-np.abs(a - 0.5))[:k]]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    cfg = yaml.safe_load(ap.parse_args().config.read_text(encoding="utf-8"))
    out_root = paths.resolve(cfg["out_dir"])
    log_dir = paths.resolve(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / f"{out_root.name}-{datetime.now():%Y%m%d}.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    log = logging.getLogger("classify_signal")
    model = MODELS[cfg["model"]](cfg)
    model.model.to(cfg["device"]).eval()
    fields = ["page", "relative_path", "label", "leads_ok"] + model.labels
    for dataset, run in cfg["runs"].items():
        out = out_root / dataset / "predictions.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            with out.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(fields)
        with out.open(encoding="utf-8") as fh:
            done = {r["page"] for r in csv.DictReader(fh)}
        todo = [p.parent for p in sorted(paths.resolve(run).glob("*/record.json")) if p.parent.name not in done]
        labels = {}
        for start in tqdm(range(0, len(todo), cfg["batch_size"]), desc=dataset, unit="batch"):
            batch = todo[start:start + cfg["batch_size"]]
            x = torch.from_numpy(np.stack([model.prepare(read_record(p), cfg["fill"]) for p in batch])).to(cfg["device"])
            with torch.no_grad():
                probs = torch.sigmoid(model(x)).cpu().numpy()
            rows = []
            for page_dir, p in zip(batch, probs):
                image = json.loads((page_dir / "record.json").read_text(encoding="utf-8"))["image"]
                folder, rel = source_key(image["source"])
                if folder not in labels:
                    labels[folder] = load_labels(folder, cfg["label_column"])
                rows.append([page_dir.name, rel, labels[folder].get(rel, ""), image["leads_ok"]] + [f"{v:.5f}" for v in p])
            with out.open("a", newline="", encoding="utf-8") as fh:  # written after the batch is scored, so a crash repeats at most one batch
                csv.writer(fh).writerows(rows)
            log.info("%s: %d pages scored", dataset, len(rows))
        with out.open(encoding="utf-8") as fh:
            print(f"\n{dataset}\n{summary(list(csv.DictReader(fh)), model.labels, cfg['top_k'])}")


if __name__ == "__main__":
    main()
