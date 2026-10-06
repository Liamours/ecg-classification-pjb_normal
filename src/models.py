"""Pretrained open-weight ECG models, each behind one adapter with its upstream preprocessing and inference code.

An adapter has `prep` (a picklable callable run in the data-loader workers: a 12 x 2000 record in mV at 500 Hz, NaN where a
lead has no data, to the model's input array), `load(device)` and `forward(batch)` (the activated outputs). `labels` lists the
output names of a model with a label head; an encoder without one has `labels = None` and returns its embedding.
`fill` sets how a lead shorter than the model input reaches its length: `zero` leaves it missing and lets the model's
upstream rule fill it, `tile` repeats the lead's digitized stretch.

| Model | Upstream code used | Input |
|---|---|---|
| ecgfounder | Net1D (net1d.py), ptbxl_eval.py preprocessing | 12 leads, 10 s, 500 Hz, missing 0, one z-score over the array |
| hubert_ecg | HuBERTECGForClassification, utils.apply_filter and utils.scaling, dataset.py crop, mean fill, decimate | 12 leads flattened, 5 s, 100 Hz |
| ecg_fm | fairseq_signals build_model_from_checkpoint, ecg-transform pipeline of infer_quickstart.ipynb | 12 leads, 5 s, 500 Hz, per-lead z-score |
| merl | ECGCLIP (utils_builder.py), finetune_dataset.py ICBEB branch, zeroshot_val.get_class_emd | 12 leads, 10 s, 500 Hz, min-max [0, 1] |
| ecg_jepa | ecg_jepa encoder (models.load_encoder), ecg_data.py resampling | 8 leads, 10 s, 250 Hz, raw mV |
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import signal as ss

from src import paths

REPO = Path(__file__).resolve().parents[1]
THIRD = REPO / "third_party"
FS = 500
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def fit_length(x: np.ndarray, n: int, fill: str) -> np.ndarray:
    """Each lead to n samples: `tile` repeats its digitized stretch, `zero` pads with NaN; a lead with no data stays NaN."""
    if fill == "tile":
        return np.stack([np.resize(lead[~np.isnan(lead)], n) if (~np.isnan(lead)).any() else np.full(n, np.nan) for lead in x])
    return np.pad(x, ((0, 0), (0, max(0, n - x.shape[1]))), constant_values=np.nan)[:, :n]


def _path(sub: str) -> None:
    p = str(THIRD / sub) if sub else str(THIRD)
    if p not in sys.path:
        sys.path.insert(0, p)


# ---- preprocessing (picklable, runs in data-loader workers) ----------------------------------------------------------

class ECGFounderPrep:
    def __init__(self, fill: str):
        self.fill = fill

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(fit_length(x, 10 * FS, self.fill), nan=0.0)  # upstream np.nan_to_num(data, nan=0)
        return ((x - np.mean(x)) / (np.std(x) + 1e-8)).astype(np.float32)  # upstream z_score_normalization


class HuBERTPrep:
    def __init__(self, fill: str):
        self.fill = fill

    def __call__(self, x: np.ndarray) -> np.ndarray:
        _path("")
        from hubert_ecg.utils import apply_filter, scaling
        out = np.full((12, x.shape[1]), np.nan)
        for i, lead in enumerate(x):
            ok = ~np.isnan(lead)
            if ok.sum() > 3 * int(0.3 * FS):  # shorter stretches cannot be filtered by the 0.3 s FIR
                out[i, ok] = scaling(apply_filter(lead[ok][None], [0.05, 47], fs=FS))[0]  # upstream ecg_preprocessing on the stretch
        out = fit_length(out, 5 * FS, self.fill)  # dataset.py: first 5 s at 500 Hz
        out = np.where(np.isnan(out), np.nanmean(out) if np.isfinite(out).any() else 0.0, out)  # dataset.py NaN imputation
        return ss.decimate(out.reshape(-1), 5).astype(np.float32)  # dataset.py flatten + decimate to 100 Hz


class ECGFMPrep:
    def __init__(self, fill: str):
        self.fill = fill

    def __call__(self, x: np.ndarray) -> np.ndarray:
        from ecg_transform.inp import ECGInput, ECGInputSchema, ECGMetadata
        from ecg_transform.sample import ECGSample
        from ecg_transform.t.common import HandleConstantLeads, LinearResample, ReorderLeads
        from ecg_transform.t.cut import SegmentNonoverlapping
        from ecg_transform.t.scale import Standardize
        x = fit_length(x, 5 * FS, self.fill)
        means = np.array([np.nanmean(lead) if np.isfinite(lead).any() else 0.0 for lead in x])
        x = np.where(np.isnan(x), means[:, None], x)  # a missing stretch takes its lead's mean, so it standardizes to 0; a missing lead is constant and zeroed upstream
        meta = ECGMetadata(sample_rate=FS, num_samples=x.shape[1], lead_names=LEADS, unit=None, input_start=0, input_end=x.shape[1])
        schema = ECGInputSchema(sample_rate=FS, expected_lead_order=LEADS, required_num_samples=5 * FS)
        transforms = [ReorderLeads(expected_order=LEADS, missing_lead_strategy="raise"), LinearResample(desired_sample_rate=FS),
                      HandleConstantLeads(strategy="zero"), Standardize(), SegmentNonoverlapping(segment_length=5 * FS)]  # infer_quickstart.ipynb
        return ECGSample(ECGInput(x, meta), schema, transforms).out[0].astype(np.float32)


class MERLPrep:
    def __init__(self, fill: str):
        self.fill = fill

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(fit_length(x, 10 * FS, self.fill), nan=0.0)  # finetune_dataset.py ICBEB: zero padding to 5000
        x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)  # normalize to 0-1
        return x[[0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]].astype(np.float32)  # switch aVL and aVF to the MIMIC-IV-ECG order


class ECGJEPAPrep:
    def __init__(self, fill: str):
        self.fill = fill

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(fit_length(x, 10 * FS, self.fill), nan=0.0)[[0, 1, 6, 7, 8, 9, 10, 11]]  # reduced_lead: I, II, V1 to V6
        return ss.resample(x, 2500, axis=1).astype(np.float32)  # ecg_data.py: resample(wave, 2500, axis=1)


# ---- models ------------------------------------------------------------------------------------------------------------

class ECGFounder:
    def __init__(self, cfg: dict):
        self.cfg, self.prep = cfg, ECGFounderPrep(cfg["fill"])
        self.labels = [t.strip() for t in (THIRD / "ecgfounder/tasks.txt").read_text(encoding="utf-8").splitlines() if t.strip()]

    def load(self, device: str) -> None:
        _path("ecgfounder")
        from net1d import Net1D
        self.net = Net1D(in_channels=12, base_filters=64, ratio=1, filter_list=[64, 160, 160, 400, 400, 1024, 1024], m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
                         kernel_size=16, stride=2, groups_width=16, verbose=False, use_bn=False, use_do=False, n_classes=len(self.labels))  # ptbxl_eval.py
        checkpoint = torch.load(paths.resolve(self.cfg["weights"]), map_location="cpu", weights_only=False)  # the upstream file also pickles its scheduler, which weights_only=True refuses
        self.net.load_state_dict(checkpoint["state_dict"], strict=True)
        self.net.to(device).eval()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(batch))


class HuBERTECG:
    def __init__(self, cfg: dict):
        self.cfg, self.prep = cfg, HuBERTPrep(cfg["fill"])
        self.labels = (THIRD / "hubert_ecg/cardio_learning_labels.txt").read_text(encoding="utf-8").split()

    def load(self, device: str) -> None:
        _path("")
        import hubert_ecg  # noqa: F401  registers the model type with transformers
        from transformers import AutoModel
        self.net = AutoModel.from_pretrained(str(paths.resolve(self.cfg["weights"])), num_labels=len(self.labels)).to(device).eval()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(batch, attention_mask=None).logits)  # evaluate.py: attention_mask=None, sigmoid for multi_label


class ECGFM:
    def __init__(self, cfg: dict):
        self.cfg, self.prep = cfg, ECGFMPrep(cfg["fill"])
        with (THIRD / "ecg_fm/label_def.csv").open(encoding="utf-8") as fh:
            self.labels = [r["name"] for r in csv.DictReader(fh)]

    def load(self, device: str) -> None:
        _path("")
        from fairseq_signals.models import build_model_from_checkpoint
        self.net = build_model_from_checkpoint(checkpoint_path=str(paths.resolve(self.cfg["weights"]))).to(device).eval()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(source=batch)["out"])


class MERL:
    def __init__(self, cfg: dict):
        self.cfg, self.prep = cfg, MERLPrep(cfg["fill"])
        self.prompts = json.loads((THIRD / "merl/CKEPE_prompt.json").read_text(encoding="utf-8")) | cfg["prompts"]
        self.labels = list(self.prompts)

    def load(self, device: str) -> None:
        _path("merl")
        import utils_builder
        network = {"ecg_model": self.cfg["arch"], "num_leads": 12, "text_model": str(paths.resolve(self.cfg["text_model"])), "free_layers": 6, "feature_dim": 768,
                   "projection_head": {"mlp_hidden_size": 256, "projection_size": 256}}
        self.net = utils_builder.ECGCLIP(network)
        self.net.load_state_dict(torch.load(paths.resolve(self.cfg["weights"]), map_location="cpu", weights_only=True), strict=True)
        self.net.to(device).eval()
        emb = []
        with torch.no_grad():  # zeroshot_val.get_class_emd: lower-cased prompt, text encoder, projection, unit norm
            for text in self.prompts.values():
                tok = self.net.tokenizer([text.lower()], add_special_tokens=True, truncation=True, max_length=256, padding="max_length", return_tensors="pt").to(device)  # utils_builder._tokenize; batch_encode_plus is gone in transformers 5
                e = self.net.proj_t(self.net.get_text_emb(tok.input_ids, tok.attention_mask))
                emb.append((e / e.norm(dim=-1, keepdim=True)).mean(0))
        self.text = torch.stack([e / e.norm() for e in emb], dim=1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        emb = self.net.ext_ecg_emb(batch)
        return (emb / emb.norm(dim=-1, keepdim=True)) @ self.text  # zeroshot_val.get_ecg_emd, softmax_eval=True: raw cosine similarity


class ECGJEPA:
    def __init__(self, cfg: dict):
        self.cfg, self.prep, self.labels = cfg, ECGJEPAPrep(cfg["fill"]), None

    def load(self, device: str) -> None:
        _path("ecg_jepa")
        from ecg_jepa import ecg_jepa
        params = {"encoder_embed_dim": 768, "encoder_depth": 12, "encoder_num_heads": 16, "predictor_embed_dim": 384, "predictor_depth": 6,
                  "predictor_num_heads": 12, "c": 8, "pos_type": "sincos", "mask_scale": (0, 0), "leads": list(range(8))}  # models.load_encoder
        self.net = ecg_jepa(**params).encoder
        self.net.load_state_dict(torch.load(paths.resolve(self.cfg["weights"]), map_location="cpu", weights_only=True)["encoder"], strict=True)
        self.net.to(device).eval()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.net.representation(batch)  # mean of the final token features


MODELS = {"ecgfounder": ECGFounder, "hubert_ecg": HuBERTECG, "ecg_fm": ECGFM, "merl": MERL, "ecg_jepa": ECGJEPA}
