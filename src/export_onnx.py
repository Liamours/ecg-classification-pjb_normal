"""The fine-tuned ECGFounder classifier (src.finetune, `model.pt`) as ONNX for ONNX Runtime on a phone, with static int8.

Graph: input `ecg` (1, 12, 5000) float32, already prepared by ECGFounderPrep (each lead's digitized stretch tiled to 10 s at
500 Hz, NaN to 0, one z-score over the array); output `scores` (1, 38), raw scores in the order of the checkpoint's `labels`.
The sigmoid and the hierarchical thresholds (src.thresholds.predict) stay outside the graph. The network is rebuilt from
ECGFounder's own Net1D (third_party/ecgfounder) and the checkpoint's `backbone.*` and `head.*` weights, the layers
src.finetune.Net runs, so the export needs none of the other model adapters. Writes into `out_dir` (configs/export_onnx.yml):

- `classifier.onnx`, `classifier.ort`: float32; `.ort` is ONNX Runtime's mobile format with graph optimizations applied
- `classifier_int8.onnx`, `classifier_int8.ort`: static int8 in QDQ format, per-channel weights, QInt8 activations and weights
  with reduce_range off (the ARM choice of the `quantize_static` docstring), calibrated on training pages of the model's split
- `labels.json`: the 38 output names in order and the two thresholds

Exported with the torch.export-based exporter at a fixed shape. The `.ort` files are written with the ONNX Runtime version
of the Android app (`android_ort_version`); the script stops if the installed version differs. Parity: every file on the
prepared inputs of the test pages and of the scans, against the probabilities the PyTorch model wrote (largest probability
difference, pages whose predicted label set changes).

Runs in the digitization repo's GPU environment (../../.venvs/digitize_gpu: ONNX Runtime 1.30 needs Python 3.11 or later):
    PYTHONPATH=. ../../.venvs/digitize_gpu/Scripts/python.exe -m src.export_onnx --config configs/export_onnx.yml
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import yaml
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from src import paths
from src.thresholds import parents, predict

REPO = Path(__file__).resolve().parents[1]


class Classifier(torch.nn.Module):
    """Net1D's embedding, then the 38-label linear layer: what src.finetune.Net computes for ECGFounder."""

    def __init__(self, n_labels: int, n_tasks: int):
        super().__init__()
        sys.path.insert(0, str(REPO / "third_party" / "ecgfounder"))
        from net1d import Net1D

        self.backbone = Net1D(in_channels=12, base_filters=64, ratio=1, filter_list=[64, 160, 160, 400, 400, 1024, 1024], m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
                              kernel_size=16, stride=2, groups_width=16, verbose=False, use_bn=False, use_do=False, n_classes=n_tasks, return_features=True)
        self.head = torch.nn.Linear(1024, n_labels)

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(ecg)[1])


class Pages(CalibrationDataReader):
    def __init__(self, x: np.ndarray):
        self.items = iter(x)

    def get_next(self) -> dict | None:
        item = next(self.items, None)
        return None if item is None else {"ecg": item[None]}


def to_ort(path: Path) -> None:
    subprocess.run([sys.executable, "-m", "onnxruntime.tools.convert_onnx_models_to_ort", str(path), "--output_dir", str(path.parent),
                    "--optimization_style", "Fixed"], check=True, capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if ort.__version__ != cfg["android_ort_version"]:
        sys.exit(f"onnxruntime {ort.__version__} is installed, the app uses {cfg['android_ort_version']}: .ort files must be written with the app's version")
    torch.set_num_threads(cfg["threads"])
    train_cfg = yaml.safe_load((REPO / cfg["train_config"]).read_text(encoding="utf-8"))
    ckpt = torch.load(paths.resolve(cfg["checkpoint"]), map_location="cpu", weights_only=True)
    labels, thresholds = ckpt["labels"], ckpt["thresholds"]
    n_tasks = len([t for t in (REPO / "third_party/ecgfounder/tasks.txt").read_text(encoding="utf-8").splitlines() if t.strip()])
    net = Classifier(len(labels), n_tasks)
    net.load_state_dict(ckpt["model"], strict=False)   # the checkpoint holds only backbone.* and head.*; Net1D's own unused output layer is absent
    net.eval()
    out = paths.resolve(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels.json").write_text(json.dumps({"labels": labels, "thresholds": thresholds}, indent=1), encoding="utf-8")

    fp32 = out / "classifier.onnx"
    torch.onnx.export(net, (torch.zeros(1, 12, 5000),), str(fp32), input_names=["ecg"], output_names=["scores"], opset_version=cfg["opset"], dynamo=True)
    to_ort(fp32)

    phone = np.load(paths.resolve(cfg["inputs"]["mac400-phone_photo"]))
    folds = pd.read_csv(paths.resolve(train_cfg["folds"]))
    split = folds[(folds["repeat"] == train_cfg["split"]["repeat"]) & (folds["fold"] == train_cfg["split"]["fold"])]
    role = dict(zip(split.relative_path, split.role))
    rel = list(phone["relative_path"])
    train = [i for i, r in enumerate(rel) if role.get(r) == "train"]
    calib = np.random.default_rng(cfg["seed"]).choice(train, cfg["calibration_pages"], replace=False)
    pre, int8 = out / "classifier_pre.onnx", out / "classifier_int8.onnx"
    quant_pre_process(str(fp32), str(pre))
    quantize_static(str(pre), str(int8), Pages(phone["x"][calib]), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, reduce_range=False)
    pre.unlink()
    to_ort(int8)

    parent = parents(labels, REPO / train_cfg["groups_config"], train_cfg["merge"], train_cfg["unknown_label"])
    sets = []
    for name, (npz, csv) in cfg["parity"].items():
        z, ref_df = np.load(paths.resolve(npz)), pd.read_csv(paths.resolve(csv)).set_index("relative_path")
        idx = [i for i, r in enumerate(z["relative_path"]) if r in ref_df.index]
        ref = ref_df.loc[[z["relative_path"][i] for i in idx], [f"{l} prob" for l in labels]].to_numpy()
        sets.append((name, z["x"][idx], ref))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = cfg["threads"], 1
    print(f"int8 calibrated on {len(calib)} training pages\n\n| File | MB | Pages | Largest probability difference | Pages whose label set changes |\n|---|---|---|---|---|")
    for file in ("classifier.ort", "classifier_int8.ort"):
        sess = ort.InferenceSession(str(out / file), opts, providers=["CPUExecutionProvider"])
        for name, x, ref in sets:
            prob = 1 / (1 + np.exp(-np.concatenate([sess.run(None, {"ecg": p[None]})[0] for p in x])))
            changed = (predict(prob, thresholds, parent) != predict(ref, thresholds, parent)).any(axis=1).sum()
            print(f"| {file} | {(out / file).stat().st_size / 2**20:.1f} | {name}: {len(x)} | {np.abs(prob - ref).max():.5f} | {changed} |")


if __name__ == "__main__":
    main()
