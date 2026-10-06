#!/usr/bin/env bash
# Pretrained ECGFounder 12-lead weights (MIT, Hugging Face PKUDigitalHealth/ECGFounder) into the project models/ folder.
set -euo pipefail
out="$(dirname "$0")/../../../models/ecgfounder-pretrained"
mkdir -p "$out"
[ -f "$out/12_lead_ECGFounder.pth" ] || curl -L --fail --progress-bar -o "$out/12_lead_ECGFounder.pth" https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/main/12_lead_ECGFounder.pth
# HuBERT-ECG BASE fine-tuned on Cardio-Learning, 164 outputs (CC BY-NC 4.0, Hugging Face Edoardo-Coppola/HuBERT-ECG-SFT-CardioLearning-base)
hub="$(dirname "$0")/../../../models/hubert_ecg-cardiolearning_base-pretrained"
mkdir -p "$hub"
for f in config.json model.safetensors; do
  [ -f "$hub/$f" ] || curl -L --fail --progress-bar -o "$hub/$f" "https://huggingface.co/Edoardo-Coppola/HuBERT-ECG-SFT-CardioLearning-base/resolve/main/$f"
done
# ECG-FM fine-tuned on MIMIC-IV-ECG, 17 outputs (MIT, Hugging Face wanglab/ecg-fm)
fm="$(dirname "$0")/../../../models/ecg_fm-mimic_iv_ecg_finetuned-pretrained"
mkdir -p "$fm"
for f in mimic_iv_ecg_finetuned.pt mimic_iv_ecg_finetuned.yaml; do
  [ -f "$fm/$f" ] || curl -L --fail --progress-bar -o "$fm/$f" "https://huggingface.co/wanglab/ecg-fm/resolve/main/$f"
done
# MERL ECG-text checkpoints (MIT, the authors' Google Drive folder from github.com/cheliu-computation/MERL-ICML2024) and its text encoder MedCPT-Query-Encoder (public domain)
merl="$(dirname "$0")/../../../models/merl-pretrained"
[ -f "$merl/vit_tiny_best_ckpt.pth" ] || uv run --with gdown python -m gdown --folder "https://drive.google.com/drive/folders/13wb4DppUciMn-Y_qC2JRWTbZdz3xX0w2" -O "$merl"
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('ncbi/MedCPT-Query-Encoder', local_dir='$(dirname "$0")/../../../models/medcpt_query_encoder-pretrained')"
# ECG-JEPA encoders, multi-block and random masking (MIT, the authors' Google Drive links from github.com/sehunfromdaegu/ECG_JEPA)
jepa="$(dirname "$0")/../../../models/ecg_jepa-pretrained"
mkdir -p "$jepa"
[ -f "$jepa/multiblock_epoch100.pth" ] || uv run --with gdown python -m gdown 1gMOT4xjQQg0GZkY1iE6NuDzua4ALw00l -O "$jepa/"
[ -f "$jepa/random_epoch100.pth" ] || uv run --with gdown python -m gdown 1mh-XL0XOvvhFbhvuZ9c2KnTHa9B4F3Wx -O "$jepa/"
