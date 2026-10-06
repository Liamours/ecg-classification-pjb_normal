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
