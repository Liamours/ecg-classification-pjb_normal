#!/usr/bin/env bash
# Pretrained ECGFounder 12-lead weights (MIT, Hugging Face PKUDigitalHealth/ECGFounder) into the project models/ folder.
set -euo pipefail
out="$(dirname "$0")/../../../models/ecgfounder-pretrained"
mkdir -p "$out"
[ -f "$out/12_lead_ECGFounder.pth" ] || curl -L --fail --progress-bar -o "$out/12_lead_ECGFounder.pth" https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/main/12_lead_ECGFounder.pth
