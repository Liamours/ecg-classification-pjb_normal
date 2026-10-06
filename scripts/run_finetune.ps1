# Fine-tuning of the models in configs/finetune.yml on the nested folds (src.finetune); every model, or only those after --models.
# Run from anywhere in PowerShell:  powershell -ExecutionPolicy Bypass -File <repo>\scripts\run_finetune.ps1 --models ecgfounder
# Resumable: rerun the same line after a stop; finished folds are skipped and an unfinished fold continues from its last epoch.
# Logs go to the project logs\ folder (finetune-<date>.log).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
uv run python -W ignore -m src.finetune --config configs/finetune.yml @args
if ($LASTEXITCODE -ne 0) { throw "fine-tuning failed" }
