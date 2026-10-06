# Fine-tuning of every model in configs/finetune.yml on the nested folds (src.finetune).
# Run from anywhere in PowerShell:  powershell -ExecutionPolicy Bypass -File <repo>\scripts\run_finetune.ps1
# Resumable: rerun the same line after a stop; finished folds are skipped and an unfinished fold continues from its last epoch.
# Logs go to the project logs\ folder (finetune-<date>.log).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
uv run python -W ignore -m src.finetune --config configs/finetune.yml
if ($LASTEXITCODE -ne 0) { throw "fine-tuning failed" }
