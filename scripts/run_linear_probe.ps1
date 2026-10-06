# Linear probe of every model with an adapter: embeddings first (src.infer), then the probe (src.linear_probe).
# Run from anywhere in PowerShell:  .\scripts\run_linear_probe.ps1
# Every step is resumable: rerun the same line after a stop and finished work is skipped. Logs go to the project logs\ folder.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$embeddings = @("ecgfounder_emb", "ecgfounder_tile_emb", "ecg_fm_emb", "ecg_fm_tile_emb", "ecg_jepa_multiblock", "ecg_jepa_random")
$i = 0
foreach ($name in $embeddings) {
    $i++
    Write-Host "[$i/$($embeddings.Count)] embeddings: $name"
    uv run python -W ignore -m src.infer --config "configs/infer/$name.yml"
    if ($LASTEXITCODE -ne 0) { throw "embeddings failed: $name" }
}
Write-Host "linear probe"
uv run python -W ignore -m src.linear_probe --config configs/linear_probe.yml
if ($LASTEXITCODE -ne 0) { throw "linear probe failed" }
