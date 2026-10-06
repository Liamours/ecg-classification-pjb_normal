# ecg-classification-pjb_normal

NORMAL vs PJB (congenital heart disease) classification from ECG images and digitized signals. Code, configs, and scripts only; datasets, checkpoints, and outputs live in the project root folders (`datasets/`, `models/`, `results/inferences/`, `logs/`), reached through `configs/paths.yml` (`@inferences/<run>`, `@mac400-scan/...`).

## Flow

```
page image -> digitize (repo ecg-digitization-synthetic_realistic) -> run folder: per panel json and csv
          -> src.records -> one record per page: 12 leads x samples, label, usable flag
          -> classifier (not written yet) -> NORMAL or PJB
```

The record is the contract between the two repos. `src.records` reads a digitize run and writes, per page, `records/<page>.npz`:

| Array | Shape | Meaning |
|---|---|---|
| `signal` | 12 x (`duration_s` x `fs`) | mV per lead in the order I, II, III, aVR, aVL, aVF, V1 to V6; NaN where there is no data |
| `ok` | 12 | the lead passed the digitizer's quality flag |
| `present` | 12 | the lead was digitized at all |

and `index.csv` with one row per page: dataset, path, label (from the dataset's `_labels/manifest.csv`), leads present, leads ok, and `usable` (at least `usable.min_ok_leads` leads ok, `configs/records.yml`). The count of usable records per label is the number that says how much of a dataset the classifier can train or be tested on.

## Run

```
uv run python -m src.records --config configs/records.yml --run @inferences/digitize_full/mac400 --out-dir @datasets/training/records-digitize_full-mac400
uv run --with pytest python -m pytest tests -q
```

Resumable: a page already in `index.csv` is skipped. A log goes to `logs/records-<out-dir name>-<date>.log`.

## Image-direct inference

`src.classify_image` runs a trained YOLO-cls model (`configs/classify_image.yml`, default the 2026-09-08 NORMAL/PJB model on whole page photos at 224 px) on every page with an ECG in each listed dataset's manifest and writes `<out_dir>/<dataset>/predictions.csv` (relative_path, label, pred, p_pjb), then prints the counts and, for labeled pages, recall per label. Resumable per page; log in `logs/`.

```
uv run python -m src.classify_image --config configs/classify_image.yml
```

## Signal inference with pretrained models

`src.infer` runs one pretrained open-weight model on the digitized records (`record.csv` of a canonical digitize run) and saves every page's output: `<out_dir>/<dataset>/predictions.csv` for a model with a label head (one column per output), `embeddings.csv` for an encoder without one. Each model is an adapter in `src.models` that calls its authors' own model class and preprocessing (`third_party/`, licenses in `third_party/README.md`):

| Config | Model | Output |
|---|---|---|
| `ecgfounder` | ECGFounder 12-lead, 10 s at 500 Hz | 150 labels |
| `hubert_ecg` | HuBERT-ECG BASE fine-tuned on Cardio-Learning, 5 s at 100 Hz | 164 labels |
| `ecg_fm` | ECG-FM fine-tuned on MIMIC-IV-ECG, 5 s at 500 Hz | 17 labels |
| `merl_res18`, `merl_vit_tiny` | MERL ECG-text model, zero-shot | cosine similarity to 146 text prompts |
| `ecg_jepa_multiblock`, `ecg_jepa_random` | ECG-JEPA encoder, 8 leads, 10 s at 250 Hz | 768-dimension embedding |

Configs ending in `_tile` repeat each lead's digitized stretch to the model's input length instead of leaving the rest missing. Runs log progress, pages per second and ETA to `logs/`, resume from their output file, and preprocess in data-loader workers while the model runs on the GPU.

```
bash scripts/get_weights.sh
uv run python -m src.infer --config configs/infer/ecgfounder.yml
bash scripts/run_infer_all.sh
```

## Choosing the label schema from the model outputs

`src.label_schema` takes the labeled pages, and for every detailed defect and every ACC-CHD group (`configs/label_schema.yml`) against the NORMAL pages finds the model output with the highest |AUROC - 0.5| over all listed runs, next to its chance level from shuffled labels. No model is trained. The table goes to `results/analyses/label_schema/label_schema.csv`.

```
uv run python -m src.label_schema --config configs/label_schema.yml
```

## Linear probe on encoders without a label head

`src.probe` fits an L2 logistic regression on the saved embeddings for PJB vs NORMAL and for every defect and ACC-CHD group against NORMAL, with folds that hold out one collection batch. Saved per dataset: `probe_scores.csv` (out-of-fold score per page on the labeled set; on other sets the score of a probe fit on all labeled pages) and `probe_auroc.csv`.

```
uv run python -m src.probe --config configs/probe/ecg_jepa_multiblock.yml
uv run python -m src.probe --config configs/probe/ecg_jepa_random.yml
```

## Linear probe (all models)

`scripts/run_linear_probe.ps1` runs, in PowerShell, the embedding inference of every model with an adapter (`configs/infer/*_emb.yml` and the ECG-JEPA configs; `output: embedding` in a config makes ECGFounder and ECG-FM write their embedding) and then `src.linear_probe` (`configs/linear_probe.yml`): per model, repeat and outer fold of `manifest_folds.csv`, one logistic regression per label of the hierarchical schema on the standardized embedding, the L2 strength chosen per label on val by average precision, the threshold per label chosen on val to maximize F1, the test fold scored once; then a probe on every labeled page scores the scans. Output in `results/inferences/linear_probe/<model>/` (`oof_r<repeat>.csv`, `mac400-scan_predictions.csv`, `parts/`) and `results/analyses/linear_probe/` (`summary.csv`, `per_label.csv`, F1 by level). One progress line with ETA, logs in `logs/`, every step resumable.

```
.\scripts\run_linear_probe.ps1
```

## Fine-tuning

`scripts/run_finetune.ps1` runs `src.finetune` (`configs/finetune.yml`) in PowerShell: per model (ECGFounder, ECG-JEPA, ECG-FM), repeat and outer fold of `manifest_folds.csv`, a new linear output layer on the model's embedding and the whole network trained on the train pages (AdamW, backbone 1e-5 and head 1e-4, BCE with per-label positive weights, mixed precision, amplitude, noise and shift augmentation), the epoch with the best validation micro F1 kept, one threshold per level chosen on validation with a diagnosis kept only when its group is predicted (`src.thresholds`), the test fold scored once; then one network per model trained on every labeled page scores the scans and is kept (`final.pt`). Output in `results/inferences/finetune/<model>/` and `results/analyses/finetune/` (F1 by level). One progress line with ETA over units plus an epoch line, a log in `logs/`, a checkpoint after every epoch so a stopped fold continues where it was.

```
powershell -ExecutionPolicy Bypass -File scripts\run_finetune.ps1
```
