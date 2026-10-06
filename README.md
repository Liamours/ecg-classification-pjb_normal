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

## Signal inference with ECGFounder

`src.classify_signal` scores each digitized page (`record.csv` of a canonical digitize run) with the pretrained 12-lead ECGFounder (Net1D, 150 sigmoid outputs, labels in `third_party/ecgfounder/tasks.txt`), no fine-tuning. Preprocessing follows the upstream `ptbxl_eval.py`: lead order I to V6, missing values 0, one z-score over the whole array. The model takes 10 s at 500 Hz; `fill` sets how the 4 s record reaches 10 s (`zero` pads, `tile` repeats each lead's digitized stretch). Output `<out_dir>/<dataset>/predictions.csv` (page, relative_path, label, leads_ok, one column per output); the summary ranks outputs by AUROC between PJB and NORMAL pages.

```
bash scripts/get_weights.sh
uv run python -m src.classify_signal --config configs/classify_signal.yml
uv run python -m src.classify_signal --config configs/classify_signal_tile.yml
```
