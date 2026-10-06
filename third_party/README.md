# Third-party source

Unmodified copies; each folder keeps its upstream license.

| Folder | Files | Source | Commit | License |
|---|---|---|---|---|
| `hubert_ecg` | `__init__.py`, `configuration.py`, `modeling.py`, `modeling_classification.py` (HuBERT-ECG model); `cardio_learning_labels.txt` is the 164 output labels exported from upstream `cardio-learning/cardio-learning-labels.pkl` (a numpy array of column names; the first three, filename, age, sex, dropped) | https://github.com/Edoar-do/HuBERT-ECG | `36061fc5a8bc0d3adaf9df9dd8ef551c6391a6e9` | CC BY-NC 4.0 |
| `ecgfounder` | `net1d.py` (Net1D model), `tasks.txt` (the 150 output labels in model order) | https://github.com/PKUDigitalHealth/ECGFounder | `04edac702b61c91face519774ddcc0cd712fef23` | MIT |

Weights are not stored here: `scripts/get_weights.sh` downloads them to `models/ecgfounder-pretrained/` and `models/hubert_ecg-cardiolearning_base-pretrained/`.
