# Third-party source

Unmodified copies; each folder keeps its upstream license.

| Folder | Files | Source | Commit | License |
|---|---|---|---|---|
| `hubert_ecg` | `__init__.py`, `configuration.py`, `modeling.py`, `modeling_classification.py` (HuBERT-ECG model); `cardio_learning_labels.txt` is the 164 output labels exported from upstream `cardio-learning/cardio-learning-labels.pkl` (a numpy array of column names; the first three, filename, age, sex, dropped) | https://github.com/Edoar-do/HuBERT-ECG | `36061fc5a8bc0d3adaf9df9dd8ef551c6391a6e9` | CC BY-NC 4.0 |
| `fairseq_signals` | the `fairseq_signals` package (pure Python; its C++ and Cython extensions are not built and not needed to load and run ECG-FM) | https://github.com/Jwoo5/fairseq-signals | `f8f0ff1c788a82c2059cb452cd5462898867489e` | MIT |
| `ecg_fm` | `label_def.csv` (the 17 outputs of the MIMIC-IV-ECG fine-tuned model, in output order) | https://github.com/bowang-lab/ECG-FM | `9f926f1911bb9f24789b5c6407677d58ad753054` | MIT |
| `merl` | `utils_builder.py` (ECGCLIP), `vit1d.py`, `resnet1d.py`, `CKEPE_prompt.json` (the authors' 131 class prompts) | https://github.com/cheliu-computation/MERL-ICML2024 | `2a38649285e16eff75b69aeb64f2366b380c1a9e` | MIT |
| `ecgfounder` | `net1d.py` (Net1D model), `tasks.txt` (the 150 output labels in model order) | https://github.com/PKUDigitalHealth/ECGFounder | `04edac702b61c91face519774ddcc0cd712fef23` | MIT |

Weights are not stored here: `scripts/get_weights.sh` downloads them to `models/ecgfounder-pretrained/`, `models/hubert_ecg-cardiolearning_base-pretrained/` `models/ecg_fm-mimic_iv_ecg_finetuned-pretrained/`, `models/merl-pretrained/` and `models/medcpt_query_encoder-pretrained/`.
