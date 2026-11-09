# Dempster-Shafer Evidential Spiking Neural Network Framework (DS-SNN-MF)

Multi-view structural MRI, PET, CT and clinical features are fused with Dempster-Shafer theory to classify CN / MCI / AD. Every prediction comes with belief, plausibility, conflict (K) and ignorance mass.

- Datasets: OASIS-3
- Protocol: subject-level stratified CV, train-only clinical normalisation, last-visit evaluation, tabular baselines on identical folds, bootstrap confidence intervals.

## Modules

| File | Role |
|------|------|
| `config.py` | Single `cfg` object; per-dataset settings in `Config.DATASETS` |
| `MODULE.DataModule.py` | Dataset-agnostic data layer: visit records, folds, clinical scaling, MRI tensors, datasets and collate |
| `MODULE.SNNModule.py` | PLIF 3D spiking encoder, clinical MLP |
| `MODULE.DSModule.py` | Dempster-Shafer algebra: combination rules, discounting, belief / plausibility / pignistic, evidential layer, temporal accumulator |
| `MODULE.FusionModule.py` | `DSSNMMF` model: views and clinical to a fused BBA and decision |
| `MODULE.LossModule.py` | `DSSNNLoss` and its components |
| `MODULE.MetricsModule.py` | Accuracy, F1, AUROC, Brier, ECE, selective accuracy |
| `MODULE.TrainerModule.py` | Training loop and evaluation |
| `Oasis3Module.py` | OASIS-3 loader (NITRC / BIDS layouts, scan-to-CDR matching) |
| `ExperimentModule.py` | CV / repeated CV / single split, baselines, bootstrap comparison, run folder |
| `main.py` | CLI |
| `OtherModules/mri_preprocess.py` | Skull strip, N4, MNI registration (batched, resumable) |
| `OtherModules/baselines.py` | Logistic regression and random forest on the same folds |
| `OtherModules/leaky_demo.py` | Deliberately leaky protocol, for methodological contrast only |
| `tests/` | Synthetic-data test suite |

## Usage

Install: Python 3.10+, `torch`, `numpy`, `pandas`, `scikit-learn`, `nibabel`, `openpyxl`, `pytest`. Preprocessing additionally needs `SimpleITK`, `templateflow` and the `hd-bet` CLI.

1. Place data under `data/oasis3/` (layout in `docs/OASIS3.md`).
2. Preprocess the MRI once. Training reads the processed tensors and fails if they are missing.
   ```bash
   python -m OtherModules.mri_preprocess
   ```
   It skips finished sessions, so interrupted runs can be restarted; `--offset` / `--limit` select a slice.
3. Run the experiment.
   ```bash
   python main.py
   ```
4. Baselines alone: `python -m OtherModules.baselines`.
5. Tests: `python -m pytest tests`. They use synthetic data and need no dataset or GPU.

Each run writes `runs/<dataset>_<ablation>_<timestamp>/` with `config.json`, `run.log`, per-fold checkpoints and metrics, `oof_predictions.csv` and `summary.json`. The summary holds per-fold and mean +/- std metrics, baseline results on the same folds, and bootstrap CIs for the difference between models.

More detail is in `docs/`.
---

Maintained by Abrar Fahyaz (University of Dhaka). Work in progress; the main idea and formulations are from Dr. Maybin K. Muyeba (University of Salford).
