# Dempster-Shafer Evidential Spiking Neural Network Framework (DS-SNN-MF)

Multi-view structural MRI (3D spiking encoder) and clinical features are fused with Dempster-Shafer theory to classify CN / MCI / AD. Every prediction comes with belief, plausibility, conflict (K) and ignorance mass.

- Datasets: OASIS-2 (default) and OASIS-3 (`--dataset oasis3`, see `docs/OASIS3.md`).
- Protocol: subject-level stratified CV, train-only clinical normalisation, last-visit evaluation, tabular baselines on identical folds, bootstrap confidence intervals.
- Proof-of-concept for the larger DS-SNN-MF framework. OASIS-2 has ~150 subjects, so fold-to-fold variation (std 0.07-0.10) is larger than most differences between variants.

## Modules

| File | Role |
|------|------|
| `config.py` | Single `cfg` object; per-dataset settings in `Config.DATASETS` |
| `DataModule.py` | Dataset-agnostic data layer: visit records, folds, clinical scaling, MRI tensors, datasets and collate |
| `OasisModule.py` | OASIS-2 loader |
| `Oasis3Module.py` | OASIS-3 loader (NITRC / BIDS layouts, scan-to-CDR matching) |
| `SNNModule.py` | PLIF 3D spiking encoder, clinical MLP |
| `DSModule.py` | Dempster-Shafer algebra: combination rules, discounting, belief / plausibility / pignistic, evidential layer, temporal accumulator |
| `FusionModule.py` | `DSSNMMF` model: views and clinical to a fused BBA and decision |
| `LossModule.py` | `DSSNNLoss` and its components |
| `MetricsModule.py` | Accuracy, F1, AUROC, Brier, ECE, selective accuracy |
| `TrainerModule.py` | Training loop and evaluation |
| `ExperimentModule.py` | CV / repeated CV / single split, baselines, bootstrap comparison, run folder |
| `main.py` | CLI |
| `OtherModules/mri_preprocess.py` | Skull strip, N4, MNI registration (batched, resumable) |
| `OtherModules/baselines.py` | Logistic regression and random forest on the same folds |
| `OtherModules/leaky_demo.py` | Deliberately leaky protocol, for methodological contrast only |
| `tests/` | Synthetic-data test suite |

## Usage

Install: Python 3.10+, `torch`, `numpy`, `pandas`, `scikit-learn`, `nibabel`, `openpyxl`, `pytest`. Preprocessing additionally needs `SimpleITK`, `templateflow` and the `hd-bet` CLI.

1. Place data under `data/oasis2/` (RAW parts plus the demographics spreadsheet) or `data/oasis3/` (layout in `docs/OASIS3.md`).
2. Preprocess the MRI once. Training reads the processed tensors and fails if they are missing.
   ```bash
   python -m OtherModules.mri_preprocess --dataset oasis2
   python -m OtherModules.mri_preprocess --dataset oasis3 --chunk 32 --workers 4
   ```
   It skips finished sessions, so interrupted runs can be restarted; `--offset` / `--limit` select a slice.
3. Run the experiment.
   ```bash
   python main.py                                   # OASIS-2, 5-fold CV
   python main.py --dataset oasis3 --repeats 3      # OASIS-3, 3x repeated CV
   python main.py --ablation clinical --single-split
   python main.py --train-visits all                # train on every visit, evaluate on the last
   python main.py --help
   ```
4. Baselines alone: `python -m OtherModules.baselines --dataset oasis2`.
5. Tests: `python -m pytest tests`. They use synthetic data and need no dataset or GPU.

Each run writes `runs/<dataset>_<ablation>_<timestamp>/` with `config.json`, `run.log`, per-fold checkpoints and metrics, `oof_predictions.csv` and `summary.json`. The summary holds per-fold and mean +/- std metrics, baseline results on the same folds, and bootstrap CIs for the difference between models.

More detail is in `docs/`: `Overview.md` (protocol and architecture), `Methodology.md`, `OASIS3.md`, and the earlier tuning notes.

## Caveats

- Hyperparameters were previously tuned while watching test folds (`docs/Fusion Retune Changelog.md`). Treat those numbers as optimistic. For new datasets, tune on validation folds only and run test CV once.
- Options added in the latest upgrade (`TRAIN_VISITS='all'`, OASIS-3 column mappings, batched HD-BET) are untested on real data.

---

Maintained by Abrar Fahyaz (University of Dhaka). Work in progress; the main idea and formulations are from Dr. Maybin K. Muyeba (University of Salford).
