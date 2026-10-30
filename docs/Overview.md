# DS-SNN-MF Overview

Classifies CN / MCI / AD from structural MRI and clinical features. Evidence from each source is expressed as a Dempster-Shafer basic belief assignment (BBA) over Θ = {CN, MCI, AD}, combined with a conflict-aware rule, and turned into a decision with explicit uncertainty.

Labels come from the Clinical Dementia Rating at each visit: 0 gives CN, 0.5 gives MCI, 1 or more gives AD. CDR and MMSE are never used as inputs.

## Pipeline

1. **Evidence sources.**
   - MRI views are cropped from the registered T1 volume, passed through a shared 3D spiking encoder (PLIF neurons, SEW-ADD residual blocks, T=8 timesteps) and mapped by an evidential head to a BBA. Evidence is `softplus`; `m({c}) = e_c / S`, `m(Θ) = |Θ| / S`.
   - Clinical features pass through an MLP and an evidential head to a second BBA.
2. **Fusion.** MRI views are folded together first, then fused with the clinical BBA. Dempster's rule is blended with a high-conflict rule by a gate `sigmoid(γ(K − κ))`, where K is the conflict mass. The high-conflict rule is `yager`, `ordinal` or `pcr5` (`cfg.CONFLICT_RULE`). Sources are discounted by branch reliability beforehand.
3. **Decision.** Pignistic probability of the fused BBA. As K rises, the decision is blended towards an auxiliary clinical-only head (`DECISION_BLEND_MAX`).
4. **Longitudinal accumulation.** `TemporalAccumulator` can fold per-visit BBAs over time; evaluation uses each subject's last visit.

## Evaluation protocol

- Subject-level stratified k-fold CV (`N_FOLDS`), optionally repeated (`REPEATS`, seed `SPLIT_SEED + r`). All visits of a subject stay in one split.
- Clinical normalisation is fitted on the training subjects of each fold only.
- Evaluation uses the last visit of each test subject.
- Logistic regression and random forest run on the same folds.
- Reported: accuracy, macro-F1, per-class F1, balanced accuracy, AUROC, Brier, ECE; selective accuracy at `SELECTIVE_COVERAGE` when ranking by `Pl − Bel` versus max probability; mean K, gate and uncertainty.
- Out-of-fold predictions are pooled and bootstrapped over subjects (`BOOTSTRAP_ROUNDS`) to give CIs for differences between models.

`TRAIN_VISITS` selects the training samples: `last` uses one per subject, `all` uses every visit with its own label. OASIS-2 defaults to `last`, OASIS-3 to `all`.

## Results so far (OASIS-2)

From `Experimental Results/`: clinical-only about 0.54 macro-F1, random forest about 0.52, fusion about 0.47-0.52, MRI-only about 0.30-0.42. Fold std is 0.07-0.10 on roughly 30 test subjects, so these differences are not statistically separable. Hyperparameters had been retuned while watching test folds, so these numbers are optimistic.

Scores below published values are expected: three classes including MCI, subject-level splits, no label leakage, and a small cohort. `OtherModules/leaky_demo.py` shows how a leaky protocol (CDR/MMSE inputs, visit-level splits) gives inflated validation scores; do not report those.

## Where things live

| Topic | File |
|-------|------|
| Settings, dataset specs | `config.py` |
| Loading, folds, scaling | `DataModule.py`, `OasisModule.py`, `Oasis3Module.py` |
| BBA algebra | `DSModule.py` |
| Model, loss | `FusionModule.py`, `LossModule.py` |
| Training, metrics | `TrainerModule.py`, `MetricsModule.py` |
| CV driver and run outputs | `ExperimentModule.py` |
| Formal description | `Methodology.md` |
| OASIS-3 setup | `OASIS3.md` |

## Open items

- Registered, multi-view MRI has not been validated at scale; the MRI branch is currently the weakest source.
- A larger cohort (OASIS-3) is needed before any claim that fusion beats the tabular baseline.
- Pretrained 3D encoders or distillation are possible if the SNN remains sample-limited.
