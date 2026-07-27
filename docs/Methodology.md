# Methodology (OASIS-2 DS–SNN)

Concise technical notes on the honest evaluation pipeline, leakage controls, the spiking encoder, and Dempster–Shafer fusion.

---

## 1. Pipeline overview

**Task.** 3-class diagnosis **CN / MCI / AD** from OASIS-2 longitudinal T1 MRI + tabular clinical features. Labels come from per-visit **CDR** (0 → CN, 0.5 → MCI, ≥1 → AD). Primary target = **final-visit** class.

**Protocol.** Subject-level **5-fold stratified CV** (`SPLIT_SEED`). Per fold: test = fold \(i\); remaining subjects → stratified train / val (~15% val). Training uses **last visit only** per subject (`CrossSectionalWrapper`), matching the RF subject-final-visit baseline.

**Model (`DSSNMMF`).**

| Branch | Encoder | Output |
|--------|---------|--------|
| MRI | Shared 3D SNN → evidential head | BBA(s) over \(\Theta=\{CN,MCI,AD\}\) |
| Clinical | MLP → evidential head | One BBA |
| Fusion | Conflict-aware Dempster / Yager | Fused BBA → Bel, Pl, BetP |
| Decision | Blend of BetP and clinical-only aux softmax | \(\hat{y} = \arg\max[(1-\alpha)\mathrm{BetP} + \alpha\,p_{\mathrm{aux}}]\) |

Ablations: `full` | `clinical` | `mri` (`config.ABLATION`).

**Loss.** Focal / CE on the blended decision, aux CE, evidential DS utility, ignorance anchoring, mild conflict regularisation. Dual LR: clinical (+ aux) hot, MRI cold.

**Reporting.** Test accuracy and **macro-F1** (mean ± std). Val is early-stopping only.

---

## 2. Leakage prevention (and where it lives)

| Risk | Prevention | Code |
|------|------------|------|
| **Subject leakage** | Split on subject indices; all visits of a subject stay in one of train/val/test | `make_stratified_folds()` in `OasisModule.py`; CV loop in `main.py` |
| **Visit / slice leakage** | No visit-level or 2D-slice splits; one 3D volume per session; train sample = last visit only | `CrossSectionalWrapper`; MRI loaders |
| **Label leakage** | **CDR / MMSE never used as inputs**; labels only | `CLINICAL_COLS` in `config.py`; `_clinical_value` only reads those cols |
| **Norm leakage** | Clinical mean/std (+ NaN impute) fit on **train subjects only**, then applied to val/test | `_fit_clinical_stats` / `_apply_clinical_norm`; per-fold deep-copy of sequences |
| **MRI cache collapse** | Cache keys include session id (`OAS2_*_MR*`), not stem alone | `_cache_path_for` → `cache_v5` |
| **Cross-subject intensity leak** | Percentile clip / [0,1] map is **per-volume** (brain mask of that scan) | `_load_nifti` |
| **Optimistic protocol** | Contrast only: `leaky_demo.py` deliberately adds CDR/MMSE to show >90% is trivial under label leak | Not used for primary metrics |

**Features used:** `Age, EDUC, SES, eTIV, nWBV, ASF, MF`.  
`eTIV` / `nWBV` / `ASF` are imaging-derived biomarkers (not CDR). They can overlap information with the MRI branch; that is modality redundancy, not label leakage.

---

## 3. How the SNN is used

**Role.** Map one MRI view (currently whole-brain 64³ after crop/resize) to a feature vector \(F^{\mathrm{mri}}\in\mathbb{R}^{64}\), then to an MRI BBA. Clinical features use an MLP, not the SNN.

**Cell.** Parametric LIF (**PLIF**): learnable decay and threshold; binary spikes via **arctan surrogate** for backprop.

**Architecture (`SNN3DEncoder`).**

1. Stem: Conv3D → GroupNorm → PLIF (analog volume → spikes).
2. Stages: spike downsample + SEW-ADD residual blocks; channels `(16, 32, 64)`.
3. Unroll **\(T=8\)** timesteps with membrane carried across steps.
4. AdaptiveAvgPool → learnable softmax weights over time → BN → Linear → \(F^{\mathrm{mri}}\).

**Coding.** Default `SNN_INPUT_CODING='direct'` (same intensity each step; dynamics come from LIF). Optional Bernoulli rate coding.

**Capacity.** One **shared** encoder for all MRI views (`SHARE_MRI_ENCODER=True`) so OASIS-2 (~150 subjects) is not over-parameterised.

---

## 4. How Dempster–Shafer is used

**Frame.** \(\Theta=\{CN,MCI,AD\}\). Mass vector length 8: \(\emptyset\), singletons, pairs (unused here), \(\Theta\).

**BBA from features (`DSEvidenceLayer`).** Subjective-logic / Dirichlet-style head:

\[
e_c=\mathrm{softplus}(\mathrm{head}(F)),\quad
S=\sum_c e_c+|\Theta|,\quad
m(\{c\})=e_c/S,\quad
m(\Theta)=|\Theta|/S
\]

Evidence is scaled/capped (`MRI_EVIDENCE_SCALE`, `MRI_EVIDENCE_CAP`) so weak MRI stays near-vacuous.

**Within-visit fusion.** Sources ordered clinical-first (if enabled), then MRI view BBAs. Pairwise combination via **`ConflictDetectionModule`**:

- Conflict \(K\) from disjoint focal elements.
- Soft gate \(g=\sigma(\gamma(K-\kappa))\).
- Output \(= g\cdot m_{\mathrm{Yager}}+(1-g)\cdot m_{\mathrm{Dempster}}\)  
  (Yager sends conflict mass to \(\Theta\) when sources disagree).

From the fused BBA: **Bel**, **Pl**, **BetP** (pignistic).

**Decision.**  
\(\mathrm{decision}=(1-\texttt{DECISION\_BLEND})\cdot\mathrm{BetP}+\texttt{DECISION\_BLEND}\cdot p_{\mathrm{aux}}\),  
with aux = clinical features only when `AUX_CLINICAL_ONLY=True`.

**Across visits (eval).** `TemporalAccumulator` starts from vacuous \(m(\Theta)=1\) and sequentially Dempster-combines (Yager) per-visit fused BBAs; reported alongside last-visit accuracy. Primary paper numbers remain cross-sectional last-visit test metrics.

---

## Key files

| File | Role |
|------|------|
| `config.py` | Protocol, features, SNN/DS hyperparameters |
| `OasisModule.py` | Load, subject folds, train-only clinical norm, MRI I/O |
| `SNNModule.py` | PLIF / 3D SNN encoder, clinical MLP |
| `DSModule.py` | Evidential BBAs, conflict gate, `DSSNMMF`, losses |
| `main.py` / `TrainerModule.py` | CV, last-visit training, longitudinal eval |
| `leaky_demo.py` | Intentional CDR/MMSE leak contrast (not primary) |
| `baselines.py` | RF / LogReg on the same subject folds |
