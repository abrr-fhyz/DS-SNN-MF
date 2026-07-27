# DS-SNN Longitudinal Fusion (OASIS-2)

Multi-view structural MRI and clinical evidence are fused with **Dempster–Shafer (DS) theory** and encoded by **spiking neural networks (SNNs)** to classify Alzheimer's disease progression on **OASIS-2**: three classes (**CN**, **MCI**, **AD**), with optional **longitudinal** evidence accumulation across repeated visits.

The original framework (see `Explainer.md`) was designed for multimodal longitudinal data (MRI + EEG + PET on OASIS-3 / ADNI). This repository is a **streamlined OASIS-2 implementation**: structural MRI (multi-view SNN) + tabular clinical features, with DS fusion at both the within-visit and across-visit levels.

---

## What we are trying to do

1. **Within-visit fusion (DS)**  
   Each visit produces several evidence sources:
   - **MRI views** (whole brain, hippocampus, ventricles) → shared 3D SNN encoder → evidential BBAs  
   - **Clinical tabular features** (age, education, imaging-derived biomarkers, sex) → MLP → BBA  
   Sources are combined with a **conflict-aware** Dempster / Yager gate. Decisions use belief, plausibility, and an auxiliary head (see `Explainer.md` §5–7).

2. **Across-visit fusion (longitudinal DS)**  
   For each subject, per-visit fused BBAs are **sequentially accumulated** (not averaged), with **recency weighting** so later visits dominate. Training and evaluation can target the **final-visit diagnosis** — matching the clinical question “what is this subject’s status at their latest scan?”

3. **Interpretability**  
   DS outputs **belief**, **plausibility**, **conflict (K)**, and **ignorance (Θ)** mass — useful for uncertainty-aware reporting, not only argmax accuracy.

4. **Honest evaluation**  
   All primary results use **subject-level stratified 5-fold cross-validation**. We explicitly avoid protocols that inflate scores (label leakage, visit-level splits, cherry-picked validation peaks).

---

## Accuracy goals (and what “good” means here)

| Target | Meaning |
|--------|---------|
| **Credible primary metric** | Subject-level **5-fold CV**: test accuracy and **macro-F1** (mean ± std) |
| **Competitive & honest** | ~**0.55–0.60 accuracy**, ~**0.50–0.55 macro-F1** on 3-class CN/MCI/AD |
| **Beat strong tabular baseline** | Outperform sklearn **Random Forest** on the same folds (~**0.52 macro-F1** without MRI) |
| **Stretch goal** | Stable **0.60–0.65 macro-F1** mean CV — ambitious on OASIS-2 without registration / larger cohorts |
| **Not a primary claim** | Single-split validation **>80%** or leaky **>90%** — useful only as a **methodological contrast** (see `leaky_demo.py`) |

**Recent honest full-model CV (cross-sectional training recipe):**  
~**0.56 ± 0.07** accuracy, ~**0.50 ± 0.07** macro-F1 — on par with the tabular RF ceiling, with non-zero MCI F1 (~0.39 mean).

The **longitudinal training** recipe (`TRAIN_MODE='longitudinal'`) is the path aligned with `Explainer.md`; re-run CV after enabling it to measure whether temporal DS accumulation improves over cross-sectional training.

---

## Why our numbers are lower than many published papers

This is usually **not** because the model “cannot learn,” but because **the task and evaluation protocol are harder and more honest**:

1. **Three classes including MCI**  
   Many OASIS papers report **binary CN vs AD** and drop or merge MCI. We keep **CN / MCI / AD** from per-visit **CDR** (0 → CN, 0.5 → MCI, ≥1 → AD). MCI is the hardest class and drags macro-F1.

2. **Subject-level splits**  
   Splits are by **subject**, not by visit. Putting different scans of the same person in train and test can inflate accuracy by **10–20+ points**. We do not do that for primary results.

3. **No label leakage in features**  
   **CDR and MMSE are excluded** from inputs (they define the label). Including them — as in `leaky_demo.py` — can push validation toward **>90%**; that is **not** generalisation.

4. **Small cohort**  
   OASIS-2 has on the order of **~150 subjects** and **~370 visits**. Deep 3D models (even compact SNNs) are data-hungry; variance across 5-fold test folds is large (e.g. one fold ~0.39 F1, others ~0.55–0.57).

5. **Unimodal OASIS-2 vs multimodal papers**  
   We do not have OASIS-3 / ADNI / PET / EEG here. Clinical signal is strong (nWBV, eTIV, etc.); raw whole-brain MRI on **unaligned** volumes is a weak incremental signal until registration or atlas ROIs are added.

6. **Small validation sets + early stopping**  
   Each fold’s validation set is ~**18 subjects**. A lucky epoch can show **~65–80% val accuracy** that **does not transfer** to held-out test subjects (~**0.55**). That is why we report **CV mean ± std**, not a single val peak.

**Takeaway for papers:** report **subject-level CV** and a **simple baseline (RF)** on the same splits. Optionally cite `leaky_demo.py` to show that high val numbers are easy under leaky protocols — which motivates the conservative pipeline.

---

## Repository layout

| File | Role |
|------|------|
| `main.py` | Entry point: 5-fold CV or single split; longitudinal or cross-sectional training |
| `config.py` | Hyperparameters, views, DS/training mode |
| `OasisModule.py` | OASIS-2 loading, CDR labels, train-only clinical normalisation, CV folds |
| `SNNModule.py` | 3D SNN encoder, clinical MLP |
| `DSModule.py` | DS utilities, evidential heads, fusion model, losses |
| `TrainerModule.py` | Training loop, longitudinal accumulation, evaluation |
| `baselines.py` | Sklearn LogReg / RF on same clinical features (sanity ceiling) |
| `leaky_demo.py` | **Intentional** leaky protocol for inflated val (paper caveat) |
| `Explainer.md` | Original longitudinal fusion framework (multimodal design doc) |

Checkpoints and per-fold metrics: `cv_runs/`.

---

## Data layout

Expected under `data/oasis2/`:

```
data/oasis2/
  oasis_longitudinal_demographics.xlsx
  OAS2_RAW_PART1/OAS2_0001_MR1/RAW/...
  OAS2_RAW_PART2/...
```

**Default now:** whole-brain RAW + brain-mask zeroing (no FSL).  
**Later (optional, slow):** `python mri_preprocess.py` then set `MRI_USE_PROCESSED=True` and re-enable ROI views / GM channels in `config.py`.
---

## How to run

**Environment:** Python 3.10+, PyTorch, nibabel, scikit-learn, pandas, openpyxl (for `.xlsx`).

### Primary evaluation (honest 5-fold CV)

```bash
python main.py
```

Key settings in `config.py`:

- `USE_CV = True`, `N_FOLDS = 5`
- `TRAIN_MODE = 'longitudinal'` — temporal DS training (Explainer §6)  
- `TRAIN_MODE = 'cross_sectional'` — per-visit flattened training (legacy)
- `ABLATION = 'full' | 'clinical' | 'mri'`

### Tabular baseline (same folds, no neural net)

```bash
python baselines.py
```

### Methodological contrast (inflated validation — do not report as performance)

```bash
python leaky_demo.py
```

Prints peak validation accuracy under CDR/MMSE feature leakage + visit-level split, plus a **formal caveat** paragraph for papers. Output: `leaky_demo_metrics.txt`.

---

## Configuration highlights

| Setting | Typical value | Notes |
|---------|---------------|--------|
| `CLINICAL_COLS` | Age, EDUC, SES, eTIV, nWBV, ASF, MF | **No CDR/MMSE** |
| `MRI_VIEWS` | whole, hippocampus, ventricles | Heuristic crops on RAW volumes |
| `TEMPORAL_FORGET` | 0.30 | Recency before each visit combine |
| `DECISION_ALPHA` | 0.60 | α·Bel + (1−α)·Pl |
| `CONFLICT_KAPPA` | 0.12 | Soft Dempster↔Yager gate |
| `CHECKPOINT_WARMUP` | 10 | Avoid saving lucky epoch-1 val on tiny val sets |

---

## Suggested reporting (WIP / conference)

**Report:**

- Subject-level **5-fold CV**: accuracy, macro-F1, balanced accuracy, per-class F1 (mean ± std)
- **RF baseline** on identical splits
- DS diagnostics: mean **K**, soft conflict gate, mean uncertainty (Pl − Bel)
- Optional: ablation (`clinical`-only, `mri`-only, no longitudinal accumulation)

**Do not report as main results:**

- Single-split validation peaks (~80%+) without subject-level CV test numbers
- Metrics from `leaky_demo.py` (except in a “leakage sensitivity” or methods caveat subsection)

**Framing:** novelty is **evidential multi-view MRI + clinical fusion with SNN encoders and longitudinal DS accumulation**, with **uncertainty-aware outputs** — evaluated under a **strict OASIS-2 3-class protocol** that many high-accuracy papers do not use.

---

## Limitations and next steps

- **Registered multi-view MRI** (`mri_preprocess.py` + ROI/GM channels) when FSL/WSL time allows
- **Pretrained 3D encoder** or CNN→DS distillation (SNN-from-scratch is sample-limited)
- **Recency-weighted longitudinal loss** tuning (`TEMPORAL_FORGET`, `LAMBDA_VISIT`)
- Access to **OASIS-3 / ADNI** would restore true multimodal fusion (EEG/PET branches in the original design)

---

## References in-repo

- `Explainer.md` — full DS + SNN + longitudinal fusion specification  
- OASIS-2: Marcus et al., longitudinal structural MRI with clinical demographics  

For questions about a specific run, inspect `logs.txt` and `cv_runs/cv_summary.json`.
