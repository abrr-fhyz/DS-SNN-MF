# DS-SNN: A Dempster-Shafer Evidential Spiking Neural Network Framework for Dementia Classification

## Overview

This paper presents **DS-SNN**, a framework for Alzheimer's disease (AD) detection that combines:
- A **3D Parametric-LIF spiking neural encoder** for MRI data
- An **MLP** for tabular clinical features
- A **Dempster-Shafer (DS) evidence theory** fusion layer
- **Conflict-aware Yager gating** for combining evidence

**Key innovation:** The framework is designed as an "honest baseline" that explicitly represents uncertainty, redirects conflicting evidence to the full frame Θ, and enforces strict leakage-free protocols.

---

## Literature Review: The Three Core Problems

### 1. The Performance Inflation/Literature Leakage Crisis

**Young et al. (2025)** - Scoping review of 44 peer-reviewed AD deep learning studies:
- Concluded that "data leakage and associated methodological flaws create a pervasive illusion of near-perfect performance"
- **Only 18%** of studies had any form of confounder control
- **13.6%** of studies fell into high-risk category with reported accuracies of 95-99%
- Nearly **80%** of surveyed studies provided no interpretability methods
- Estimated **credible performance range after leakage control: 66-90%**, with clinical accuracy typically 70-85%
- Identified "the interpretability-validation chasm" - disconnect between technical capability and clinical utility

**Yagis et al. (2021)** - Empirical quantification of leakage effects:
- **Slice-level cross-validation inflated OASIS accuracy by ~30%** relative to subject-level splits
- **Randomly labelled OASIS** produced ~96% accuracy under slice-level splitting but only 50% under subject-level splitting
- Confirmed leakage by inspecting source code of published papers
- Concluded CNN models reporting >93% accuracy are likely inflated by intra-subject correlation

**Ghiasi et al. (2026)** - Established leakage-free benchmark on OASIS-2:
- Addressed three leakage vectors:
  - Target leakage (excluding CDR from inputs)
  - Subject leakage (group-aware splitting)
  - Temporal leakage (using engineered rate-based features)
- Models achieved approximately **57.9-69.7% accuracy** under defined protocol
- Several studies reported 82.5-99.1% accuracy when CDR was included

### 2. Under-Explored Neuromorphic Modelling (SNNs)

**Turkson et al. (2021)** - First use of SNNs for AD classification:
- Primarily motivated by energy-efficiency advantages rather than neuromorphic benefits

**Wu et al. (2025)** - Introduced FasterSNN:
- Lightweight LIF-based SNN with Weighted Attention Module
- Framed SNNs as "brain-inspired design well suited to model sparse and event-driven patterns of neural degeneration"
- **Limitation:** Lacks explicit confounder control and clear handling of longitudinal subjects → possible temporal leakage risk

**Doborjeh (2021)** - Individualized neural models per patient:
- Mapped selected MRI features into 3D SNN with hemisphere-wise analysis
- Sydney Memory and Ageing Study cohort (175 adults)
- Reported 91-96% accuracy but **suffers from feature weighting applied to entire dataset prior to cross-validation**

### 3. Dempster-Shafer Theory for Medical Decision-Making

**Bahodoran and Ghasemi (2025)** - Proposed DS fusion with multimodal MRI:
- Reported 91% accuracy on ADNI
- **Limitation:** No leakage prevention discussed - PCA performed on entire dataset before cross-validation (severe feature-leakage failure)

**Sensoy et al. (2018)** - Subjective-logic formulation for BBA construction (foundational work)

---

## Research Gap Positioning

DS-SNN is positioned at the intersection of three gaps:

1. **Overconfident outputs** - Dominant CNN/transformers cannot separate confident from ignorant predictions
2. **Under-explored neuromorphic modelling** - SNNs are biologically plausible but underrepresented in AD literature
3. **Systemic leakage** - No prior SNN-based or DS-based AD work has adopted strict leakage-free protocols

---

## Methodology Highlights

### Problem Setup
- **3-class classification:** Θ = {CN (Cognitively Normal), MCI (Mild Cognitive Impairment), AD (Alzheimer's Disease)}
- **Labels derived from CDR:** CDR=0 → CN, CDR=0.5 → MCI, CDR=1 → AD
- **Subject-level 5-fold stratified cross-validation** with fixed seed
- **Training uses each subject's last visit only**

### Seven Leakage Controls Implemented:
| Control Type | Implementation |
|--------------|----------------|
| Subject-level splits | All visits of a subject in one partition |
| Slice-level prevention | Retain only one image per subject |
| Target leakage | Exclude CDR/MMSE from inputs (primary protocol) |
| Normalisation leakage | Train-only clinical statistics |
| Temporal leakage | Last-visit training |
| Feature selection | No dataset-wide PCA |
| Confounder control | Stratified splits |

### Input Features (Clinical)
- Age, EDUC (Education), SES (Socioeconomic Status), eTIV (Estimated Total Intracranial Volume), nWBV (Normalized Whole Brain Volume), ASF (Atlas Scaling Factor), MF (Manual/Field)

### Spiking MRI Encoder
- **Neuron model:** Parametric Leaky-Integrate-Fire (PLIF) with learnable membrane decay and threshold
- **Surrogate gradient:** Arctan for backpropagation
- **Architecture:**
  - Stem: Conv3D-GroupNorm-PLIF block
  - Stages: Spike downsampling + SEW-ADD residual blocks (channels: 16, 32, 64)
  - Unrolled for T = 8 timesteps
  - Head: Adaptive average pooling + softmax-weighted temporal pooling → F^mri ∈ ℝ⁶⁴

### Evidential Heads and BBA Construction
- **Evidence for class c:** e_c = softplus(head(F))
- **Total strength:** S = Σ e_c + |Θ|
- **Mass assigned to singleton:** m({c}) = e_c / S
- **Mass assigned to full frame:** m(Θ) = |Θ| / S

### Conflict-Aware Fusion
- **Conflict mass K** computed from disjoint focal elements
- **Soft gate:** g = σ(γ(K - κ))
- **Output BBA:** m_fused = g·m_Yager + (1-g)·m_Dempster
  - Low conflict → Dempster combination
  - High conflict → Yager rule (reallocates conflicting mass to Θ)

### Decision Rule
- Blends pignistic transform of fused BBA with auxiliary clinical-only softmax head:
  - ŷ = arg max[(1-α)BetP + α·p_aux]

---

## Results

### Classical Baselines (Subject-level 5-fold CV)

| Model | Protocol | Accuracy | Macro-F1 | Balanced Acc. |
|-------|----------|----------|----------|---------------|
| Majority | final-visit | 0.487 | — | — |
| LR | final-visit | 0.493 ± 0.095 | 0.468 ± 0.084 | 0.509 ± 0.086 |
| RF | final-visit | **0.560 ± 0.049** | 0.519 ± 0.059 | 0.544 ± 0.073 |
| LR | all-visits | 0.522 ± 0.072 | 0.472 ± 0.071 | 0.534 ± 0.042 |
| RF | all-visits | 0.535 ± 0.038 | 0.438 ± 0.022 | 0.466 ± 0.017 |

**Key observation:** Random Forest on subject-final-visit is strongest classical reference at 0.560 accuracy / 0.519 macro-F1.

---

### DS-SNN and Modality Ablations

Table below reports the DS-SNN framework and its two ablations under subject-level 5-fold CV. Two independent runs of the full framework are reported to give a sense of run-to-run variance under our small-cohort protocol.

| Configuration | Accuracy | Macro-F1 | Balanced Acc. | Long. (Dempster) | CN F1 | MCI F1 | AD F1 |
|---------------|----------|----------|---------------|------------------|-------|--------|-------|
| MRI only | 0.420 ± 0.051 | 0.304 ± 0.067 | 0.341 ± 0.060 | 0.373 ± 0.109 | 0.576 | 0.230 | 0.104 |
| Clinical only | **0.567 ± 0.075** | **0.539 ± 0.088** | 0.546 ± 0.101 | 0.540 ± 0.044 | 0.666 | 0.454 | 0.497 |
| DS-SNN (Run 1) | 0.547 ± 0.056 | 0.525 ± 0.069 | 0.517 ± 0.062 | 0.520 ± 0.051 | 0.650 | 0.384 | 0.540 |
| DS-SNN (Run 2) | **0.560 ± 0.092** | 0.519 ± 0.097 | 0.525 ± 0.076 | 0.553 ± 0.069 | 0.647 | 0.437 | 0.477 |

**Key observations:**
1. **MRI-only underperforms:** With whole-brain, unregistered \(64^3\) volumes and no atlas- or ROI-guided preprocessing, the SNN branch alone recovers CN weakly (F1 = 0.576) and is essentially non-informative on AD (F1 = 0.104). This is likely a consequence of feeding raw volumes to a shared 3D PLIF encoder on 150 subjects.
2. **Clinical-only is a strong reference:** The clinical MLP + evidential head alone reaches 0.567 accuracy / 0.539 macro-F1, exceeding the RF baseline in accuracy and macro-F1.
3. **Full DS-SNN is competitive with clinical-only:** Fusing the weak MRI BBA does not degrade accuracy (Run 2: 0.560, Run 1: 0.547) but remains considerably behind the lower bound of the credible range identified by Young et al. (2025). That being said, the results are closer to the ~58–70% range reported by Ghiasi et al. (2026) on OASIS-2.
4. **Strict protocol:** The overall performance dip can likely be attributed to the exclusion of MMSE as a feature due to the stricter protocol setting, as demonstrated in the following section.

---

### Effect of Including MMSE

To quantify the contribution of MMSE, we repeated the full DS-SNN experiment with MMSE restored as a clinical input. This setting operates in a state closer to the one described in literature.

| Configuration | Accuracy | Macro-F1 | Balanced Acc. | Long. (Dempster) | CN F1 | MCI F1 | AD F1 |
|---------------|----------|----------|---------------|------------------|-------|--------|-------|
| DS-SNN | **0.633 ± 0.082** | **0.595 ± 0.091** | 0.595 ± 0.073 | 0.447 ± 0.080 | 0.758 | 0.447 | 0.581 |

Restoring MMSE improves performance across all aggregate metrics, increasing accuracy and macro-F1 to approximately 63.3% and 0.60 respectively. These results place the proposed framework substantially closer to the ~66% performance reported by Young et al. (2025) and within the band identified by Ghiasi et al. (2026), although marginally below their reported peak of approximately 69.7% accuracy.

---

### Leakage-Contrast Experiment

| Configuration | Accuracy | Macro-F1 | Balanced Acc. | CN F1 | MCI F1 | AD F1 |
|---------------|----------|----------|---------------|-------|--------|-------|
| DS-SNN + CDR/MMSE (partial) | 0.847 ± 0.084 | 0.834 ± 0.069 | 0.855 ± 0.075 | 0.908 | 0.748 | 0.847 |
| DS-SNN + CDR/MMSE (full leak) | **0.980 ± 0.030** | **0.979 ± 0.028** | 0.981 ± 0.028 | 0.987 | 0.970 | 0.982 |

**Key observation:** Same architecture reaches **98% accuracy** when CDR admitted as input, quantitatively reproducing the inflated performance pattern documented in literature by relaxing exactly one protocol constraint.

**This confirms the framework has capacity, and the headline 56% is protocol-bound, not capacity-bound.**

---

### Evidential and Fusion Diagnostics

| Metric | Run 1 | Run 2 | With MMSE |
|--------|-------|-------|-----------|
| Mean conflict mass K | 0.099 ± 0.016 | 0.044 ± 0.015 | 0.086 ± 0.051 |
| High-conflict rate | 0.000 | 0.000 | 0.000 |
| Soft-gate g | 0.400 | 0.186 | 0.356 |

**Key observations:**
- Conflict mass remains low across all configurations
- MRI and clinical BBAs rarely produce strongly contradictory evidence
- Soft-gate is active but not saturated - evidence partially redirected to Θ
- **Longitudinal accumulation does not improve** over last-visit prediction:
  - Run 1: 0.520 vs 0.547 last-visit
  - Run 2: 0.553 vs 0.560 last-visit
  - With MMSE: 0.447 vs 0.633 last-visit

---

## Stated Goals and Fulfillment

### 1. Evidential Outputs
- **Goal:** "Both branches emit Dirichlet-style BBAs on Θ = {CN, MCI, AD}, preserving uncertainty as an explicit quantity"
- **Achieved:** BBA construction with m(Θ) explicitly represents ignorance; weak modalities remain vacuous rather than confidently wrong

### 2. Conflict-Aware Fusion
- **Goal:** "A spiking neural encoder coupled with a DS evidential layer where MRI and clinical BBAs are combined through a learned gate that interpolates between Dempster's and Yager's rules"
- **Achieved:** Soft-gate g = σ(γ(K-κ)); redirects conflicting evidence to Θ; mean conflict low; gate active but not saturated

### 3. Leakage-Free Evaluation
- **Goal:** "Subject-level 5-fold cross-validation, last-visit training, train-only clinical statistics, and deliberate exclusion of CDR/MMSE"
- **Achieved:** Seven leakage controls operationalized; performance drops from 98% (leaky) to 56% (strict)

### 4. Honest Baseline
- **Goal:** "Establish an honest evidential-neuromorphic baseline in the credible-accuracy regime"
- **Achieved:** Results align with credible range (58-70%) identified in literature; framework explicitly explains inflated numbers via leakage-contrast experiment

---

## Limitations Identified

1. **Statistical underpowering:** n ≈ 150 subjects split five ways (~30 test per fold); wide standard deviations (macro-F1 σ up to ~0.10)

2. **Temporal accumulator ineffective:** Dempster accumulation does not improve on last-visit accuracy

3. **No MRI preprocessing:** Raw volumes hamper performance; skull/neck tissue dominates input; slow convergence

4. **Performance trails clinical-accuracy band:** 56-63% vs 70-85% identified by Young et al.; larger cohorts needed

5. **MCI lowest F1:** Classic artefact of MCI being diagnostic middle class; DS-SNN doesn't substantially improve MCI classification

---

## Future Work Directions

1. **Larger datasets:** Extend to ADNI, AIBL, OASIS-3 for generalisability and external validation

2. **Multimodal fusion:** Extend to DS-SNN-MF incorporating EEG, PET modalities

3. **MRI preprocessing:** Upgrade with FSL library for registration, skull-stripping, ROI cropping

---

## Conclusion

DS-SNN represents the **first honest evidential-neuromorphic baseline** for Alzheimer's detection that:

- Explicitly represents ignorance
- Redirects conflicting evidence to Θ
- Enforces strict leakage-free protocol

Performance: 0.547-0.560 accuracy (without MMSE) / 0.633 (with MMSE)

Key contribution: Demonstrates that the same architecture can achieve 98% accuracy with leakage, quantitatively explaining the inflated numbers common in the field.

**Best understood as a foundation for multimodal, larger-scale extensions rather than a leaderboard entry.**