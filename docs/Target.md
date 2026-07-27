# Longitudinal Fusion Framework - Main Points

## 1. Overview
- **Longitudinal data**: Multiple measurements from same individual over time
- **Goal**: Track disease progression (Alzheimer's) using multimodal data
- **Key advantage**: Individual trajectory is more informative than single timepoint

## 2. Data Modalities

### MRI (Structural)
- T1-weighted imaging captures grey matter, hippocampus, cortical thickness
- Feature extraction via **3D-SNN encoder**
- Tracks structural atrophy over time

### EEG (Functional)
- Records electrical brain activity
- Feature extraction via **SpikFormer encoder**
- Tracks oscillatory dynamics, coherence, spectral entropy
- Individual signatures require within-patient comparison

### PET (Molecular/Metabolic)
- FDG-PET: glucose hypometabolism
- Amyloid/tau PET: protein accumulation
- Feature extraction via **LIF-Conv3D encoder**
- Tracks metabolic and molecular changes

---

## 3. Spiking Neural Network (SNN) Core Mechanics

### LIF Neuron Dynamics
```
τ_m * du_i/dt = -u_i(t) + R_i * I_i(t)
```
- Integrates spikes, leaks charge, fires at threshold

### Spike Emission
```
S_i(t) = 1 if u_i(t) ≥ u_thr, else 0
u_i⁺ = u_reset when firing
```
- Binary spikes, reset after firing

### STDP Learning (Unsupervised)
- Pre-before-post: strengthen connection (LTP)
- Post-before-pre: weaken connection (LTD)

### Feature Extraction
- **Firing rate**: `r_i = (1/T) * Σ_t S_i(t)`
- **Linear readout**: `z^μ = W_out * r^μ + b`

### Training
- Uses **surrogate gradient** to handle non-differentiable spikes

---

## 4. Dempster-Shafer Theory for Evidence Fusion

### Frame of Discernment
```
Θ = {CN, MCI, AD}  (Cognitive Normal, Mild Cognitive Impairment, Alzheimer's)
```

### Basic Belief Assignment (BBA)
```
m(∅) = 0,  Σ_{A⊆Θ} m(A) = 1
```
- Allocates belief mass to all subsets (including uncertainty)

### Belief vs Plausibility
- **Bel(A)** = Lower bound (mass directly supporting A)
- **Pl(A)** = Upper bound (mass not contradicting A)
- **Uncertainty** = Pl(A) - Bel(A)

---

## 5. Within-Visit Multimodal Fusion

### Per-Modality BPA Generation
```
m^μ{c} = Σ_{k∈c} β_k * exp(-d_k(F^μ))
m^μ(Θ) = 1 - Σ_c m^μ{c}
```
- Prototype-based distance to generate belief masses
- Remaining mass → uncertainty (ignorance)

### Dempster's Combination Rule
```
m(C) = (Σ_{A∩B=C} m₁(A)·m₂(B)) / (1 - K)
```
where `K = Σ_{A∩B=∅} m₁(A)·m₂(B)`

### Yager's Fallback
- When conflict `K > κ` (threshold ~0.45)
- Conflict mass routed to ignorance (Θ) instead of normalization
- Prevents false confidence from contradictory modalities

---

## 6. Longitudinal (Across-Visit) Fusion

### Temporal Accumulation
```
m_v^acc = m_{v-1}^acc ⊕ m_v^fused
```
- Starts from **vacuous prior**: `m₀^acc(Θ) = 1` (complete ignorance)
- **Sequential evidence accumulation** (not averaging)
- Earlier visits permanently shape belief state

### Key Insight
- Consistent evidence → mass concentrates on a single class
- Contradictory evidence → conflict rises → Yager fallback → uncertainty widens

---

## 7. Decision Making & Uncertainty Output

### Final Decision Rule
```
c* = argmax_c [α·Bel_acc(c) + (1-α)·Pl_acc(c)]
```
- α controls caution vs optimism
- **α low**: Conservative (favours belief)
- **α high**: Liberal (favours plausibility)

### Uncertainty Interval
```
Ω = [Bel_acc(c*), Pl_acc(c*)]
```
- **Narrow interval** → Confident diagnosis
- **Wide interval** → Flag for human review
- Clinically interpretable output

---

## 8. Summary Benefits

- ✅ **Biologically plausible** processing via SNNs
- ✅ **Uncertainty quantification** via Dempster-Shafer
- ✅ **Sequential evidence accumulation** (not averaging)
- ✅ **Conflict handling** via Yager fallback
- ✅ **Clinically interpretable** outputs [Bel, Pl]
- ✅ **Energy efficient** SNN computation