# DS-SNN-MF Fusion Conflict Action-Items

*AIM: This document presents a detailed analysis of the fusion failure and action items for the amygdala-ROI fusion conflict --- response to the processed-MRI update*

## 1. Summary of the Update

Skull-stripping and registration to the 2009cAsym template improved MRI performance substantially once the search moved from whole-brain to specific ROIs. The amygdala ROI (0.54 / 0.46) now approaches the clinical branch (0.56 / 0.52) in isolation, a genuine improvement over the conference report's MRI-only numbers. However, fusing the amygdala MRI evidence with the clinical evidence produces a combined result worse than either source alone, with conflict mass K exceeding 0.6 across folds. This is a fusion-layer problem, not an encoder problem, and it is diagnosable and fixable without abandoning the DS approach.

### Per-source results (5-fold CV)

| Source | Accuracy | Macro-F1 |
|---|---:|---:|
| Clinical (reference) | 0.56 | 0.52 |
| Unprocessed MRI - whole brain | 0.41 | 0.30 |
| Processed MRI - whole brain | 0.44 | 0.35 |
| Processed MRI -- ventricles | 0.42 | 0.33 |
| Processed MRI - hippocampus | 0.45 | 0.37 |
| Processed MRI - medial temporal lobe | 0.46 | 0.37 |
| Processed MRI -- amygdala | 0.54 | 0.46 |

## 2. Why This Is Happening --- DST Theory

The observed failure mode is a known and well-documented consequence of how conflict is handled in Dempster--Shafer combination, not evidence that DST is unsuitable here. Three theoretical points explain it directly:

- **Conflict mass K is not itself an error signal, it is the disagreement, in probability mass, between two sources' assignments to disjoint focal elements:**

$$K = \sum_{A \cap B = \varnothing} m_{1}(A)m_{2}(B)$$

When both branches assign most of their mass to singleton classes but pick different singletons, K is large by construction --- this is expected once MRI stops being near-vacuous and starts making confident, sometimes wrong, calls of its own.

- **The Yager component of the gate is deliberately conservative under conflict:** the soft gate

$$g = \sigma\left( \gamma(K - \kappa) \right)$$

pushes the fusion toward Yager's rule as K grows past κ. Yager's rule does not renormalise conflicting mass back onto the surviving singletons (as classical Dempster combination does), it redirects K wholesale onto Θ. This is exactly why the framework was originally built this way: it avoids Zadeh's classical counterexample, where naive Dempster renormalisation under high conflict can assign near-total belief to a class that both sources individually rated as near-impossible. The current result, a fused belief less decisive than either marginal, is the Yager rule doing precisely what it was designed to do. The problem is not that it is misbehaving; it is that at K > 0.6 it is behaving too honestly for a hard-decision accuracy metric.

- **Switching outright to raw Dempster renormalisation is not a safe fix:** it would very likely resolve the decisiveness problem but reintroduce exactly the paradox the Yager gate exists to prevent i.e. a small, spurious majority mass on one side of a high-conflict pair can be renormalised into a false, overconfident singleton. Any fix needs to reduce K itself or redistribute it more surgically, not simply discard the conflict-aware design.

## 3. Recommended Actions

### A. Diagnose before changing the fusion rule

- **Check whether the MRI branch was retrained at its new significance, or only rescaled post-hoc:** the original evidence cap on the MRI branch, and the per-class ignorance-anchoring targets in the loss (`L_ign`: please check the ignorance targets in the DSModule.py lines 346-349, e.g. `"cfg.IGNORANCE_TARGET_CN"`), were tuned to keep MRI near-vacuous. If "increasing its significance" meant raising the evidence cap or scale factor at inference time on a checkpoint trained under the old, near-vacuous constraint, the branch's raw evidence is no longer calibrated and its confidence no longer reflects genuine epistemic strength. That alone can manufacture spurious conflict that has nothing to do with real disagreement between modalities. If this is what happened, the fix is to retrain (or at least fine-tune the evidential head) with the ignorance-anchoring target relaxed to match the new intended operating point, not to rescale a stale checkpoint.

- **Break K down by fold and by class pair, not just the aggregate:** report a table of conflict mass per fold and, within each fold, which class pairs are driving it (e.g. MRI says AD / clinical says MCI vs. MRI says CN / clinical says MCI). Given MCI is the known collapse class in every configuration reported so far, it is very likely that MCI-adjacent disagreement is the dominant source of conflict rather than conflict being spread evenly across all three classes.

- **Check evidence commensurability across branches:** DS combination implicitly assumes both sources' total evidence strength S is on a comparable scale. Compare the distribution of S (or of m(Θ)) produced by the MRI branch versus the clinical branch on held-out folds. If MRI's mass, once its cap is raised, swings from mostly-Θ to mostly-confident far more abruptly than clinical's, the two sources are not contributing evidence on equivalent footing, and conflict will be partly a scale artifact rather than a genuine epistemic disagreement.

### B. Fix conflict handling at the DST level

- **Discount by reliability before combining, rather than only redirecting after combining:** Shafer's classical discounting operator scales a source's mass by a reliability factor r before fusion, m'(A) = (1−r)·m(A) for A ≠ Θ, with the discounted mass added to m(Θ). Estimate r per branch from out-of-fold macro-F1 (or from the calibration/ECE work already planned for the paper) and apply it to the MRI branch before combination. This front-loads caution proportional to demonstrated reliability, which tends to reduce spurious high-conflict cases up front instead of relying entirely on the post-hoc Yager redirection to absorb them.

- **Consider Proportional Conflict Redistribution (PCR5/PCR6) as a third rule in the gate, not just a Dempster/Yager mixture:** for a conflicting pair of focal elements A, B with A ∩ B = ∅, PCR5 redistributes only the conflicting mass generated by that specific pair back onto A and B, in proportion to their own masses, instead of sending all conflicting mass to Θ:

$$\Delta m(A) = \frac{{m(A)}^{2}}{m(A) + m(B)} \cdot m(B)$$

This sits theoretically between Dempster (fully renormalises, risks Zadeh's paradox) and Yager (fully redirects to ignorance, kills decisiveness). Extending the existing sigmoid gate to a three-way blend across Dempster / PCR5 / Yager or simply replacing the Yager branch with PCR5 is a natural next experiment and is well precedented in the DSmT literature (Smarandache & Dezert).

- **Exploit the ordinal structure of the frame instead of treating it as flat:** CN, MCI, and AD are not an unordered nominal set but they represent increasing severity along a single axis. A flat Θ = {CN, MCI, AD} treats a CN/AD disagreement (clinically implausible, two full stages apart) identically to a CN/MCI or MCI/AD disagreement (clinically routine, adjacent stages). Restricting the focal-element set to contiguous subsets only: {CN}, {MCI}, {AD}, {CN,MCI}, {MCI,AD}, and Θ, dropping the non-contiguous {CN,AD} gives conflicting evidence a clinically meaningful "half-way" resting place (e.g. {MCI,AD}) instead of forcing all disagreement straight to full ignorance. This is very likely to reduce the practical impact of high conflict mass without weakening the framework's honesty.

### C. Architectural next steps

- **Fuse ROI-specific MRI branches with each other before fusing with clinical:** the plan to train separate SNNs per ROI (hippocampus, amygdala, medial temporal lobe) and combine them with DST is theoretically the right structure, but combine them in two stages, not one flat combination across all sources. First combine the MRI-ROI branches with each other (they are all derived from the same modality and are far more likely to be positively correlated / low-conflict), producing one consolidated MRI BBA with genuinely reduced uncertainty. Only then combine that consolidated MRI BBA with the clinical BBA. This hierarchical pattern is standard in multi-sensor DS fusion and should produce a less erratic single MRI-side input to the modality-level fusion step than combining many raw ROI sources directly against clinical all at once.

- **Make the decision layer conflict-aware, not just the evidence layer:** the pignistic blend currently uses a fixed α between BetP and the clinical-only auxiliary head. Consider making α a function of K e.g. increasing α (trusting the clinical auxiliary head more) specifically on high-conflict folds/samples. This gives the framework a principled fallback for hard decisions without discarding the evidential fused output as the primary reported quantity.

- **Report γ and κ explicitly, and treat them as tunable, not fixed:** these were already flagged as missing from the manuscript's reported hyperparameters. Given the gate's behaviour is central to this failure mode, a small sweep over κ (the conflict threshold at which Yager begins to dominate) is a cheap, high-value experiment before any architectural change --- the current failure may partly be that κ was tuned for the old, near-vacuous MRI branch and is now triggering Yager mode far too early for the new, more assertive MRI branch.

### D. Before concluding that this is a dead end

- **Re-run the amygdala-only vs. fused comparison with the discounting fix alone (B, first bullet), holding the rest of the pipeline fixed:** this isolates whether the failure is primarily a calibration/commensurability problem (fixed by discounting) or a structural problem with the combination rule itself (requiring PCR5 or the ordinal frame).

- **Report variance, not just the mean conflict mass:** confirm whether K > 0.6 holds uniformly across all 5 folds or is driven by one or two folds --- this changes whether the fix should be global (change the rule) or targeted (handle a specific failure pattern, e.g. one fold with a distribution shift).

## 4. Quick Reference --- DST Fusion Rules Compared

| Rule | Handles conflict by... | Risk |
|---|---|---|
| Dempster (classical) | Renormalising over non-conflicting focal elements | Can produce Zadeh's paradox: false confidence under high conflict |
| Yager | Redirecting all conflicting mass K to Θ (ignorance) | Over-conservative --- kills decisiveness when K is large, as currently observed |
| PCR5 / PCR6 | Redistributing conflicting mass proportionally back onto the specific elements that generated it | More complex to implement; still an active research area (DSmT) |
| Discounting (pre-fusion) | Scaling down a less-reliable source's mass before combination | Requires a good reliability estimate (e.g. from validation macro-F1 or ECE) |

**Bottom line:** the amygdala result is a real win worth keeping. The fusion failure is a well-characterised DST phenomenon, not a sign the evidential approach is wrong for this problem, and so the next experiments should target evidence commensurability and conflict redistribution before any larger architectural rewrite is done.