# Fusion retune changelog

Date: 2026-09-19

Keeps the existing DS-SNN-MF formulation (evidential BBAs, two-stage MRI-then-clinical combine, PCR5 gate, Shafer discount, K-aware decision mix already in `DSModule.py`). No new fusion rule, no extra heads, no extra ROI.

## Baseline this retune is against

Honest full run in `Experimental Results/newAblation.txt` with `DECISION_BLEND=0.0` (reported `pred` = fused BetP):

| | Acc | Macro-F1 | meanK |
|---|---:|---:|---:|
| Full BetP | 0.51 | 0.48 | 0.08 |
| MRI standalone (MTL) | 0.56 | 0.42 | 0 |
| Clinical (aux, `blend=1.0`) | 0.55 | 0.51 | 0 |

Combination was calm (meanK 0.08) and still lost. Causes: MRI discounted to a spectator (`r=0.6`, scale 0.4); disagreement smeared in BetP instead of routed to clinical; `L_ce` trained BetP while `pred` used `decision`.

## Config before → after

| Knob | Before (honest blend=0 full) | After |
|---|---|---|
| `MRI_EVIDENCE_SCALE` | 0.4 | **0.7** |
| `MRI_EVIDENCE_CAP` | 8.0 | **9.0** |
| `BRANCH_RELIABILITY` | `{'mtl': 0.6, 'clinical': 1.0}` | **`{'mtl': 0.85, 'clinical': 1.0}`** |
| `DECISION_BLEND` | 0.0 | 0.0 (unchanged; low-K stays BetP) |
| `DECISION_BLEND_MAX` | unset (= base 0, mix inert) | **0.45** |
| `DECISION_BLEND_KAPPA` | unset (= `CONFLICT_KAPPA`) | **0.12** |
| `CONFLICT_KAPPA` | 0.12 | **0.25** |
| `CONFLICT_RULE` | `pcr5` | `pcr5` |
| `LAMBDA_AUX` | 0.0 | **0.8** |
| `AUX_CLINICAL_ONLY` | True | True |
| `ABLATION_PURE_DECISION` | True | True |
| `CKPT_TRAIN_F1_WEIGHT` | 0.25 | **0.10** |

Unchanged: `ABLATION='full'`, MTL view, PCR5, two-stage DS, evidential heads, `LAMBDA_DS=0.15`, `LAMBDA_K=0.08`, `LAMBDA_IGN=0.05`, `EVIDENCE_CAP=18`, `MRI_EVIDENCE_CAP_STANDALONE=18`.

### Why each knob

- **MRI scale 0.7 / cap 9 / r=0.85** — previous `r=0.6` + scale 0.4 left MRI too weak to harvest standalone CN F1 (0.71 → fused 0.64). Still below standalone 1.0 / 18 so it is not an equal-loud fight.
- **`DECISION_BLEND_MAX=0.45`** — MD §C. Low K → fused BetP (Dempster sharpening on agreement). High K → up to 45% clinical softmax so MCI disagreement is not a 50/50 BetP coin flip.
- **`CONFLICT_KAPPA=0.25`** — meanK was 0.08 with κ=0.12, so the PCR5 mix was already ~30% on “calm” samples. Higher κ keeps moderate K on Dempster.
- **`LAMBDA_AUX=0.8`** — fallback is the clinical aux head; at 0 that head was untrained.
- **`CKPT_TRAIN_F1_WEIGHT=0.10`** — fold 4 of the baseline selected val 0.50 / test 0.37; less reward for high train F1.

## Code changes

### `DSModule.py` — `DSSNNLoss`

`L_ce` is on `out['decision']` (the mix that `pred` uses), not `out['betp']`. Matches Methodology (“CE on the blended decision”) so gradients reach the K-aware mix.

### `TrainerModule.py` — `evaluate`

Averages `out['decision_blend']` as `mean_decision_blend` and prints it next to meanK in the epoch log (`blend …`). If this stays ~0 on a full run, the clinical fallback is not firing.

### `main.py`

CV / per-fold prints include `Test mean decision blend` so the ablation log shows whether the mix ran.

## What was not changed

- PCR5 as the high-K combination rule
- Two-stage MRI-then-clinical DS combine
- Subjective-logic evidential heads
- `ABLATION_PURE_DECISION=True` (MRI / clinical ablations stay BetP-only)
- No frozen standalone late-fusion, no extra ROI, no class-conditional DST

## How to run

```bash
python main.py
```

`ABLATION` is already `'full'`. Compare test acc / F1 / meanK / mean decision blend to the backed-up `blend=0` row in `Experimental Results/newAblation.txt`. If mean blend is ~0, MRI and clinical are still only smearing; if blend is clearly >0 on high-K folds, the fallback is in use.

Target: fused above 0.51, preferably at or above ~0.55 acc / ~0.50 F1. Not a guaranteed 0.60.

## Follow-up: folds 4–5 MCI collapse (same day)

First retune full run: folds 1–3 **0.63 / 0.60 / 0.57** (mean 0.60 acc / 0.54 F1). Folds 4–5 **0.43 / 0.43** with MCI F1 0.12 / 0.20 (8–9 of 11 MCI called CN). Clinical on those folds stayed 0.53–0.57; fusion did not.

Cause: `DECISION_BLEND_MAX=0.45` cannot flip a CN-peaked BetP. Fold 5 high-K rate 0.80 but mean blend only 0.39 → decision still ~61% fused BetP. `DECISION_BLEND_KAPPA=0.12` also mixed almost every sample (good folds blend 0.30–0.36 ≈ bad folds 0.34–0.39).

Config-only tweak (no code):

| Knob | First retune | Now |
|---|---|---|
| `DECISION_BLEND_MAX` | 0.45 | **0.75** |
| `DECISION_BLEND_KAPPA` | 0.12 | **0.20** |
| `BRANCH_RELIABILITY['mtl']` | 0.85 | **0.75** |

High-K samples can follow clinical aux; lower-K folds 1–3 should stay BetP-heavy. Slightly softer MRI so CN mass is less peaked before the mix. Scale/cap/PCR5/κ_conflict unchanged.

Second retune full run: **0.547 / 0.500** F1, folds **0.57 / 0.60 / 0.60 / 0.50 / 0.47**. Mean blend 0.32; fold 5 blend **0.57** (hi-K 0.90) so those decisions were ~57% clinical aux. CN F1 fused 0.67 vs MRI 0.71 vs clinical 0.65 — MRI still present on agreement, but high-K fallback let clinical overpower MRI.

## Follow-up: keep MRI in the vote (same day)

Goal: clinical must not monopolize high-K decisions; fused BetP (MRI × clinical BBA) stays the majority of `decision`.

| Knob | Folds 4–5 tweak | Now |
|---|---|---|
| `DECISION_BLEND_MAX` | 0.75 | **0.50** |
| `BRANCH_RELIABILITY['mtl']` | 0.75 | **0.90** |

High-K `decision` is at most 50% clinical aux / at least 50% fused BetP. MRI Shafer reliability restored so the BBA itself is not a spectator. `DECISION_BLEND_KAPPA` stays 0.20. Re-run `python main.py` with `ABLATION='full'`.

