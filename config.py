from pathlib import Path
import torch


class Config:
    """Configuration for the OASIS-2 DS-SNN model.

    Scope: OASIS-2 only. Dempster-Shafer fusion of multi-view MRI (SNN) and
    clinical tabular features. Tuned for a tiny subject-level dataset (~150).

    Primary protocol: subject-level 5-fold CV, last-visit-only samples
    (matches RF subject-final-visit), clinical LR hot / MRI LR cold.
    """

    # ── Task ──────────────────────────────────────────────────────────────
    CLASSES     = ['CN', 'MCI', 'AD']
    NUM_CLASSES = 3
    SEED        = 42
    NUM_WORKERS = 1

    # ── Data (OASIS-2) ────────────────────────────────────────────────────
    DATA_ROOT  = Path("data")
    OASIS2_DIR = DATA_ROOT / "oasis2"
    OASIS2_CSV = OASIS2_DIR / "oasis_longitudinal_demographics.xlsx"
    MRI_SHAPE  = (96, 96, 96)
    MAX_VISITS = 5

    # CDR/MMSE excluded (label leakage). M/F is a legitimate demographic.
    # nWBV/eTIV/ASF are imaging-derived biomarkers, not the diagnosis.
    CLINICAL_COLS = ['Age', 'EDUC', 'SES', 'eTIV', 'nWBV', 'ASF', 'MF', 'MMSE']

    TRAIN_FRAC = 0.70
    VAL_FRAC   = 0.15
    SPLIT_SEED = 42

    USE_CV      = True
    N_FOLDS     = 5
    CV_VAL_FRAC = 0.15

    # ── Ablation ──────────────────────────────────────────────────────────
    # 'full'     — multi-view MRI + clinical (default performance run)
    # 'clinical' — tabular only (fast RF-comparison diagnostic; minutes)
    # 'mri'      — MRI views only
    ABLATION = 'full'

    # ── Spiking neural network ────────────────────────────────────────────
    # Capacity cut hard: two independent 4-stage SNNs were too big for OASIS-2.
    # Shared encoder + 3 stages + smaller feat dim ≈ 4× fewer MRI params.
    T_STEPS          = 8
    LIF_TAU          = 2.0
    LIF_THRESHOLD    = 1.0
    SURROGATE_ALPHA  = 4.0
    SNN_INPUT_CODING = 'direct'
    ENCODER_CHANNELS = (16, 32, 64)   # was (16, 32, 64, 128)
    ENCODER_DROPOUT  = 0.10
    VIEW_FEAT_DIM    = 64             # was 128
    SHARE_MRI_ENCODER = True          # one SNN for all views
    SNN_BLOCKS_PER_STAGE = 1

    # ── MRI views / preprocessing ─────────────────────────────────────────
    # INTERIM (no FSL / no overnight preprocess): whole-brain RAW only +
    # brain-mask zeroing. Hippocampus/ventricle crops + GM/ROI channels need
    # registered volumes from ``mri_preprocess.py`` (enable later).
    VIEW_SHAPE = (64, 64, 64)
    MRI_VIEWS  = {
        'whole': (0.00, 1.00, 0.00, 1.00, 0.00, 1.00),
        # Registered-only ROI boxes (kept for later; not used while disabled):
        # 'hippocampus': (0.28, 0.58, 0.32, 0.68, 0.22, 0.78),
        # 'ventricles':  (0.38, 0.72, 0.38, 0.68, 0.38, 0.62),
    }
    MRI_CENTER_CROP_FRAC = 0.08
    MRI_PERCENTILE_CLIP  = (1.0, 99.0)

    # Offline registered pipeline — off until you run mri_preprocess.py on WSL.
    MRI_USE_PROCESSED = False
    MRI_REQUIRE_PROCESSED = False
    MRI_PROCESSED_DIR = OASIS2_DIR / 'processed_v5'
    MRI_TEMPLATE_PATH = OASIS2_DIR / 'templates' / 'MNI152_T1_1mm_brain.nii.gz'
    MRI_USE_GM_CHANNEL = False
    MRI_USE_ROI_PRIOR_CHANNELS = False

    MRI_CACHE_DIR = 'data/oasis2/cache_v5'  # v5 = brain-mask zeroed RAW tensors
    MRI_ZERO_BACKGROUND = True             # cheap win: drop air/skull from encoder

    @staticmethod
    def mri_in_channels() -> int:
        """Number of MRI channels fed to the SNN (T1 [+ GM] [+ ROI priors])."""
        n = 1
        if Config.MRI_USE_GM_CHANNEL:
            n += 1
        if Config.MRI_USE_ROI_PRIOR_CHANNELS:
            n += 2
        return n

    # ── Clinical branch ───────────────────────────────────────────────────
    # RF baseline (~0.52 macro-F1) shows tabular is the strong signal — give
    # it enough capacity and fuse it first in DS combination.
    USE_CLINICAL      = True
    CLINICAL_FEAT_DIM = 64
    CLINICAL_FIRST    = True          # clinical BBA is the initial mass
    CLINICAL_LR       = 1e-4          # match stronger full-run recipe

    # ── Dempster-Shafer ───────────────────────────────────────────────────
    EVIDENCE_CAP   = 18.0
    # Unaligned RAW MRI is weak: keep BBAs near-vacuous so full ≈ clinical
    # until registered multi-view preprocess is available.
    MRI_EVIDENCE_CAP   = 2.5
    MRI_EVIDENCE_SCALE = 0.12
    # Soft Dempster↔Yager gate — restore operating band from better full run.
    CONFLICT_KAPPA = 0.12
    CONFLICT_REPORT_KAPPA = 0.20

    # Aux head: clinical features only when clinical is active.
    AUX_CLINICAL_ONLY = True

    # Prefer clinical aux in the hard decision while MRI still trains via DS.
    # (Best full log used 0.7; 0.5 over-weighted noisy BetP.)
    DECISION_BLEND = 0.75

    # ── Loss weights ──────────────────────────────────────────────────────
    # Sampler already balances class exposure ≈ 1/3. Do NOT also put full
    # inv-freq weights in the loss — that double-counts and collapses to MCI.
    LAMBDA_AUX      = 1.5
    LAMBDA_DS       = 0.15
    LAMBDA_K        = 0.02
    LAMBDA_IGN      = 0.10
    LABEL_SMOOTHING = 0.02
    FOCAL_GAMMA     = 1.0
    MCI_LOSS_BOOST  = 1.0

    IGNORANCE_TARGET_CN  = 0.10
    IGNORANCE_TARGET_MCI = 0.15
    IGNORANCE_TARGET_AD  = 0.10

    # ── Training ──────────────────────────────────────────────────────────
    BATCH_SIZE       = 4
    GRAD_ACCUM_STEPS = 4
    NUM_EPOCHS       = 45
    LEARNING_RATE    = 5e-4           # cold MRI / SNN LR (see CLINICAL_LR)
    WEIGHT_DECAY     = 1e-4
    MIN_LR           = 1e-6
    # Tiny val (~18 subjects): ep1 can look great by chance. Do not checkpoint
    # or start the patience clock until CHECKPOINT_WARMUP, and never early-stop
    # before MIN_EPOCHS.
    CHECKPOINT_WARMUP = 10
    MIN_EPOCHS        = 30
    PATIENCE          = 10
    # Conservative selection (fix 5): reject lucky val peaks that don't
    # agree with train, and never save a ckpt that misses a class on val.
    CKPT_REQUIRE_ALL_CLASSES = True
    CKPT_MAX_VAL_TRAIN_GAP   = 0.15   # reject if val_f1 - train_f1 > this
    CKPT_TRAIN_F1_WEIGHT     = 0.25   # score mixes in train macro-F1

    USE_LOSS_CLASS_WEIGHTS = False

    # ── Device ────────────────────────────────────────────────────────────
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        import nibabel  # noqa: F401
        NIBABEL_AVAILABLE = True
    except ImportError:
        NIBABEL_AVAILABLE = False
