"""Single configuration object shared by every module.

Import the shared instance (``from config import cfg``) and change values at
runtime (e.g. ``cfg.DATASET = 'oasis3'``) before building datasets or models.
Dataset-specific values live in ``DATASETS`` and are read through ``cfg.spec``.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    clinical_cols: Tuple[str, ...]
    max_visits: int
    train_visits: str          # 'last': one sample per subject | 'all': every visit


class Config:
    """Hyperparameters and paths for the DS-SNN-MF pipeline.

    Primary protocol: subject-level stratified k-fold CV, clinical LR hot and
    MRI LR cold, evaluation on each subject's final visit.
    """

    # ── Task ──────────────────────────────────────────────────────────────
    CLASSES     = ['CN', 'MCI', 'AD']
    NUM_CLASSES = 3
    SEED        = 42
    NUM_WORKERS = 1

    # ── Data ──────────────────────────────────────────────────────────────
    DATASET    = 'oasis2'      # key of DATASETS
    DATA_ROOT  = Path("data")
    OASIS2_DIR = DATA_ROOT / "oasis2"
    OASIS2_CSV = OASIS2_DIR / "oasis_longitudinal_demographics.xlsx"
    OASIS3_DIR = DATA_ROOT / "oasis3"

    # CDR/MMSE are excluded from inputs (they define the label). Sex is a demographic;
    # eTIV/nWBV/ASF are imaging-derived volumes, not the diagnosis.
    DATASETS: Dict[str, dict] = {
        'oasis2': dict(root=OASIS2_DIR, max_visits=5, train_visits='last',
                       clinical_cols=('Age', 'EDUC', 'SES', 'eTIV', 'nWBV', 'ASF', 'MF')),
        # SES/ASF do not exist in OASIS-3; 'APOE4' (allele count) can be added.
        'oasis3': dict(root=OASIS3_DIR, max_visits=8, train_visits='all',
                       clinical_cols=('Age', 'EDUC', 'MF', 'eTIV', 'nWBV')),
    }
    TRAIN_VISITS = None        # None → dataset default; 'last' | 'all'

    OASIS3_CDR_FILE          = 'OASIS3_UDSb4_cdr.csv'
    OASIS3_DEMOGRAPHICS_FILE = 'OASIS3_demographics.csv'
    OASIS3_FREESURFER_FILE   = 'OASIS3_Freesurfer_output.csv'
    OASIS3_T1_GLOB           = '*T1w.nii*'
    MAX_CDR_GAP_DAYS         = 365   # max |MR day − CDR day| for pairing a scan with a label

    MRI_SHAPE          = (96, 96, 96)
    PROCESSED_SUBDIR   = 'processed_v5'
    MRI_PERCENTILE_CLIP = (1.0, 98.0)
    MRI_USE_GM_CHANNEL = False

    # ── Splits ────────────────────────────────────────────────────────────
    SPLIT_SEED  = 42
    USE_CV      = True
    N_FOLDS     = 5
    REPEATS     = 1            # repeated CV: repeat r uses SPLIT_SEED + r
    CV_VAL_FRAC = 0.15
    TRAIN_FRAC  = 0.70         # single-split mode only
    VAL_FRAC    = 0.15

    # ── Ablation ──────────────────────────────────────────────────────────
    # 'full' (MRI + clinical) | 'clinical' (tabular only) | 'mri' (MRI views only)
    ABLATION = 'full'
    ABLATION_PURE_DECISION = True  # ablations: prediction = BetP only

    # ── Spiking neural network ────────────────────────────────────────────
    T_STEPS          = 8
    LIF_TAU          = 2.0
    LIF_THRESHOLD    = 1.0
    SURROGATE_ALPHA  = 4.0
    SNN_INPUT_CODING = 'direct'
    ENCODER_CHANNELS = (16, 32, 64)
    ENCODER_DROPOUT  = 0.10
    VIEW_FEAT_DIM    = 64
    SHARE_MRI_ENCODER = True
    SNN_BLOCKS_PER_STAGE = 1

    # ── MRI views ─────────────────────────────────────────────────────────
    VIEW_SHAPE = (64, 64, 64)
    MRI_VIEWS  = {
        # 'whole':       (0.00, 1.00, 0.00, 1.00, 0.00, 1.00),
        # 'hippocampus': (0.28, 0.58, 0.32, 0.68, 0.22, 0.78),
        # 'ventricles':  (0.38, 0.72, 0.38, 0.68, 0.38, 0.62),
        'mtl': (0.22, 0.56, 0.28, 0.74, 0.18, 0.82),
        # 'amygdala':    (0.30, 0.55, 0.45, 0.70, 0.25, 0.75),
    }

    # ── Clinical branch ───────────────────────────────────────────────────
    USE_CLINICAL      = True
    CLINICAL_FEAT_DIM = 64
    CLINICAL_LR       = 7e-4

    # ── Dempster-Shafer ───────────────────────────────────────────────────
    EVIDENCE_CAP   = 18.0
    MRI_EVIDENCE_CAP = 9.0
    MRI_EVIDENCE_CAP_STANDALONE = 18.0
    CLINICAL_EVIDENCE_CAP_STANDALONE = 18.0
    MRI_EVIDENCE_SCALE = 0.7

    # Soft gate uses CONFLICT_RULE when K > kappa. One of:
    #   'yager'   all conflict mass redirected onto Θ
    #   'ordinal' adjacent-class conflict → {CN,MCI}/{MCI,AD}; CN↔AD and compound → Θ
    #   'pcr5'    proportional conflict redistribution onto the conflicting pair
    CONFLICT_RULE  = 'pcr5'
    CONFLICT_KAPPA = 0.25
    CONFLICT_REPORT_KAPPA = 0.20
    CONFLICT_GAMMA = 20.0
    AUX_CLINICAL_ONLY = True
    # Low K → fused BetP; high K → up to DECISION_BLEND_MAX of the clinical aux head.
    DECISION_BLEND = 0.0
    DECISION_BLEND_MAX = 0.50
    DECISION_BLEND_KAPPA = 0.20
    DECISION_BLEND_SHARPNESS = 20.0
    BRANCH_RELIABILITY = {'mtl': 0.90, 'clinical': 1.0}

    # ── Loss ──────────────────────────────────────────────────────────────
    LAMBDA_AUX      = 0.8
    LAMBDA_AUX_STANDALONE = 0.8
    LAMBDA_DS       = 0.15
    LAMBDA_K        = 0.08
    LAMBDA_IGN      = 0.05
    LAMBDA_IGN_STANDALONE = 0.0
    LABEL_SMOOTHING = 0.02
    FOCAL_GAMMA     = 1.0
    MCI_LOSS_BOOST  = 1.0
    USE_LOSS_CLASS_WEIGHTS = False

    IGNORANCE_TARGET_CN  = 0.10
    IGNORANCE_TARGET_MCI = 0.15
    IGNORANCE_TARGET_AD  = 0.10

    # ── Training ──────────────────────────────────────────────────────────
    BATCH_SIZE       = 4
    GRAD_ACCUM_STEPS = 4
    NUM_EPOCHS       = 40
    LEARNING_RATE    = 7e-4           # MRI / SNN LR
    WEIGHT_DECAY     = 1e-4
    MIN_LR           = 1e-6

    CHECKPOINT_WARMUP = 10
    MIN_EPOCHS        = 30
    PATIENCE          = 10
    CKPT_REQUIRE_ALL_CLASSES = True
    CKPT_MAX_VAL_TRAIN_GAP   = 0.15   # reject checkpoints with val_f1 − train_f1 above this
    CKPT_TRAIN_F1_WEIGHT     = 0.10

    # ── Evaluation / output ───────────────────────────────────────────────
    BOOTSTRAP_ROUNDS = 1000
    SELECTIVE_COVERAGE = 0.80
    OUT_DIR = Path("runs")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Derived values ────────────────────────────────────────────────────
    @property
    def spec(self) -> DatasetSpec:
        if self.DATASET not in self.DATASETS:
            raise KeyError(f"Unknown dataset {self.DATASET!r}; choose from {list(self.DATASETS)}")
        values = dict(self.DATASETS[self.DATASET])
        if self.TRAIN_VISITS:
            values['train_visits'] = self.TRAIN_VISITS
        if values['train_visits'] not in ('last', 'all'):
            raise ValueError(f"train_visits must be 'last' or 'all', got {values['train_visits']!r}")
        values['clinical_cols'] = tuple(values['clinical_cols'])
        return DatasetSpec(name=self.DATASET, **values)

    @property
    def CLINICAL_COLS(self) -> Tuple[str, ...]:
        return self.spec.clinical_cols

    @property
    def MRI_PROCESSED_DIR(self) -> Path:
        return self.spec.root / self.PROCESSED_SUBDIR

    def mri_in_channels(self) -> int:
        return 2 if self.MRI_USE_GM_CHANNEL else 1

    def snapshot(self) -> dict:
        """JSON-friendly copy of every setting, for run provenance."""
        names = [n for n in dir(self) if n.isupper()] + ['CLINICAL_COLS', 'MRI_PROCESSED_DIR']
        out = {n: getattr(self, n) for n in names}
        out['spec'] = self.spec.__dict__
        return out


cfg = Config()
