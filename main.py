"""Entry point: subject-level cross-validated DS-SNN-MF experiment.

    python main.py                                  # OASIS-3, 5-fold CV
    python main.py --repeats 3                      # 3x repeated CV
    python main.py --ablation full --single-split
    python main.py --ablation mri+pet+clinical
"""
import argparse
import logging
from pathlib import Path

import torch

from config import cfg, parse_ablation
from ExperimentModule import run_experiment

log = logging.getLogger("DS-SNN")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=list(cfg.DATASETS), default=cfg.DATASET)
    p.add_argument('--ablation', default=cfg.ABLATION,
                   help="full, or a plus-separated subset of mri, pet, ct, clinical")
    p.add_argument('--folds', type=int, default=cfg.N_FOLDS)
    p.add_argument('--repeats', type=int, default=cfg.REPEATS, help='repeated CV with different split seeds')
    p.add_argument('--train-visits', choices=['last', 'all'], default=None,
                   help="train on each subject's last visit or on every visit (default: per dataset)")
    p.add_argument('--seed', type=int, default=cfg.SEED)
    p.add_argument('--single-split', action='store_true', help='one train/val/test split instead of CV')
    p.add_argument('--out-dir', default=str(cfg.OUT_DIR))
    return p.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    cfg.DATASET, cfg.ABLATION = args.dataset, args.ablation
    parse_ablation(cfg.ABLATION)
    cfg.N_FOLDS, cfg.REPEATS, cfg.SEED = args.folds, args.repeats, args.seed
    cfg.TRAIN_VISITS = args.train_visits
    cfg.USE_CV = not args.single_split
    cfg.OUT_DIR = Path(args.out_dir)


def main() -> None:
    apply_args(parse_args())
    spec = cfg.spec
    log.info(f"Device: {cfg.DEVICE} | dataset={spec.name} | ablation={cfg.ABLATION} "
             f"sources={sorted(cfg.active_sources())} | "
             f"train_visits={spec.train_visits} | views={list(cfg.MRI_VIEWS)} | clinical={list(spec.clinical_cols)} | "
             f"kappa={cfg.CONFLICT_KAPPA} rule={cfg.CONFLICT_RULE} | clin_lr={cfg.CLINICAL_LR} "
             f"mri_lr={cfg.LEARNING_RATE} | blend_max={cfg.DECISION_BLEND_MAX}")
    run_experiment(torch.device(cfg.DEVICE))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Training interrupted. Exiting.")
