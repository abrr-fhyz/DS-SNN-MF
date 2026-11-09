"""Frozen tabular ensemble used as the clinical DS source.

Members (equal soft votes): LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees,
RBF-SVM. Boosting libraries that are not installed are dropped with a warning.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from torch import Tensor

from config import cfg
from MODULE.DSModule import FULL_FRAME, SINGLETONS

log = logging.getLogger("DS-SNN")

_KNOWN_MEMBERS = ('lgbm', 'xgb', 'catboost', 'rf', 'extratrees', 'svm')


def proba_to_bba(p: Tensor, cap: Optional[float] = None, n_classes: int = cfg.NUM_CLASSES) -> Tensor:
    """Map a class simplex to a subjective-logic BBA (same form as ``DSEvidenceLayer``)."""
    if p.ndim != 2 or p.shape[1] != n_classes:
        raise ValueError(f"expected (B, {n_classes}) probabilities, got {tuple(p.shape)}")
    cap = float(cfg.CLINICAL_EVIDENCE_CAP_STANDALONE if cap is None else cap)
    evidence = p.clamp(min=0.0) * cap
    strength = evidence.sum(dim=1, keepdim=True) + n_classes
    m = p.new_zeros(p.shape[0], 8)
    idx = torch.as_tensor(SINGLETONS, device=p.device, dtype=torch.long)
    m[:, idx] = evidence / strength
    m[:, FULL_FRAME] = (n_classes / strength).squeeze(1)
    return m


def ensemble_ckpt_path(ckpt_path) -> Path:
    """``fold_r1f1_best.pt`` → ``fold_r1f1_best.clinical_ensemble.joblib``."""
    path = Path(ckpt_path)
    return path.with_name(path.stem + '.clinical_ensemble.joblib')


def _optional_import(module: str):
    try:
        return __import__(module)
    except ImportError:
        return None


def _estimators(seed: int, names: Sequence[str]) -> List[Tuple[str, object]]:
    wanted = [n.strip().lower() for n in names if n and n.strip()]
    unknown = [n for n in wanted if n not in _KNOWN_MEMBERS]
    if unknown:
        raise ValueError(f"Unknown clinical ensemble members {unknown}; choose from {_KNOWN_MEMBERS}")
    built: List[Tuple[str, object]] = []
    for name in wanted:
        est = _make_estimator(name, seed)
        if est is None:
            log.warning(f"Clinical ensemble: skipping {name!r} (package not installed)")
            continue
        built.append((name, est))
    if len(built) < 2:
        raise RuntimeError(
            f"Clinical ensemble needs at least two members; built { [n for n, _ in built] }. "
            "Install xgboost, lightgbm, catboost, or keep sklearn rf/extratrees/svm.")
    return built


def _make_estimator(name: str, seed: int):
    if name == 'lgbm':
        lgbm = _optional_import('lightgbm')
        if lgbm is None:
            return None
        return lgbm.LGBMClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, class_weight='balanced', random_state=seed,
            verbosity=-1, n_jobs=1)
    if name == 'xgb':
        xgb = _optional_import('xgboost')
        if xgb is None:
            return None
        return xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, objective='multi:softprob', eval_metric='mlogloss',
            tree_method='hist', random_state=seed, verbosity=0, n_jobs=1)
    if name == 'catboost':
        cb = _optional_import('catboost')
        if cb is None:
            return None
        return cb.CatBoostClassifier(
            iterations=100, depth=3, learning_rate=0.05, random_seed=seed,
            auto_class_weights='Balanced', verbose=0, allow_writing_files=False,
            thread_count=1)
    if name == 'rf':
        return RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=2,
            class_weight='balanced_subsample', random_state=seed, n_jobs=1)
    if name == 'extratrees':
        return ExtraTreesClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=2,
            class_weight='balanced_subsample', random_state=seed, n_jobs=1)
    if name == 'svm':
        return SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=seed)
    raise ValueError(name)


class ClinicalEnsemble(nn.Module):
    """Sklearn voting ensemble wrapped as a torch module. ``forward`` returns class probabilities."""

    def __init__(self, n_features: int, n_classes: int = cfg.NUM_CLASSES,
                 members: Optional[Sequence[str]] = None):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.requested = tuple(members if members is not None else cfg.CLINICAL_ENSEMBLE_MEMBERS)
        self.vote: Optional[VotingClassifier] = None
        self.member_names: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ClinicalEnsemble':
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"expected (N, {self.n_features}) clinical matrix, got {X.shape}")
        if np.isnan(X).any():
            raise ValueError("Clinical ensemble needs imputed features (ClinicalScaler).")
        estimators = _estimators(cfg.SEED, self.requested)
        self.member_names = [n for n, _ in estimators]
        combine = str(getattr(cfg, 'CLINICAL_ENSEMBLE_COMBINE', 'soft_vote')).lower()
        if combine not in ('soft_vote', 'vote', 'soft'):
            raise ValueError(f"CLINICAL_ENSEMBLE_COMBINE must be soft_vote, got {combine!r}")
        self.vote = VotingClassifier(estimators=estimators, voting='soft')
        self.vote.fit(X, y)
        log.info(f"Clinical ensemble fitted on {len(y)} rows | members={self.member_names}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.vote is None:
            raise RuntimeError("ClinicalEnsemble.fit has not been called.")
        X = np.asarray(X, dtype=np.float64)
        p = self.vote.predict_proba(X)
        out = np.zeros((len(X), self.n_classes), dtype=np.float64)
        for i, cls in enumerate(self.vote.classes_):
            out[:, int(cls)] = p[:, i]
        return out

    def forward(self, x: Tensor) -> Tensor:
        p = self.predict_proba(x.detach().cpu().numpy())
        return torch.as_tensor(p, device=x.device, dtype=x.dtype)

    def bba(self, x: Tensor, cap: Optional[float] = None) -> Tensor:
        return proba_to_bba(self.forward(x), cap=cap, n_classes=self.n_classes)

    def save(self, path) -> None:
        import joblib
        if self.vote is None:
            raise RuntimeError("Cannot save an unfitted clinical ensemble.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'vote': self.vote, 'member_names': self.member_names,
                     'n_features': self.n_features, 'n_classes': self.n_classes}, path)

    def load(self, path) -> None:
        import joblib
        blob = joblib.load(Path(path))
        self.vote = blob['vote']
        self.member_names = list(blob.get('member_names') or [])
        self.n_features = int(blob.get('n_features', self.n_features))
        self.n_classes = int(blob.get('n_classes', self.n_classes))
