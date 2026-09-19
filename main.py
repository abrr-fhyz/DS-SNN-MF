import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from config import Config
from OasisModule import Oasis2Dataset, make_stratified_folds, subject_labels
from DSModule import DSSNMMF
from TrainerModule import Trainer, evaluate_longitudinal

cfg = Config()

# ──────────────────────────────────────────────────────────────
# Subject-level view: one sample per subject = final visit
# ──────────────────────────────────────────────────────────────

class CrossSectionalWrapper(Dataset):
    def __init__(self, oasis_dataset: Oasis2Dataset, augment: bool = False):
        self.oasis_dataset = oasis_dataset
        self.augment = augment
        self.samples = [(seq[-1], int(seq[-1]['label']))
                        for seq in oasis_dataset.sequences]

    def __len__(self):
        return len(self.samples)

    def _augment_mri(self, mri: torch.Tensor) -> torch.Tensor:
        """Light 3D augmentation for the small OASIS-2 training set. mri: (C,D,H,W)."""
        if random.random() < 0.5:
            mri = torch.flip(mri, dims=[-1])                    # sagittal mirror
        shifts = [random.randint(-2, 2) for _ in range(3)]
        if any(shifts):
            mri = torch.roll(mri, shifts=tuple(shifts), dims=(-3, -2, -1))
        # Intensity jitter on T1 (ch0); keep GM/ROI priors geometrically transformed only.
        t1 = mri[0:1]
        t1 = t1 * (1.0 + (random.random() - 0.5) * 0.1) + (random.random() - 0.5) * 0.05
        if random.random() < 0.3:
            t1 = t1.clamp(min=0.0).pow(1.0 + (random.random() - 0.5) * 0.3)   # gamma
        if random.random() < 0.3:
            t1 = t1 + torch.randn_like(t1) * 0.01
        t1 = t1.clamp(0.0, 1.0)
        if mri.shape[0] == 1:
            return t1
        rest = mri[1:].clamp(0.0, 1.0)
        return torch.cat([t1, rest], dim=0)

    def __getitem__(self, idx):
        visit, label = self.samples[idx]
        # Clinical-only ablation: skip expensive MRI I/O (zeros; unused by model).
        if getattr(cfg, 'ABLATION', 'full') == 'clinical':
            c = int(cfg.mri_in_channels()) if hasattr(cfg, 'mri_in_channels') else 1
            mri = torch.zeros((c, *cfg.MRI_SHAPE), dtype=torch.float32)
        else:
            mri = self.oasis_dataset.load_mri_tensor(visit)
            if self.augment:
                mri = self._augment_mri(mri)
        return {
            'mri':      mri,
            'clinical': torch.tensor(visit['clinical'], dtype=torch.float32),
            'label':    torch.tensor(label, dtype=torch.long),
        }

    def label_counts(self, num_classes: int = cfg.NUM_CLASSES):
        counts = [0] * num_classes
        for _, label in self.samples:
            counts[int(label)] += 1
        return counts

# ──────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────

def set_seed(seed: int = cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def longitudinal_collate_fn(batch):
    """Pad each subject's visit sequence to the batch max and keep visit order."""
    labels     = torch.stack([item['label'] for item in batch])
    max_visits = max(len(item['visits']) for item in batch)

    visits_by_idx = []
    for v in range(max_visits):
        mri_list, clin_list, has_visit = [], [], []
        for item in batch:
            if v < len(item['visits']):
                visit = item['visits'][v]
                mri_list.append(visit['mri'])
                clin_list.append(visit['clinical'])
                has_visit.append(True)
            else:
                ref = item['visits'][0]
                mri_list.append(torch.zeros_like(ref['mri']))
                clin_list.append(torch.zeros_like(ref['clinical']))
                has_visit.append(False)
        visits_by_idx.append({
            'mri':       torch.stack(mri_list),
            'clinical':  torch.stack(clin_list),
            'has_visit': has_visit,
        })
    return {'visits': visits_by_idx, 'label': labels}

def _inv_freq_weights(counts):
    total = sum(counts)
    return torch.tensor(
        [total / (cfg.NUM_CLASSES * max(c, 1)) for c in counts],
        dtype=torch.float32,
    )

def _param_groups(model: DSSNMMF):
    """Clinical / aux hot; MRI encoder + MRI DS heads cold (full / mri ablations)."""
    clin_lr = float(getattr(cfg, 'CLINICAL_LR', 1e-3))
    mri_lr  = float(cfg.LEARNING_RATE)

    if not getattr(model, 'use_mri', False) or not getattr(model, 'use_clinical', False):
        lr = clin_lr if getattr(model, 'use_clinical', False) else mri_lr
        return [{'params': list(model.parameters()), 'lr': lr}]

    hot: list = []
    if hasattr(model, 'clinical_encoder'):
        hot += list(model.clinical_encoder.parameters())
    if hasattr(model, 'ds_clinical'):
        hot += list(model.ds_clinical.parameters())
    hot += list(model.aux_head.parameters())
    hot_ids = {id(p) for p in hot}
    cold = [p for p in model.parameters() if id(p) not in hot_ids]
    groups = [{'params': hot, 'lr': clin_lr}]
    if cold:
        groups.append({'params': cold, 'lr': mri_lr})
    return groups

def _make_loaders(train_seq, val_seq, test_seq):
    train_ds = CrossSectionalWrapper(train_seq, augment=True)
    val_ds   = CrossSectionalWrapper(val_seq,   augment=False)
    test_ds  = CrossSectionalWrapper(test_seq,  augment=False)

    counts = train_ds.label_counts()
    inv_freq = _inv_freq_weights(counts)
    print(f"  Train class counts: {counts}  →  sampler weights: "
          f"{[round(w, 3) for w in inv_freq.tolist()]}")

    sample_weights = [inv_freq[label].item() for _, label in train_ds.samples]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, sampler=sampler,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True)
    test_xs_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.NUM_WORKERS, pin_memory=True)
    test_long_loader = DataLoader(test_seq, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                  num_workers=cfg.NUM_WORKERS,
                                  collate_fn=longitudinal_collate_fn)

    # Full inv-freq CE weights + extra MCI boost (the class that collapses).
    if cfg.USE_LOSS_CLASS_WEIGHTS:
        class_weights = inv_freq.clone()
        class_weights[1] *= cfg.MCI_LOSS_BOOST
    else:
        class_weights = torch.ones(cfg.NUM_CLASSES)

    return train_loader, val_loader, test_xs_loader, test_long_loader, class_weights, counts

def _run_one_split(train_seq, val_seq, test_seq, device, save_path: str) -> dict:
    train_loader, val_loader, test_xs_loader, test_long_loader, class_weights, counts = \
        _make_loaders(train_seq, val_seq, test_seq)

    model   = DSSNMMF().to(device)
    trainer = Trainer(model, device=device, class_weights=class_weights,
                      param_groups=_param_groups(model))

    trainer.fit(train_loader, val_loader, save_path=save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))
    test_metrics = trainer.evaluate(test_xs_loader, verbose=True)
    long_metrics = evaluate_longitudinal(model, test_long_loader, device=device)

    cm = test_metrics['confusion']
    bal_acc = float(np.mean(np.diag(cm) / np.clip(cm.sum(axis=1), 1, None)))

    result = {
        'train_counts': counts,
        'val_acc':      trainer.best_val_acc,
        'val_macro_f1': trainer.best_val_f1,
        'val_score':    trainer.best_score,
        'test_acc':     test_metrics['acc'],
        'test_macro_f1': test_metrics['macro_f1'],
        'test_bal_acc': bal_acc,
        'test_per_class_f1': test_metrics['per_class_f1'],
        'test_mean_K':  test_metrics['mean_K'],
        'test_conflict_rate': test_metrics['conflict_rate'],
        'test_mean_gate': test_metrics['mean_gate'],
        'test_mean_decision_blend': test_metrics.get('mean_decision_blend', 0.0),
        'longitudinal_acc': long_metrics['longitudinal_acc'],
        'longitudinal_acc_dempster': long_metrics.get('longitudinal_acc_dempster',
                                                     long_metrics['longitudinal_acc']),
        'confusion':    cm.tolist(),
    }
    print(f"\n  Cross-sectional TEST | acc={result['test_acc']:.4f} "
          f"macro_f1={result['test_macro_f1']:.4f} bal_acc={result['test_bal_acc']:.4f} "
          f"per_class_f1={[f'{f:.2f}' for f in result['test_per_class_f1']]} "
          f"meanK={result['test_mean_K']:.3f} hi-rate={result['test_conflict_rate']:.2f} "
          f"gate={result['test_mean_gate']:.2f} blend={result['test_mean_decision_blend']:.2f}")
    print(f"  Longitudinal TEST (last-visit): {result['longitudinal_acc']:.4f}  |  "
          f"equal-Dempster accum: {result['longitudinal_acc_dempster']:.4f}")
    return result

def _summarise(fold_results: list) -> dict:
    keys = ['test_acc', 'test_macro_f1', 'test_bal_acc', 'test_mean_K',
            'test_conflict_rate', 'test_mean_gate', 'test_mean_decision_blend',
            'longitudinal_acc',
            'longitudinal_acc_dempster', 'val_acc', 'val_macro_f1']
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in fold_results], dtype=np.float64)
        summary[k] = {'mean': float(vals.mean()), 'std': float(vals.std(ddof=1)),
                      'folds': vals.tolist()}

    pc = np.array([r['test_per_class_f1'] for r in fold_results], dtype=np.float64)
    summary['test_per_class_f1'] = {
        cfg.CLASSES[i]: {
            'mean': float(pc[:, i].mean()),
            'std':  float(pc[:, i].std(ddof=1)),
            'folds': pc[:, i].tolist(),
        }
        for i in range(cfg.NUM_CLASSES)
    }
    return summary

def _print_summary(summary: dict, n_folds: int):
    print("\n" + "=" * 72)
    print(f"SUBJECT-LEVEL {n_folds}-FOLD CV SUMMARY  (mean ± std)")
    print("=" * 72)
    def line(name, key):
        s = summary[key]
        print(f"  {name:<28} {s['mean']:.4f} ± {s['std']:.4f}   "
              f"folds={[f'{v:.3f}' for v in s['folds']]}")
    line("Test accuracy",            'test_acc')
    line("Test macro-F1",            'test_macro_f1')
    line("Test balanced accuracy",   'test_bal_acc')
    line("Test meanK",               'test_mean_K')
    line("Test high-conflict rate",  'test_conflict_rate')
    line("Test mean soft-gate",      'test_mean_gate')
    line("Test mean decision blend", 'test_mean_decision_blend')
    line("Longitudinal (last-visit)", 'longitudinal_acc')
    line("Longitudinal (Dempster)",   'longitudinal_acc_dempster')
    line("Val accuracy (selected)",  'val_acc')
    line("Val macro-F1 (selected)",  'val_macro_f1')
    print("  Per-class F1 (test):")
    for cname, s in summary['test_per_class_f1'].items():
        print(f"    {cname:<6} {s['mean']:.4f} ± {s['std']:.4f}   "
              f"folds={[f'{v:.3f}' for v in s['folds']]}")

    rf_f1 = 0.519
    f1_folds = summary['test_macro_f1']['folds']
    n_beat = sum(1 for v in f1_folds if v >= rf_f1)
    print("-" * 72)
    print(f"  Folds with macro-F1 ≥ RF baseline ({rf_f1:.3f}): "
          f"{n_beat}/{n_folds}")
    print("=" * 72)
    print("Primary paper metric: Test macro-F1 and Test accuracy (mean ± std).")
    print("Val numbers are for early-stopping only — do not report them as results.")

def run_cv(device: torch.device):
    print(f"Loading OASIS-2 sequences for {cfg.N_FOLDS}-fold subject-level CV...")
    all_sequences = Oasis2Dataset.load_all_sequences()
    y = subject_labels(all_sequences)
    print(f"  Subjects: {len(all_sequences)} | "
          f"final-label counts CN/MCI/AD = "
          f"{[int((y == c).sum()) for c in range(cfg.NUM_CLASSES)]}")
    folds = make_stratified_folds(all_sequences, n_folds=cfg.N_FOLDS, seed=cfg.SPLIT_SEED, val_frac=cfg.CV_VAL_FRAC)
    fold_results = []
    out_dir = Path("cv_runs")
    out_dir.mkdir(exist_ok=True)

    for fold_i, indices in enumerate(folds):
        print("\n" + "#" * 72)
        print(f"# FOLD {fold_i + 1}/{cfg.N_FOLDS}  "
              f"train={len(indices['train'])}  val={len(indices['val'])}  "
              f"test={len(indices['test'])}")
        print("#" * 72)
        set_seed(cfg.SEED + fold_i)

        train_seq = Oasis2Dataset(split='train', fold_indices=indices, all_sequences=all_sequences)
        val_seq   = Oasis2Dataset(split='val',   fold_indices=indices, all_sequences=all_sequences)
        test_seq  = Oasis2Dataset(split='test',  fold_indices=indices, all_sequences=all_sequences)

        save_path = str(out_dir / f"fold{fold_i + 1}_best.pt")
        result = _run_one_split(train_seq, val_seq, test_seq, device, save_path)
        result['fold'] = fold_i + 1
        fold_results.append(result)

        with open(out_dir / f"fold{fold_i + 1}_metrics.json", "w") as f:
            json.dump({k: v for k, v in result.items()}, f, indent=2)

    summary = _summarise(fold_results)
    _print_summary(summary, cfg.N_FOLDS)

    with open(out_dir / "cv_summary.json", "w") as f:
        json.dump({'folds': fold_results, 'summary': summary}, f, indent=2)
    print(f"\nWrote per-fold checkpoints + metrics to {out_dir}/")
    return summary

def run_single_split(device: torch.device):
    print("Running single subject-level train/val/test split (USE_CV=False)...")
    set_seed()
    train_seq = Oasis2Dataset(split='train')
    val_seq   = Oasis2Dataset(split='val')
    test_seq  = Oasis2Dataset(split='test')
    result = _run_one_split(train_seq, val_seq, test_seq, device, "ds_snn_mf_best.pt")
    return result

# ──────────────────────────────────────────────────────────────
# Main Function(s)
# ──────────────────────────────────────────────────────────────

def main():
    device = torch.device(cfg.DEVICE)
    n_ch = int(cfg.mri_in_channels()) if hasattr(cfg, 'mri_in_channels') else 1
    views = list(getattr(cfg, 'MRI_VIEWS', {}).keys())
    print(f"Device: {device} | ablation={cfg.ABLATION} | protocol=last-visit-only | "
          f"kappa={cfg.CONFLICT_KAPPA} | clin_lr={getattr(cfg, 'CLINICAL_LR', 1e-3)} | "
          f"mri_lr={cfg.LEARNING_RATE} | blend={cfg.DECISION_BLEND} | "
          f"aux_clin_only={getattr(cfg, 'AUX_CLINICAL_ONLY', False)} | "
          f"mri_scale={getattr(cfg, 'MRI_EVIDENCE_SCALE', 1.0)} | "
          f"channels={cfg.ENCODER_CHANNELS} | share_enc={cfg.SHARE_MRI_ENCODER} | "
          f"focalγ={cfg.FOCAL_GAMMA} | mri_in_ch={n_ch} | views={views} | "
          f"processed={getattr(cfg, 'MRI_USE_PROCESSED', False)} | "
          f"clinical={cfg.CLINICAL_COLS}")
    if cfg.USE_CV:
        run_cv(device)
    else:
        run_single_split(device)

def testSpecificFold(thisfold, device: torch.device):
    print(f"Loading OASIS-2 sequences for {cfg.N_FOLDS}-fold subject-level CV...")
    all_sequences = Oasis2Dataset.load_all_sequences()
    y = subject_labels(all_sequences)
    print(f"  Subjects: {len(all_sequences)} | "
              f"final-label counts CN/MCI/AD = "
              f"{[int((y == c).sum()) for c in range(cfg.NUM_CLASSES)]}")
    
    folds = make_stratified_folds(all_sequences, n_folds=cfg.N_FOLDS, seed=cfg.SPLIT_SEED, val_frac=cfg.CV_VAL_FRAC)
    fold_results = []
    out_dir = Path("test_runs")
    out_dir.mkdir(exist_ok=True)
    fold_i = thisfold
    indices = folds[fold_i]
    for _ in range(5):
        print("\n" + "#" * 72)
        print(f"# FOLD {fold_i + 1}/{cfg.N_FOLDS}  "
                  f"train={len(indices['train'])}  val={len(indices['val'])}  "
                  f"test={len(indices['test'])}")
        print("#" * 72)
        set_seed(cfg.SEED + fold_i)
        train_seq = Oasis2Dataset(split='train', fold_indices=indices, all_sequences=all_sequences)
        val_seq   = Oasis2Dataset(split='val',   fold_indices=indices, all_sequences=all_sequences)
        test_seq  = Oasis2Dataset(split='test',  fold_indices=indices, all_sequences=all_sequences)
    
        save_path = str(out_dir / f"fold{fold_i + 1}_best.pt")
        result = _run_one_split(train_seq, val_seq, test_seq, device, save_path)
        result['fold'] = fold_i + 1
        fold_results.append(result)
    
        with open(out_dir / f"fold{fold_i + 1}_metrics.json", "w") as f:
            json.dump({k: v for k, v in result.items()}, f, indent=2)
    
    summary = _summarise(fold_results)
    _print_summary(summary, cfg.N_FOLDS)
    
    with open(out_dir / "cv_summary.json", "w") as f:
        json.dump({'folds': fold_results, 'summary': summary}, f, indent=2)
    print(f"\nWrote per-fold checkpoints + metrics to {out_dir}/")
    return summary

if __name__ == "__main__":
    try:
        main()
        #device = torch.device(cfg.DEVICE)
        #testSpecificFold(4, device)
    except KeyboardInterrupt:
        print("Training interrupted. Exiting.")
