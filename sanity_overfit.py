"""MRI + Dempster–Shafer memorization probe (8 subjects).

Keeps the full DSSNMMF / DSSNNLoss stack. Goal: see whether the MRI path can
fit labels at all before chasing CV architecture tweaks.

  Pass: train accuracy → ~1.0 within STEPS (features + DS can memorize).
  Fail: stuck near chance / single-class collapse (preproc, BN, or dead encoder).

Run:
  python sanity_overfit.py
  python sanity_overfit.py --no-bn          # drop feat BatchNorm (collapse check)
  python sanity_overfit.py --steps 600 --n 8
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from config import Config
from OasisModule import Oasis2Dataset, subject_labels
from DSModule import DSSNMMF, DSSNNLoss, FULL_FRAME
from main import CrossSectionalWrapper, set_seed


def _apply_probe_config(disable_bn: bool) -> None:
    """MRI-only, full-strength evidence (undo full-model softener for this probe)."""
    Config.ABLATION = 'mri'
    Config.MRI_EVIDENCE_SCALE = 1.0
    Config.MRI_EVIDENCE_CAP = 15.0
    Config.DECISION_BLEND = 1.0
    Config.ENCODER_DROPOUT = 0.0
    Config.LABEL_SMOOTHING = 0.0
    Config.FOCAL_GAMMA = 0.0
    # Keep DS terms on — this is a DS stack probe, not an aux-only bypass.
    Config.LAMBDA_AUX = 1.5
    Config.LAMBDA_DS = 0.4
    Config.LAMBDA_IGN = 0.1
    Config.LAMBDA_K = 0.0  # single source ⇒ K is always 0
    # Prefer processed MRI when present; allow RAW fallback for local probes.
    Config.MRI_REQUIRE_PROCESSED = False
    Config._SANITY_DISABLE_BN = disable_bn  # read below after model build


def _pick_subjects(sequences: List[List[dict]], n: int, seed: int) -> np.ndarray:
    """Stratified pick across CN/MCI/AD (final-visit labels)."""
    y = subject_labels(sequences)
    rng = np.random.default_rng(seed)
    chosen: List[int] = []
    per = max(1, n // Config.NUM_CLASSES)
    for c in range(Config.NUM_CLASSES):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        take = min(per, len(idx))
        chosen.extend(rng.choice(idx, size=take, replace=False).tolist())
    rest = [i for i in range(len(sequences)) if i not in chosen]
    rng.shuffle(rest)
    while len(chosen) < n and rest:
        chosen.append(rest.pop())
    chosen = chosen[:n]
    if len(chosen) < n:
        raise RuntimeError(f"Only found {len(chosen)} subjects; need {n}.")
    return np.asarray(chosen, dtype=np.int64)


def _disable_feat_bn(model: DSSNMMF) -> None:
    encoders = []
    if getattr(model, 'shared_encoder', None) is not None:
        encoders.append(model.shared_encoder)
    if getattr(model, 'encoders', None) is not None:
        encoders.extend(model.encoders.values())
    for enc in encoders:
        if hasattr(enc, 'feat_bn'):
            enc.feat_bn = nn.Identity()
            print("  Replaced encoder.feat_bn with Identity")


def _make_tiny_dataset(all_sequences, indices: np.ndarray) -> Oasis2Dataset:
    """Oasis2Dataset whose train split is exactly the probe subjects."""
    fold = {
        'train': indices,
        'val':   indices[:1],
        'test':  indices[:1],
    }
    return Oasis2Dataset(split='train', fold_indices=fold, all_sequences=all_sequences)


def _load_all_samples(wrap: CrossSectionalWrapper):
    mris, clin, labs = [], [], []
    for i in range(len(wrap)):
        s = wrap[i]
        mris.append(s['mri'])
        clin.append(s['clinical'])
        labs.append(s['label'])
    return (torch.stack(mris), torch.stack(clin),
            torch.stack(labs).long())


def _report_input_diversity(mris: torch.Tensor, labels: torch.Tensor) -> None:
    """Did we load distinct volumes, or is the cache/loader collapsing subjects?"""
    flat = mris.reshape(mris.shape[0], -1)
    n = flat.shape[0]
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(torch.norm(flat[i] - flat[j]).item())
    # hash uniqueness at coarse precision
    uniq = len({tuple(np.round(flat[i].numpy(), 4).flatten()[::997].tolist())
                for i in range(n)})
    print("INPUT DIVERSITY")
    print(f"  MRI shape={tuple(mris.shape)}  min/max/mean="
          f"{float(mris.min()):.3f}/{float(mris.max()):.3f}/{float(mris.mean()):.3f}")
    print(f"  pairwise L2  min/median/max="
          f"{min(dists):.1f}/{float(np.median(dists)):.1f}/{max(dists):.1f}")
    print(f"  coarse unique volumes ≈ {uniq}/{n}  "
          f"(1/{n} ⇒ loader/cache returning identical tensors)")
    if uniq <= 1 and n > 1:
        print("  *** BUG LIKELY: identical MRI tensors across subjects.")
        print("      If using cache: ensure keys include session id (cache_v4),")
        print("      delete stale cache_v3, and re-run this probe.")
    counts = np.bincount(labels.numpy(), minlength=3)
    maj = float(counts.max() / counts.sum())
    print(f"  label counts CN/MCI/AD={counts.tolist()}  "
          f"majority-class acc baseline={maj:.3f}")


@torch.no_grad()
def _report_feature_diversity(model: DSSNMMF, mris: torch.Tensor,
                              device: torch.device, tag: str) -> torch.Tensor:
    import DSModule as dsm
    model.eval()
    view_name = list(Config.MRI_VIEWS.keys())[0]
    box = Config.MRI_VIEWS[view_name]
    feats = []
    for i in range(mris.shape[0]):
        view = dsm.extract_view(mris[i:i + 1].to(device), box, Config.VIEW_SHAPE)
        feats.append(model._encode_view(view_name, view).cpu())
    F = torch.cat(feats, dim=0)
    std = float(F.std(dim=0).mean())
    # mean off-diagonal cosine similarity
    Fn = torch.nn.functional.normalize(F, dim=1)
    cos = Fn @ Fn.T
    off = cos[~torch.eye(cos.shape[0], dtype=torch.bool)]
    print(f"FEATURE DIVERSITY [{tag}]")
    print(f"  feat std(mean over dims)={std:.5f}  "
          f"mean pairwise cos={float(off.mean()):.4f}  "
          f"(cos→1 ⇒ collapsed / identical features)")
    fire = getattr(getattr(model, 'shared_encoder', SimpleNamespace()),
                   'last_stem_fire_rate', float('nan'))
    if fire == fire:
        print(f"  last stem fire_rate={fire:.4f}")
    return F


def _raw_linear_overfit(mris: torch.Tensor, labels: torch.Tensor,
                        steps: int, lr: float, device: torch.device) -> float:
    """Can a linear classifier memorize downsampled raw voxels? (no SNN, no DS).

    If this PASSES while MRI+DS FAILS ⇒ bottleneck is SNN/DS stack, not the volumes.
    If this also FAILS ⇒ volumes as loaded are non-discriminative / near-identical.
    """
    # Pool to 8³ so the linear map is tiny and easy to memorize.
    x = torch.nn.functional.adaptive_avg_pool3d(mris, (8, 8, 8))
    x = x.reshape(x.shape[0], -1).to(device)
    y = labels.to(device)
    # z-score features for stable CE
    x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)

    clf = nn.Linear(x.shape[1], Config.NUM_CLASSES).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    print("-" * 72)
    print("RAW-VOXEL LINEAR OVERFIT (control — no SNN, no DS)")
    best = 0.0
    for t in range(steps):
        logits = clf(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        acc = (logits.argmax(1) == y).float().mean().item()
        best = max(best, acc)
        if t % 50 == 0 or t + 1 == steps:
            print(f"  raw step {t:4d}  loss={loss.item():.4f}  acc={acc:.3f}")
    print(f"  raw-linear best acc={best:.3f}  "
          f"({'PASS' if best >= 0.99 else 'FAIL'})")
    return best


def main():
    parser = argparse.ArgumentParser(description='MRI+DS 8-subject overfit probe')
    parser.add_argument('--n', type=int, default=8, help='Number of subjects')
    parser.add_argument('--steps', type=int, default=400, help='Optimization steps')
    parser.add_argument('--lr', type=float, default=3e-4, help='AdamW learning rate')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size (default=all)')
    parser.add_argument('--no-bn', action='store_true',
                        help='Replace SNN feat BatchNorm with Identity')
    parser.add_argument('--skip-raw', action='store_true',
                        help='Skip raw-voxel linear control overfit')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    _apply_probe_config(disable_bn=args.no_bn)
    seed = Config.SEED if args.seed is None else args.seed
    set_seed(seed)

    # Re-bind module-level cfg objects that already imported Config (class attrs).
    import DSModule as _dsm
    import SNNModule as _snn
    import main as _main
    _dsm.cfg = Config()
    _snn.cfg = Config()
    _main.cfg = Config()

    device = torch.device(Config.DEVICE)
    print("=" * 72)
    print("SANITY OVERFIT — MRI + Dempster–Shafer (memorization probe)")
    print("=" * 72)
    print(f"  ablation={Config.ABLATION}  evidence_scale={Config.MRI_EVIDENCE_SCALE}  "
          f"cap={Config.MRI_EVIDENCE_CAP}  blend={Config.DECISION_BLEND}")
    print(f"  n={args.n}  steps={args.steps}  lr={args.lr}  no_bn={args.no_bn}  device={device}")

    print("Loading OASIS-2 sequences...")
    all_sequences = Oasis2Dataset.load_all_sequences()
    idx = _pick_subjects(all_sequences, n=args.n, seed=seed)
    y = subject_labels(all_sequences)
    labels_sel = y[idx].tolist()
    print(f"  Selected subject indices: {idx.tolist()}")
    print(f"  Final-visit labels (0=CN,1=MCI,2=AD): {labels_sel}  "
          f"counts={np.bincount(y[idx], minlength=3).tolist()}")

    oasis = _make_tiny_dataset(all_sequences, idx)
    wrap = CrossSectionalWrapper(oasis, augment=False)
    assert len(wrap) == args.n, f"Expected {args.n} samples, got {len(wrap)}"

    mris_cpu, clin_cpu, labs_cpu = _load_all_samples(wrap)
    _report_input_diversity(mris_cpu, labs_cpu)
    maj_acc = float(np.bincount(labs_cpu.numpy(), minlength=3).max() / len(labs_cpu))

    raw_best = None
    if not args.skip_raw:
        raw_best = _raw_linear_overfit(
            mris_cpu, labs_cpu, steps=min(args.steps, 400),
            lr=1e-2, device=device)

    bs = min(args.batch_size, len(wrap))
    loader = DataLoader(wrap, batch_size=bs, shuffle=True,
                        num_workers=0, drop_last=False)

    model = DSSNMMF().to(device)
    if args.no_bn:
        _disable_feat_bn(model)
    assert model.use_mri and not model.use_clinical, (
        "Probe expects ABLATION='mri' (MRI on, clinical off)."
    )

    _report_feature_diversity(model, mris_cpu, device, tag='init')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    loss_fn = DSSNNLoss(class_weights=torch.ones(Config.NUM_CLASSES, device=device))
    loss_fn.set_epoch(10_000)  # skip DS/IGN warmup — full weights immediately

    print("-" * 72)
    print("MRI + DS OVERFIT")
    print(f"{'step':>6}  {'loss':>8}  {'acc':>6}  {'macroF1':>8}  "
          f"{'m_Θ':>6}  {'aux':>7}  {'ds':>7}  {'ce':>7}  fire")
    print("-" * 72)

    model.train()
    step = 0
    best_acc = 0.0
    history = []

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            mri = batch['mri'].to(device)
            clinical = batch['clinical'].to(device)
            labels = batch['label'].to(device)

            out = model(mri=mri, clinical=clinical)
            losses = loss_fn(out, labels)

            opt.zero_grad(set_to_none=True)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            with torch.no_grad():
                preds = out['pred']
                acc = (preds == labels).float().mean().item()
                f1 = float(f1_score(
                    labels.cpu().numpy(), preds.cpu().numpy(),
                    average='macro', zero_division=0,
                    labels=list(range(Config.NUM_CLASSES)),
                ))
                m_theta = out['m_fused'][:, FULL_FRAME].mean().item()
                fire = getattr(
                    getattr(model, 'shared_encoder', SimpleNamespace()),
                    'last_stem_fire_rate', float('nan'),
                )

            best_acc = max(best_acc, acc)
            history.append({'step': step, 'acc': acc, 'f1': f1, 'm_theta': m_theta})

            if step % 25 == 0 or step + 1 == args.steps:
                fire_s = f"{fire:.3f}" if fire == fire else "n/a"
                print(f"{step:6d}  {losses['total'].item():8.4f}  {acc:6.3f}  {f1:8.3f}  "
                      f"{m_theta:6.3f}  {losses['aux'].item():7.4f}  "
                      f"{losses['ds'].item():7.4f}  {losses['ce'].item():7.4f}  {fire_s}")
            step += 1

    # Final full-set eval (one pass, train mode off for BN running stats if BN kept)
    model.eval()
    all_preds, all_labels = [], []
    m_thetas = []
    with torch.no_grad():
        for batch in DataLoader(wrap, batch_size=bs, shuffle=False, num_workers=0):
            mri = batch['mri'].to(device)
            clinical = batch['clinical'].to(device)
            labels = batch['label'].to(device)
            out = model(mri=mri, clinical=clinical)
            all_preds.extend(out['pred'].cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            m_thetas.extend(out['m_fused'][:, FULL_FRAME].cpu().tolist())

    final_acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    final_f1 = float(f1_score(
        all_labels, all_preds, average='macro', zero_division=0,
        labels=list(range(Config.NUM_CLASSES)),
    ))
    pred_hist = np.bincount(all_preds, minlength=3).tolist()

    _report_feature_diversity(model, mris_cpu, device, tag='after-train')

    # Where did DS mass go?
    with torch.no_grad():
        out = model(mri=mris_cpu.to(device), clinical=clin_cpu.to(device))
        # singleton masses at indices 1,2,4
        sing = out['m_fused'][:, [1, 2, 4]].cpu()
        print(f"  mean singleton mass CN/MCI/AD={sing.mean(0).tolist()}")
        print(f"  aux_logits mean={out['aux_logits'].mean(0).cpu().tolist()}")

    print("=" * 72)
    print("RESULT")
    print(f"  Majority baseline acc              = {maj_acc:.3f}")
    if raw_best is not None:
        print(f"  Raw-voxel linear best acc          = {raw_best:.3f}")
    print(f"  Final MRI+DS eval  acc={final_acc:.3f}  macroF1={final_f1:.3f}  "
          f"mean m_Θ={float(np.mean(m_thetas)):.3f}")
    print(f"  Pred class counts CN/MCI/AD = {pred_hist}")
    print(f"  Best train-batch acc during run = {best_acc:.3f}")
    print("-" * 72)
    if abs(final_acc - maj_acc) < 0.02 and pred_hist[0] == args.n:
        print("  COLLAPSE — predicting majority (CN) for everyone.")
        print("  m_Θ low means DS is *confidently* wrong (mass on {CN}), not vacuous.")
    if final_acc >= 0.99 or best_acc >= 0.99:
        print("  PASS — MRI+DS can memorize these subjects.")
        print("  CV failure is generalization / recipe (BN, sampler, registration),")
        print("  not 'encoder cannot extract any signal'.")
    elif final_acc >= 0.75:
        print("  WEAK PASS — partial memorization. Try --no-bn and/or more --steps.")
    else:
        print("  FAIL — cannot memorize 8 subjects with MRI+DS.")
        if raw_best is not None and raw_best >= 0.99:
            print("  Control PASS (raw linear) ⇒ volumes differ; SNN/DS path is the bottleneck.")
            print("  Next: replace SNN with plain Conv3D (keep DS), or fix spike collapse.")
        elif raw_best is not None and raw_best < 0.75:
            print("  Control also FAIL ⇒ volumes as loaded are not class-separable")
            print("  (alignment / preproc / cache). Registration is the next real fix.")
        if not args.no_bn:
            print("  Also try:  python sanity_overfit.py --no-bn")
    print("=" * 72)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
