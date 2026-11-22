"""
Gradient-boosting (GB) baseline for the carbon-environment task, on the same
data, hold-out and CV folds as the AugerNet CNN.

    python gradient_boosting_classifier.py [config.yml]
        cv.yml           30 un-merged classes (merge_scheme: none)   
        cv_chemical.yml  13 merged classes    (merge_scheme: chemical)

Three models, one per input:
    gb_be        GB-E_b       z-normalised Delta E_b only
    gb_spec      GB-AES       fitted Auger spectrum (751 grid points)
    gb_spec_be   GB-AES+E_b   spectrum with z-normalised Delta E_b appended

Writes to <result_dir>/baseline_results/<merge_scheme>/:
    gb_metrics_per_fold.csv, gb_metrics_summary.csv, gb_results.pkl

No hyper-parameter is tuned on the hold-out or eval sets.
"""
import os
import sys
import types
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingClassifier

from augernet.config import load_config
from augernet import carbon_dataframe as cdf
from augernet.cnn_train_utils import classification_metrics
from augernet.backend_cnn import (load_data, cnn_fold_split, _dataset_params,
                                  _resolve_class_names, _present)
from augernet.train_driver import split_holdout

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else 'cv.yml')
out_dir = os.path.join(cfg.result_dir, 'baseline_results', cfg.merge_scheme)
os.makedirs(out_dir, exist_ok=True)
SEED = cfg.random_seed
SPLITS = ('val', 'holdout', 'eval')


# ── Models ────────────────────────────────────────────────────────────────
# (input features, estimator factory, OpenMP threads; None = all cores)
def gb(learning_rate, l2):
    # early_stopping off: its internal validation split is random by carbon,
    # not by molecule, so it would leak between molecules.
    return HistGradientBoostingClassifier(
        max_iter=100, learning_rate=learning_rate, l2_regularization=l2,
        early_stopping=False, random_state=SEED)


MODELS = {
    # On a single feature, classes with one or two training carbons make the
    # Newton leaf values diverge without strong L2; lr and l2 were the best of
    # a small grid judged on the validation folds only.  One thread: on one
    # feature the thread overhead makes multi-threading ~8x slower.
    'gb_be':      ('be',      lambda: gb(0.05, 10.0), 1),
    # Spectrum models: one fixed, untuned setting.
    'gb_spec':    ('spec',    lambda: gb(0.1, 1.0),   None),
    'gb_spec_be': ('spec+be', lambda: gb(0.1, 1.0),   None),
}

FEATURES = {
    'be':      lambda s: s['be_z'],
    'spec':    lambda s: s['spec'],
    'spec+be': lambda s: np.hstack([s['spec'], s['be_z']]),
}


# ── Data: same frames and spectra as backend_cnn ──────────────────────────
data = split_holdout(load_data(cfg), cfg.model)
class_names = _resolve_class_names(cfg.merge_scheme, _present(data))
df = data['cnn_df']
frames = {'calc':    df[df['source'] == 'calc'].reset_index(drop=True),
          'holdout': data['test_df'].reset_index(drop=True),
          'eval':    df[df['source'] == 'eval'].reset_index(drop=True)}
calc_df = frames['calc']

# Raw Delta E_b out of the dataset; it is z-normalised per fold below.
ds_params = {**_dataset_params(cfg), 'include_augmentation': 'none'}
RAW = {'be_mu': 0.0, 'be_std': 1.0}


def to_arrays(frame):
    """(spectra (n, 751), raw Delta E_b (n,), labels (n,), molecule names)."""
    ds = cdf.CarbonDataset(frame, norm_stats=RAW, **ds_params)
    spec, dbe, y = zip(*(ds[i] for i in range(len(ds))))
    return (np.stack([s.numpy() for s in spec]),
            np.asarray(dbe, dtype=np.float64),
            np.asarray([int(l) for l in y], dtype=np.int64),
            frame['mol_name'].to_numpy())


arrays = {name: to_arrays(frame) for name, frame in frames.items()}
spec_c, be_c, y_c, mol_c = arrays['calc']


def fold_splits(fold):
    """Train / val / hold-out / eval arrays for one CV fold."""
    train_mols, val_mols = cnn_fold_split(
        list(dict.fromkeys(calc_df['mol_name'])), calc_df, fold, cfg.n_folds,
        cfg.split_method, cfg.random_seed, butina_cutoff=cfg.butina_cutoff,
        verbose=(fold == 1))
    tr = calc_df['mol_name'].isin(set(train_mols)).to_numpy()
    va = calc_df['mol_name'].isin(set(val_mols)).to_numpy()

    # Delta E_b z-normalised with the training split's mean and std (ddof=1),
    # as CarbonDataset does for the CNN.
    be_train = calc_df['delta_be'][tr]
    mu, sd = be_train.mean(), be_train.std() + 1e-8

    # The CNN's inverse-square-root-frequency class weights, from the training
    # split, passed to GB as per-carbon sample weights.
    w, _ = cdf.CarbonDataset.get_class_weights_and_counts(
        types.SimpleNamespace(df=calc_df[tr]), len(class_names))
    w = w.numpy()

    parts = {'train': (spec_c[tr], be_c[tr], y_c[tr], mol_c[tr]),
             'val':   (spec_c[va], be_c[va], y_c[va], mol_c[va]),
             'holdout': arrays['holdout'], 'eval': arrays['eval']}
    split = {name: {'spec': spec, 'be_z': ((be - mu) / sd)[:, None],
                    'y': y, 'mols': mols}
             for name, (spec, be, y, mols) in parts.items()}
    split['train']['sample_weight'] = w[split['train']['y']]
    print(f'fold {fold}: train {tr.sum()} C, val {va.sum()} C, '
          f'Delta E_b mu={mu:.6f} std={sd:.6f}')
    return split


# ── Train and evaluate ────────────────────────────────────────────────────
# results[model][fold][split] = {'metrics', 'y_true', 'y_pred'}
results = {name: {} for name in MODELS}
rows = []
for fold in range(1, cfg.n_folds + 1):
    s = fold_splits(fold)
    for name, (feat, make, threads) in MODELS.items():
        X = FEATURES[feat]
        model = make()
        with threadpool_limits(limits=threads, user_api='openmp'):
            model.fit(X(s['train']), s['train']['y'],
                      sample_weight=s['train']['sample_weight'])
        results[name][fold] = {}
        for sp in SPLITS:
            y_true, y_pred = s[sp]['y'], model.predict(X(s[sp]))
            m = classification_metrics(s[sp]['mols'], y_true, y_pred)
            results[name][fold][sp] = {'metrics': m, 'y_true': y_true,
                                       'y_pred': y_pred}
            rows.append({'model': name, 'fold': fold, 'split': sp,
                         'n': len(y_true), **m})
        print(f'  {name:<11s} ' + '  '.join(
            f"{sp} acc {results[name][fold][sp]['metrics']['accuracy']:.4f}"
            for sp in SPLITS), flush=True)


# ── Fold-averaged metrics (std ddof=1, as the CNN CV summary) ─────────────
per_fold = pd.DataFrame(rows)
metric_cols = [c for c in per_fold.columns
               if c not in ('model', 'fold', 'split', 'n')]
summary = per_fold.groupby(['model', 'split'], sort=False)[metric_cols].agg(
    ['mean', 'std'])
summary.columns = [f'{stat}_{metric}' for metric, stat in summary.columns]
summary = summary.reset_index()

show = ('accuracy', 'recall_macro', 'f1_macro')
print(f"\n{'model':<11s} {'split':<8s} " + ' '.join(f'{k:>17s}' for k in show))
for _, r in summary.iterrows():
    print(f"{r['model']:<11s} {r['split']:<8s} " + ' '.join(
        f"{r[f'mean_{k}']:>8.4f} +/-{r[f'std_{k}']:.4f}" for k in show))

per_fold.to_csv(os.path.join(out_dir, 'gb_metrics_per_fold.csv'), index=False)
summary.to_csv(os.path.join(out_dir, 'gb_metrics_summary.csv'), index=False)
with open(os.path.join(out_dir, 'gb_results.pkl'), 'wb') as fh:
    pickle.dump({'results': results, 'class_names': class_names}, fh)
print(f'\nwritten to {out_dir}')
