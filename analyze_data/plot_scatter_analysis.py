"""
Scatter-plot analysis of Carbon Auger spectra.
===============================================

Generates publication-quality figures:

1. **lineshape_metrics**     — Distribution bar + 2×2 scatter (vs CEBE) + legend
2. **lineshape_eneg**        — Same layout but x-axis = electronegativity score
3. **pca_scatter**           — PCA (PC1 vs PC2) + LDA (LD1 vs LD2) + legend
4. **cebe_vs_eneg**          — CEBE vs electronegativity score, per environment

All figures support an optional ``--merge-scheme`` argument to use
chemically merged classes (e.g. ``chemical``, ``conservative``, etc.).

Usage
-----
    python analyze_data/plot_scatter_analysis.py                        # 36 classes
    python analyze_data/plot_scatter_analysis.py --merge-scheme chemical   # 17 merged
    python analyze_data/plot_scatter_analysis.py --merge-scheme practical  # 11 merged
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# ── Resolve project root & imports ────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from augernet import carbon_dataframe as cdf
from augernet.carbon_environment import IDX_TO_CARBON_ENV
from augernet.class_merging import (
    apply_label_merging,
    get_available_schemes,
)
from augernet.eneg_diff import get_e_neg_score
from augernet.env_vis import (
    compute_spectral_scalars,
    get_environment_colors,
    get_environment_markers,
    get_group_ordered_envs,
    get_ordered_unique_envs,
)

# ── Style ─────────────────────────────────────────────────────────────────────

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

OUTPUT_DIR = str(SCRIPT_DIR / 'pngs_scatter')

# ── Data helpers ──────────────────────────────────────────────────────────────


def _add_eneg_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-atom electronegativity score and store as ``e_neg_score``."""
    if 'e_neg_score' in df.columns:
        return df

    print("  Computing environment electronegativity scores...")
    scores = np.full(len(df), np.nan)
    for smiles, grp in df.groupby('smiles'):
        try:
            mol_scores = get_e_neg_score(smiles)
        except Exception as exc:
            print(f"    ⚠ Could not compute e-neg for {smiles}: {exc}")
            continue
        for row_idx, (_, row) in zip(grp.index, grp.iterrows()):
            atom_idx = int(row['atom_idx'])
            if atom_idx < len(mol_scores):
                scores[row_idx] = mol_scores[atom_idx]
    df['e_neg_score'] = scores
    valid = np.isfinite(scores).sum()
    print(f"  ✓ e_neg_score: {valid}/{len(df)} atoms with valid values")
    return df


def _ensure_mol_cebe(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``mol_cebe`` column (= true_cebe or atomic_be - delta_be)."""
    if 'mol_cebe' in df.columns:
        return df
    if 'true_cebe' in df.columns:
        df['mol_cebe'] = df['true_cebe']
    elif 'atomic_be' in df.columns and 'delta_be' in df.columns:
        df['mol_cebe'] = df['atomic_be'] - df['delta_be']
    else:
        raise ValueError("Cannot compute CEBE — need 'true_cebe' or "
                         "'atomic_be' + 'delta_be' columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Figure: Lineshape Metrics  (distribution + 2×2 scatter vs CEBE or eneg)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_lineshape(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    filename_stem: str,
    merged_scheme: str | None,
    output_dir: str,
    axis_font: int = 60,
):
    """
    Generic lineshape-metrics figure.

    Layout: [distribution | spacer | 2×2 scatter | legend]

    *x_col* / *x_label* control what goes on the scatter x-axis
    (``mol_cebe`` / ``e_neg_score``).
    """
    ordered_envs, env_to_label = get_ordered_unique_envs(df, merged_scheme)
    color_map = get_environment_colors(df, merged_scheme)
    marker_map = get_environment_markers(df, merged_scheme)
    active = set(int(e) for e in df['carbon_env_label'].unique() if int(e) >= 0)
    legend_envs = get_group_ordered_envs(active, merged_scheme)

    metric_info = [
        ('spectrum_energy',   'Centroid (eV)'),
        ('spectrum_width',    'Width (eV)'),
        ('spectrum_skewness', 'Skewness'),
        ('spectrum_entropy',  'Entropy'),
    ]

    fig = plt.figure(figsize=(60, 30))
    gs_outer = GridSpec(
        1, 4, width_ratios=[0.5, 0.10, 2.0, 0.08],
        wspace=0.08, figure=fig,
        left=0.05, right=0.98, top=0.96, bottom=0.06,
    )
    gs_dist = gs_outer[0, 0].subgridspec(1, 1)
    gs_scatter = gs_outer[0, 2].subgridspec(2, 2, wspace=0.15, hspace=0.02)
    gs_leg = gs_outer[0, 3].subgridspec(1, 1, wspace=0.01)

    # ── Distribution bar (column 0) ──────────────────────────────────────
    ax_dist = fig.add_subplot(gs_dist[0, 0])
    env_counts = df['carbon_env_label'].value_counts().sort_values()
    env_counts = env_counts[env_counts.index >= 0]
    bar_indices = env_counts.index.tolist()
    bar_counts = env_counts.values.tolist()
    global_max = max(bar_counts)

    for i, (idx, count) in enumerate(zip(bar_indices, bar_counts)):
        n_seg = 100
        xs = np.linspace(0, count, n_seg)
        cs = plt.cm.viridis(xs / global_max)
        for j in range(n_seg - 1):
            rect = plt.Rectangle(
                (xs[j], i - 0.4), xs[j + 1] - xs[j], 0.8,
                facecolor=cs[j], edgecolor='none', zorder=1,
            )
            ax_dist.add_patch(rect)

    ax_dist.set_yticks(range(len(bar_indices)))
    ax_dist.set_yticklabels(
        [env_to_label.get(i, str(i)) for i in bar_indices],
        fontsize=axis_font,
    )
    ax_dist.set_xlabel('Count', fontsize=axis_font, fontweight='bold')
    ax_dist.set_xlim(0, global_max * 1.12)
    ax_dist.set_ylim(-0.5, len(bar_indices) - 0.5)
    ax_dist.grid(True, alpha=0.3, axis='x', zorder=0)
    ax_dist.tick_params(axis='x', labelsize=axis_font)
    ax_dist.tick_params(axis='y', labelsize=axis_font)
    for i, count in enumerate(bar_counts):
        ax_dist.text(
            count, i, f' {count}', va='center',
            fontsize=axis_font, fontweight='bold', zorder=3,
        )

    # ── 2×2 scatter panels ───────────────────────────────────────────────
    axes = [fig.add_subplot(gs_scatter[r, c]) for r in range(2) for c in range(2)]

    for ax, (col, ylabel) in zip(axes, metric_info):
        for env_idx in ordered_envs:
            mask = df['carbon_env_label'] == env_idx
            if not mask.any():
                continue
            ax.scatter(
                df.loc[mask, x_col],
                df.loc[mask, col],
                c=[color_map[env_idx]],
                marker=marker_map[env_idx],
                alpha=0.4, s=200, edgecolors='none',
                label=env_to_label[env_idx],
            )
        ax.set_ylabel(ylabel, fontsize=axis_font, fontweight='bold')
        ax.grid(True, alpha=0.3, linewidth=1.0)
        ax.tick_params(axis='both', labelsize=axis_font)

    for ax in axes[2:]:
        ax.set_xlabel(x_label, fontsize=axis_font, fontweight='bold')
    for ax in axes[:2]:
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelbottom=False)

    # ── Legend ────────────────────────────────────────────────────────────
    ax_leg = fig.add_subplot(gs_leg[0, 0])
    ax_leg.axis('off')
    handles = [
        plt.scatter(
            [], [], c=[color_map[e]], marker=marker_map[e],
            s=120, edgecolors='none', alpha=0.4,
        )
        for e in legend_envs if e in color_map
    ]
    labels = [env_to_label[e] for e in legend_envs if e in env_to_label]
    ax_leg.legend(
        handles, labels, loc='center left',
        bbox_to_anchor=(0, 0.5),
        fontsize=axis_font, ncol=1, framealpha=0.95, edgecolor='black',
        markerscale=3,
    )

    # ── Save ─────────────────────────────────────────────────────────────
    for ext in ('png', 'pdf'):
        path = os.path.join(output_dir, f'{filename_stem}.{ext}')
        fig.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f"  ✓ Saved: {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Figure: PCA + LDA scatter
# ══════════════════════════════════════════════════════════════════════════════

def plot_pca_lda(
    df: pd.DataFrame,
    merged_scheme: str | None,
    output_dir: str,
    axis_font: int = 38,
):
    """PCA (PC1 vs PC2) + LDA (LD1 vs LD2) side-by-side with legend."""
    print("\nGenerating pca_scatter (PCA + LDA)...")

    ordered_envs, env_to_label = get_ordered_unique_envs(df, merged_scheme)
    color_map = get_environment_colors(df, merged_scheme)
    marker_map = get_environment_markers(df, merged_scheme)
    active = set(int(e) for e in df['carbon_env_label'].unique() if int(e) >= 0)
    legend_envs = get_group_ordered_envs(active, merged_scheme)

    # Build spectrum matrix
    spectra = np.stack(df['spectrum_intensity_only'].values)
    scaler = StandardScaler()
    spectra_scaled = scaler.fit_transform(spectra)
    env_labels = df['carbon_env_label'].values

    # PCA
    n_comp = min(50, *spectra.shape)
    pca = PCA(n_components=n_comp)
    pca_scores = pca.fit_transform(spectra_scaled)
    vr = pca.explained_variance_ratio_

    # LDA
    n_lda = min(2, len(ordered_envs) - 1)
    lda = LinearDiscriminantAnalysis(n_components=n_lda)
    lda_scores = lda.fit_transform(spectra_scaled, env_labels)
    lda_vr = lda.explained_variance_ratio_
    print(f"  PCA: PC1={vr[0]*100:.1f}%, PC2={vr[1]*100:.1f}%")
    if n_lda >= 2:
        print(f"  LDA: LD1={lda_vr[0]*100:.1f}%, LD2={lda_vr[1]*100:.1f}%")

    fig = plt.figure(figsize=(36, 20))
    gs_main = GridSpec(
        1, 2, width_ratios=[2.0, 0.25], wspace=0.02,
        left=0.06, right=0.98, top=0.96, bottom=0.08,
    )
    gs_scatter = gs_main[0, 0].subgridspec(1, 2, wspace=0.15)
    gs_leg = gs_main[0, 1].subgridspec(1, 1)

    # ── PCA ───────────────────────────────────────────────────────────────
    ax_pca = fig.add_subplot(gs_scatter[0, 0])
    for env_idx in ordered_envs:
        mask = env_labels == env_idx
        if not mask.any():
            continue
        ax_pca.scatter(
            pca_scores[mask, 0], pca_scores[mask, 1],
            c=[color_map[env_idx]], marker=marker_map[env_idx],
            alpha=0.4, s=160, edgecolors='none',
        )
    ax_pca.set_xlabel(f'PC1 ({vr[0]*100:.1f}%)', fontsize=axis_font, fontweight='bold')
    ax_pca.set_ylabel(f'PC2 ({vr[1]*100:.1f}%)', fontsize=axis_font, fontweight='bold')
    ax_pca.grid(True, alpha=0.3)
    ax_pca.tick_params(labelsize=axis_font)

    # ── LDA ───────────────────────────────────────────────────────────────
    ax_lda = fig.add_subplot(gs_scatter[0, 1])
    for env_idx in ordered_envs:
        mask = env_labels == env_idx
        if not mask.any():
            continue
        if n_lda >= 2:
            ax_lda.scatter(
                lda_scores[mask, 0], lda_scores[mask, 1],
                c=[color_map[env_idx]], marker=marker_map[env_idx],
                alpha=0.4, s=160, edgecolors='none',
            )
        else:
            ax_lda.scatter(
                lda_scores[mask, 0], np.zeros(mask.sum()),
                c=[color_map[env_idx]], marker=marker_map[env_idx],
                alpha=0.4, s=160, edgecolors='none',
            )
    if n_lda >= 2:
        ax_lda.set_xlabel(f'LD1 ({lda_vr[0]*100:.1f}%)', fontsize=axis_font, fontweight='bold')
        ax_lda.set_ylabel(f'LD2 ({lda_vr[1]*100:.1f}%)', fontsize=axis_font, fontweight='bold')
    else:
        ax_lda.set_xlabel(f'LD1 ({lda_vr[0]*100:.1f}%)', fontsize=axis_font, fontweight='bold')
    ax_lda.grid(True, alpha=0.3)
    ax_lda.tick_params(labelsize=axis_font)

    # ── Legend ────────────────────────────────────────────────────────────
    ax_l = fig.add_subplot(gs_leg[0, 0])
    ax_l.axis('off')
    handles = [
        plt.scatter(
            [], [], c=[color_map[e]], marker=marker_map[e],
            s=140, edgecolors='none', alpha=0.5,
        )
        for e in legend_envs if e in color_map
    ]
    labels = [env_to_label[e] for e in legend_envs if e in env_to_label]
    ax_l.legend(
        handles, labels, loc='center left', bbox_to_anchor=(0, 0.5),
        fontsize=axis_font - 8, ncol=1, framealpha=0.95, edgecolor='black',
        markerscale=2,
    )

    stem = 'pca_scatter'
    if merged_scheme and merged_scheme != 'none':
        stem += f'_{merged_scheme}'
    for ext in ('png', 'pdf'):
        path = os.path.join(output_dir, f'{stem}.{ext}')
        fig.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f"  ✓ Saved: {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Figure: CEBE vs Electronegativity Score
# ══════════════════════════════════════════════════════════════════════════════

def plot_cebe_vs_eneg(
    df: pd.DataFrame,
    merged_scheme: str | None,
    output_dir: str,
    axis_font: int = 38,
):
    """
    CEBE vs electronegativity score, coloured by environment.

    Uses the CNN DataFrame directly (no GNN data file needed).
    """
    print("\nGenerating cebe_vs_eneg scatter...")

    ordered_envs, env_to_label = get_ordered_unique_envs(df, merged_scheme)
    color_map = get_environment_colors(df, merged_scheme)
    marker_map = get_environment_markers(df, merged_scheme)
    active = set(int(e) for e in df['carbon_env_label'].unique() if int(e) >= 0)
    legend_envs = get_group_ordered_envs(active, merged_scheme)

    fig = plt.figure(figsize=(30, 20))
    gs_main = GridSpec(
        1, 2, width_ratios=[2.0, 0.30], wspace=0.02,
        left=0.08, right=0.98, top=0.96, bottom=0.08,
    )
    ax = fig.add_subplot(gs_main[0, 0])

    for env_idx in ordered_envs:
        mask = df['carbon_env_label'] == env_idx
        if not mask.any():
            continue
        ax.scatter(
            df.loc[mask, 'mol_cebe'],
            df.loc[mask, 'e_neg_score'],
            c=[color_map[env_idx]],
            marker=marker_map[env_idx],
            alpha=0.45, s=200, edgecolors='none',
            label=env_to_label[env_idx],
        )

    ax.set_xlabel('CEBE (eV)', fontsize=axis_font, fontweight='bold')
    ax.set_ylabel('Electronegativity Score', fontsize=axis_font, fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=1.0)
    ax.tick_params(axis='both', labelsize=axis_font)

    # Legend
    ax_l = fig.add_subplot(gs_main[0, 1])
    ax_l.axis('off')
    handles = [
        plt.scatter(
            [], [], c=[color_map[e]], marker=marker_map[e],
            s=160, edgecolors='none', alpha=0.5,
        )
        for e in legend_envs if e in color_map
    ]
    labels = [env_to_label[e] for e in legend_envs if e in env_to_label]
    ax_l.legend(
        handles, labels, loc='center left', bbox_to_anchor=(0, 0.5),
        fontsize=axis_font - 8, ncol=1, framealpha=0.95, edgecolor='black',
        markerscale=2,
    )

    stem = 'cebe_vs_eneg'
    if merged_scheme and merged_scheme != 'none':
        stem += f'_{merged_scheme}'
    for ext in ('png', 'pdf'):
        path = os.path.join(output_dir, f'{stem}.{ext}')
        fig.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f"  ✓ Saved: {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Scatter-plot analysis of carbon Auger spectra.",
    )
    parser.add_argument(
        '--merge-scheme', type=str, default='none',
        choices=get_available_schemes(),
        help="Class-merging scheme (default: none = all 36 classes).",
    )
    parser.add_argument(
        '--output-dir', type=str, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    merge = args.merge_scheme if args.merge_scheme != 'none' else None
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    data_path = str(PROJECT_ROOT / 'data' / 'processed' / 'cnn_auger_calc.pkl')
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        sys.exit(1)

    print(f"Loading carbon dataframe from: {data_path}")
    df = cdf.load_carbon_dataframe(data_path)
    print(f"✓ Loaded {len(df)} carbon atoms")

    # ── Optional label merging ────────────────────────────────────────────
    if merge:
        df = apply_label_merging(df, merge)
        # Drop atoms that couldn't be mapped
        n_before = len(df)
        df = df[df['carbon_env_label'] >= 0].reset_index(drop=True)
        n_after = len(df)
        if n_before != n_after:
            print(f"  Dropped {n_before - n_after} unmapped atoms")

    # ── Feature engineering ───────────────────────────────────────────────
    _ensure_mol_cebe(df)
    _add_eneg_scores(df)

    print("\nComputing spectral scalar features...")
    scalars = compute_spectral_scalars(df)
    df['spectrum_energy'] = scalars['centroid']
    df['spectrum_width'] = scalars['width']
    df['spectrum_skewness'] = scalars['skewness']
    df['spectrum_entropy'] = scalars['entropy']

    suffix = f'_{merge}' if merge else ''

    # ── Figure 1: Lineshape metrics vs CEBE ───────────────────────────────
    print("\nGenerating lineshape_metrics (vs CEBE)...")
    _plot_lineshape(
        df, x_col='mol_cebe', x_label='CEBE (eV)',
        filename_stem=f'lineshape_metrics{suffix}',
        merged_scheme=merge, output_dir=out,
    )

    # ── Figure 2: Lineshape metrics vs Electronegativity ──────────────────
    print("\nGenerating lineshape_eneg (vs Electronegativity)...")
    df_valid = df.dropna(subset=['e_neg_score']).reset_index(drop=True)
    if len(df_valid) > 0:
        _plot_lineshape(
            df_valid, x_col='e_neg_score',
            x_label='Env. Electronegativity Score',
            filename_stem=f'lineshape_eneg{suffix}',
            merged_scheme=merge, output_dir=out,
        )
    else:
        print("  ⚠ No valid electronegativity scores — skipping")

    # ── Figure 3: PCA + LDA ───────────────────────────────────────────────
    plot_pca_lda(df, merged_scheme=merge, output_dir=out)

    # ── Figure 4: CEBE vs electronegativity ───────────────────────────────
    if len(df_valid) > 0:
        plot_cebe_vs_eneg(df_valid, merged_scheme=merge, output_dir=out)

    print("\n" + "=" * 80)
    scheme_tag = merge or '36-class (none)'
    print(f"✓ Scatter analysis plots complete!  [scheme: {scheme_tag}]")
    print(f"  Output: {out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
