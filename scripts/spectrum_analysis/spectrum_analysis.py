import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# need to activate augernet conda env for script to work
# requires augernet class merging and spectral fitting routines
from augernet import DATA_PROCESSED_DIR
import augernet.class_merging as cm
import augernet.spec_utils as su 
import env_vis

CALC_PKL = Path(DATA_PROCESSED_DIR) / "cnn_auger_calc.pkl"
EVAL_PKL = Path(DATA_PROCESSED_DIR) / "cnn_auger_eval.pkl"

SCRIPT_DIR = Path(__file__).parent

SPEC_OUTPUT_DIR = SCRIPT_DIR / "pngs_spectra"
SPEC_OUTPUT_DIR.mkdir(exist_ok=True)

# -- energy grid & broadening -----------------------------------------
E_MIN, E_MAX, N_PTS = 200.0, 275.0, 1501
FWHM = 1.6        # Gaussian broadening FWHM (eV) applied to stick spectra
E_GRID = np.linspace(E_MIN, E_MAX, N_PTS)
NORM_I = True

def fit_spectra(df: pd.DataFrame):

    all_spec = []   

    for i in range(len(df)):
        row = df.iloc[i]
        se = np.asarray(row['sing_stick_energies'], dtype=np.float64)
        si = np.asarray(row['sing_stick_intensities'], dtype=np.float64)
        te = np.asarray(row['trip_stick_energies'], dtype=np.float64)
        ti = np.asarray(row['trip_stick_intensities'], dtype=np.float64)

        #for now concatenate
        energies = np.concatenate([se, te])
        intensities = np.concatenate([si, ti])

        # Fit the concatenated spectrum
        _, intensity_grid = su.fit_spectrum_to_grid(energies, intensities, FWHM, E_MIN, E_MAX, N_PTS, normalize=NORM_I)

        all_spec.append(intensity_grid)

    #Store back in DataFrame
    df['fitted_intensity'] = all_spec

def plot_mean_env_spectra(df: pd.DataFrame, suffix: str):

    env_counts = df['carbon_env_label'].value_counts()
    #print(env_counts)
    ordered_envs = env_counts.index.tolist()
    #print(ordered_envs)
    ### SINGLE PANEL ###
    fig, ax = plt.subplots(figsize=(14, 8))
    for env in ordered_envs:
        subset = df[df['carbon_env_label'] == env]
        stacked_spectra = np.stack(subset['fitted_intensity'].values)
        mean_spectrum = stacked_spectra.mean(axis=0)
        ax.plot(E_GRID, mean_spectrum, color='k', linewidth=1.5, label=env)

    ax.set_xlabel("Kinetic Energy (eV)")
    ax.set_ylabel("Mean Intensity (a.u.)")
    ax.set_title(f"Environment Class Averaged Spectra")
    ax.set_xlim(E_MIN, E_MAX)
    ax.legend(fontsize=7, ncol=3, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(SPEC_OUTPUT_DIR / f"single_panel_mean_spectra{suffix}.png", dpi=300, bbox_inches="tight")

    ### MULTI PANEL ###
    n_envs = len(env_counts)
    n_cols = 4
    n_rows = int(np.ceil(n_envs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3.2 * n_rows),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, env in enumerate(ordered_envs):
        ax = axes_flat[i]
        subset = df[df['carbon_env_label'] == env]
        stacked_spectra = np.stack(subset['fitted_intensity'].values)
        mean_spectrum = stacked_spectra.mean(axis=0)
        std_spec = stacked_spectra.std(axis=0)

        ax.plot(E_GRID, mean_spectrum, color='k', linewidth=1.8)
        ax.fill_between(E_GRID, mean_spectrum - std_spec, mean_spectrum + std_spec,
                        color='k', alpha=0.20)
        n_env = len(subset)
        title = env + f"  (n={n_env})"
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        if i % n_cols == 0:
            ax.set_ylabel("Mean Intensity (a.u.)", fontsize=8)
        if i >= n_envs - n_cols:
            ax.set_xlabel("Kinetic Energy (eV)", fontsize=8)

    fig.suptitle(f"Environment Class Averaged Spectra with +/- STD",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(SPEC_OUTPUT_DIR / f"multi_panel_mean_spectra{suffix}.png", dpi=300, bbox_inches="tight")

def plot_pca_lda(
        df: pd.DataFrame,
        merged_scheme: str | None,
        output_dir: str,
        axis_font: int = 38,
    ):

    env_counts = df['carbon_env_label'].value_counts()
    #print(env_counts)
    ordered_envs = env_counts.index.tolist()

    color_map = {env: env_vis._name_to_color(env, merged_scheme) for env in ordered_envs}
    legend_envs = env_vis.get_group_ordered_envs_str(set(ordered_envs), merged_scheme)

    # Build marker_map so each env within a hue family gets a DIFFERENT shape.
    # Iterate in legend order (group-sorted) so cycling is stable per family.
    MARKER_CYCLE = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', 'p', 'h', '*', 'd', 'H']
    marker_map: dict = {}
    family_counter: dict = {}
    for env in legend_envs:
        family = env_vis.get_env_family(env, merged_scheme)
        i = family_counter.get(family, 0)
        marker_map[env] = MARKER_CYCLE[i % len(MARKER_CYCLE)]
        family_counter[family] = i + 1
    # Fallback for any env not surfaced by get_group_ordered_envs_str
    for env in ordered_envs:
        if env not in marker_map:
            marker_map[env] = env_vis._name_to_marker(env, merged_scheme)

    #use sklearn to get spectrum matrix
    spectra = np.stack(df['fitted_intensity'].values)
    #performs mean and std normalization
    scaler = StandardScaler()
    spectra_scaled = scaler.fit_transform(spectra) # fitted scalars are stored internally, for use again
    env_labels = df['carbon_env_label'].values

    env_label_to_idx = {label: idx for idx, label in enumerate(ordered_envs)}
    env_labels_numeric = np.array([env_label_to_idx[label] for label in env_labels])

    # PCA
    n_comp = min(50, *spectra.shape)
    pca = PCA(n_components=n_comp)
    pca_scores = pca.fit_transform(spectra_scaled)
    vr = pca.explained_variance_ratio_

    # LDA
    #max (num_classes -1)
    n_lda = min(2, len(ordered_envs) - 1)
    lda = LinearDiscriminantAnalysis(n_components=n_lda)
    # LDA supervised, uses labels
    lda_scores = lda.fit_transform(spectra_scaled, env_labels)
    lda_vr = lda.explained_variance_ratio_

    fig = plt.figure(figsize=(36, 20))
    gs_main = GridSpec(
            1, 2, width_ratios=[2.0, 0.25], wspace=0.02,
            left=0.06, right=0.98, top=0.96, bottom=0.08,
    )
    gs_scatter = gs_main[0, 0].subgridspec(1, 2, wspace=0.15)
    gs_leg = gs_main[0, 1].subgridspec(1, 1)  

    # -- PCA --
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

    # -- LDA --
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
    ax_lda.set_ylim(-9.8, 5.2)
    # -- Legend -- grouped by chemical family with section headers --
    ax_l = fig.add_subplot(gs_leg[0, 0])
    ax_l.axis('off')

    # Build group membership lookup so we can insert headers at group boundaries.
    # Map each env name -> its hue family group (works for all schemes)
    env_to_group = {e: env_vis.get_env_family(e, merged_scheme) for e in legend_envs}

    handles, labels = [], []
    current_group = None
    blank = plt.scatter([], [], s=0, alpha=0)   # invisible placeholder

    for e in legend_envs:
        if e not in color_map:
            continue
        grp = env_to_group.get(e, '')
        if grp != current_group:
            current_group = grp
        handles.append(
            plt.scatter([], [], c=[color_map[e]], marker=marker_map[e],
                        s=140, edgecolors='none', alpha=0.8)
        )
        count = int(env_counts.get(e, 0))
        labels.append(f'  {env_vis.format_env_label(e)} ({count})')

    leg = ax_l.legend(
        handles, labels, loc='center left', bbox_to_anchor=(0, 0.5),
        fontsize=axis_font - 10, ncol=1, framealpha=0.95, edgecolor='black',
        markerscale=2, handlelength=1.5,
    )

    stem = 'pca_scatter'
    if merged_scheme and merged_scheme != 'none':
        stem += f'_{merged_scheme}'
    for ext in ('png', 'pdf'):
        path = Path(output_dir) / f'{stem}.{ext}'
        fig.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Carbon environment and spectrum analysis script"
        )
    parser.add_argument(
        '--merge-scheme', type=str, default='none',
        # List of available merging schemes, currently only none or chemical
        choices=cm.get_available_schemes(),
        help="Class-merging scheme (default: none = all 36 classes).",
    )
    args = parser.parse_args()

    merge = args.merge_scheme 
    suffix = f"_{merge}"

    calc_df = pd.read_pickle(CALC_PKL)
    eval_df = pd.read_pickle(EVAL_PKL)

    print(f"\nLoading data with merging scheme: {merge}")
    if merge != 'none':
        print("Class merging calc data")
        calc_df = cm.apply_label_merging(calc_df, merge)
        print("Class merging eval data")
        eval_df = cm.apply_label_merging(eval_df, merge)

    print(f"Fitting calc spectra with: FWHM={FWHM} eV, E_MIN={E_MIN} eV, E_MAX={E_MAX} eV, N_PTS={N_PTS}, normalize={NORM_I}")
    fit_spectra(calc_df)
    print(f"Fitting eval spectra with: FWHM={FWHM} eV, E_MIN={E_MIN} eV, E_MAX={E_MAX} eV, N_PTS={N_PTS}, normalize={NORM_I}")
    fit_spectra(eval_df)

    plot_mean_env_spectra(calc_df, suffix)

    plot_pca_lda(calc_df, merge, str(SPEC_OUTPUT_DIR))

if __name__ == "__main__":
    main()