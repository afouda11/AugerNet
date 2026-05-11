import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# need to activate augernet conda env for script to work
# requires augernet class merging and spectral fitting routines
from augernet import DATA_PROCESSED_DIR
import augernet.class_merging as cm
import augernet.spec_utils as su 

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

if __name__ == "__main__":
    main()