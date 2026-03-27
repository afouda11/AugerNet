"""
Analysis and visualization of Carbon DataFrame features.

Generates publication-quality figures:
1. carbon_environment_distribution.png - Carbon environment distribution by class
2. spectrum_centroid.png - Spectrum centroid vs Delta BE (1x2 panel)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from augernet import carbon_dataframe as cdf
from augernet.carbon_environment import (
    IDX_TO_CARBON_ENV,
)


def format_env_label(env_name: str) -> str:
    """Format environment name: remove 'C_' prefix and replace underscores with spaces."""
    return env_name.replace('C_', '').replace('_', ' ')


def get_environment_colors_old(df: pd.DataFrame) -> Dict[int, tuple]:
    """
    BACKUP: Original colour map using tab20/tab20b colourmap.
    Create a colour map for environments using a perceptually uniform colourmap.
    Each active environment gets a distinct colour.
    """
    unique_envs = sorted(df['carbon_env_label'].unique())
    n = len(unique_envs)
    cmap = plt.cm.get_cmap('tab20') if n <= 20 else plt.cm.get_cmap('tab20b')
    colors = cmap(np.linspace(0, 1, max(n, 2)))
    return {env_idx: colors[i] for i, env_idx in enumerate(unique_envs)}


# ---------------------------------------------------------------------------
# Semantic colour + marker scheme (grouped by heteroatom neighbour)
# ---------------------------------------------------------------------------
# Each group gets a hue family; lightness varies within the group.
# Marker shapes give a second visual channel (colorblind-friendly).

# Group definitions: group_name → (list of env names, hue-family base HSL, marker)
#   Hue families chosen to be distinguishable under deuteranopia/protanopia:
#     Carbonyl/C=O  → purples/violets
#     C–O single    → blues
#     C–N           → greens/teals
#     Halogen (F)   → oranges/reds
#     Aromatic      → warm pinks/magentas
#     Aliphatic C   → greys
#     Unsaturated   → yellows/golds
#     Other/fallback→ brown

import matplotlib.colors as mcolors

_ENV_GROUPS = {
    'carbonyl': {
        'envs': ['C_carboxylic_acid', 'C_carboxylate', 'C_ester_carbonyl',
                 'C_amide_carbonyl', 'C_ketone', 'C_aldehyde',
                 'C_CO2', 'C_ketene'],
        'base_hue': 270,   # purple
        'sat': 0.70,
        'marker': 's',     # square
    },
    'oxygen_single': {
        'envs': ['C_ether', 'C_alcohol', 'C_ester_alkyl', 'C_phenol',
                 'C_enol', 'C_aryl_ether'],
        'base_hue': 210,   # blue
        'sat': 0.65,
        'marker': '^',     # triangle up
    },
    'nitrogen': {
        'envs': ['C_nitrile', 'C_imine', 'C_amine', 'C_aryl_amine',
                 'C_aryl_nitro', 'C_arom_N', 'C_arom_O_N',
                 'C_isocyanate', 'C_carbodiimide', 'C_ketenimine'],
        'base_hue': 150,   # green/teal
        'sat': 0.65,
        'marker': 'D',     # diamond
    },
    'halogen': {
        'envs': ['C_fluorinated', 'C_aryl_halide', 'C_acyl_halide'],
        'base_hue': 20,    # orange
        'sat': 0.80,
        'marker': 'P',     # plus (filled)
    },
    'aromatic': {
        'envs': ['C_aromatic', 'C_arom_O'],
        'base_hue': 330,   # pink/magenta
        'sat': 0.60,
        'marker': 'o',     # circle
    },
    'unsaturated': {
        'envs': ['C_alkyne', 'C_allene', 'C_vinyl'],
        'base_hue': 50,    # yellow/gold
        'sat': 0.75,
        'marker': 'v',     # triangle down
    },
    'aliphatic': {
        'envs': ['C_methyl', 'C_methylene', 'C_methine', 'C_quaternary'],
        'base_hue': 0,     # grey (sat=0)
        'sat': 0.0,
        'marker': 'X',     # X (filled)
    },
}


def _hsl_to_rgb(h: float, s: float, l: float) -> Tuple[float, float, float]:
    """Convert HSL (h in degrees, s and l in 0–1) to RGB (0–1)."""
    import colorsys
    return colorsys.hls_to_rgb(h / 360.0, l, s)


def _build_semantic_palette() -> Tuple[Dict[str, tuple], Dict[str, str]]:
    """
    Build env_name → RGBA colour and env_name → marker dicts.
    Within each group, lightness is spread evenly from light to dark.
    """
    color_by_name: Dict[str, tuple] = {}
    marker_by_name: Dict[str, str] = {}

    for grp_name, grp in _ENV_GROUPS.items():
        envs = grp['envs']
        n = len(envs)
        hue = grp['base_hue']
        sat = grp['sat']
        marker = grp['marker']

        # Lightness range: 0.35 (dark) to 0.75 (light)
        if n == 1:
            lightnesses = [0.55]
        else:
            lightnesses = np.linspace(0.75, 0.35, n).tolist()

        for env_name, light in zip(envs, lightnesses):
            r, g, b = _hsl_to_rgb(hue, sat, light)
            color_by_name[env_name] = (r, g, b, 1.0)
            marker_by_name[env_name] = marker

    return color_by_name, marker_by_name


# Pre-compute once at module load
_SEMANTIC_COLORS, _SEMANTIC_MARKERS = _build_semantic_palette()


def get_environment_colors(df: pd.DataFrame) -> Dict[int, tuple]:
    """
    Semantic colour map: environments grouped by heteroatom neighbour.
    Returns env_idx → RGBA tuple.
    """
    result: Dict[int, tuple] = {}
    for env_idx in sorted(df['carbon_env_label'].unique()):
        env_name = IDX_TO_CARBON_ENV.get(int(env_idx), '')
        if env_name in _SEMANTIC_COLORS:
            result[env_idx] = _SEMANTIC_COLORS[env_name]
        else:
            # Fallback grey
            result[env_idx] = (0.5, 0.5, 0.5, 1.0)
    return result


def get_environment_markers(df: pd.DataFrame) -> Dict[int, str]:
    """
    Marker map: environments grouped by heteroatom neighbour.
    Returns env_idx → matplotlib marker string.
    """
    result: Dict[int, str] = {}
    for env_idx in sorted(df['carbon_env_label'].unique()):
        env_name = IDX_TO_CARBON_ENV.get(int(env_idx), '')
        if env_name in _SEMANTIC_MARKERS:
            result[env_idx] = _SEMANTIC_MARKERS[env_name]
        else:
            result[env_idx] = 'o'
    return result


def get_ordered_unique_envs(df: pd.DataFrame) -> Tuple[List[int], Dict[int, str]]:
    """
    Get unique environments ordered by count (most frequent first).
    """
    env_counts = df['carbon_env_label'].value_counts()  # already sorted desc
    ordered_indices = env_counts.index.tolist()
    
    env_to_formatted = {
        env_idx: format_env_label(IDX_TO_CARBON_ENV.get(int(env_idx), f'Unknown_{env_idx}'))
        for env_idx in df['carbon_env_label'].unique()
    }
    
    return ordered_indices, env_to_formatted


def compute_spectral_scalars(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Convert spectral data to scalar representations.

    Returns a dict with keys:
        centroid, width, skewness, entropy, cdf_auc
    """
    E_MIN = 200.0  # eV
    E_MAX = 270.0  # eV
    E_RANGE = E_MAX - E_MIN  # 70 eV

    centroid_list = []
    width_list = []
    skewness_list = []
    entropy_list = []
    cdf_auc_list = []

    for idx, row in df.iterrows():
        spec = np.array(row['spectrum_intensity_only'])
        spec_cdf = np.array(row['cdf'])

        n_points = len(spec)
        energy_axis = np.linspace(E_MIN, E_MAX, n_points)
        dE = E_RANGE / (n_points - 1)

        total_intensity = np.sum(spec) + 1e-8

        # Centroid (intensity-weighted mean energy)
        mu = np.sum(spec * energy_axis) / total_intensity
        centroid_list.append(mu)

        # Width (intensity-weighted standard deviation)
        sigma = np.sqrt(np.sum(spec * (energy_axis - mu) ** 2) / total_intensity)
        width_list.append(sigma)

        # Skewness (third standardised moment)
        if sigma > 1e-10:
            skew = np.sum(spec * ((energy_axis - mu) / sigma) ** 3) / total_intensity
        else:
            skew = 0.0
        skewness_list.append(skew)

        # Entropy
        spec_norm = spec / total_intensity
        entropy = -np.sum(spec_norm * np.log(spec_norm + 1e-10))
        entropy_list.append(entropy)

        # CDF AUC
        auc = np.trapz(spec_cdf, dx=dE) / E_RANGE
        cdf_auc_list.append(auc)

    return {
        'centroid': np.array(centroid_list),
        'width': np.array(width_list),
        'skewness': np.array(skewness_list),
        'entropy': np.array(entropy_list),
        'cdf_auc': np.array(cdf_auc_list),
    }


def create_plots(df: pd.DataFrame, output_dir: str = './pngs_publication'):
    """
    Create publication figures:
    1. carbon_environment_distribution.png
    2. spectrum_centroid.png
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute spectral scalars (needed for centroid plot)
    print("\nComputing spectral scalar features...")
    scalars = compute_spectral_scalars(df)
    df['spectrum_energy'] = scalars['centroid']
    df['spectrum_width'] = scalars['width']
    df['spectrum_skewness'] = scalars['skewness']
    df['spectrum_entropy'] = scalars['entropy']
    df['cdf_auc'] = scalars['cdf_auc']
    
    # =========================================================================
    # Plot 2: Carbon Environment Distribution (large, detailed)
    # =========================================================================
    print("\nGenerating carbon_environment_distribution...")
    fig_sep2, ax_sep2 = plt.subplots(figsize=(16, 20))
    
    # Sort environments by count (ascending so most frequent at top of horizontal bar chart)
    env_counts_sep2 = df['carbon_env_label'].value_counts().sort_values()
    ordered_indices_sep2 = env_counts_sep2.index.tolist()
    ordered_counts_sep2 = env_counts_sep2.values.tolist()
    
    # Use viridis gradient based on count, normalized to GLOBAL maximum
    global_max_count_sep2 = max(ordered_counts_sep2)
    
    # Create gradient-filled bars with shared normalization
    for i, (idx, count) in enumerate(zip(ordered_indices_sep2, ordered_counts_sep2)):
        n_segments = 100
        x_positions = np.linspace(0, count, n_segments)
        colors_for_bar = plt.cm.viridis(x_positions / global_max_count_sep2)
        
        for j in range(n_segments - 1):
            x_start = x_positions[j]
            x_end = x_positions[j + 1]
            rect = plt.Rectangle((x_start, i - 0.4), x_end - x_start, 0.8,
                                facecolor=colors_for_bar[j], edgecolor='none', zorder=1)
            ax_sep2.add_patch(rect)
    
    ax_sep2.set_yticks(range(len(ordered_indices_sep2)))
    formatted_labels_sep2 = [IDX_TO_CARBON_ENV[int(i)].replace('C_', '').replace('_', ' ') for i in ordered_indices_sep2]
    ax_sep2.set_yticklabels(formatted_labels_sep2, fontsize=28)
    ax_sep2.set_xlabel('Count', fontsize=26, fontweight='bold')
    ax_sep2.set_xlim(0, max(ordered_counts_sep2) * 1.08)
    ax_sep2.set_ylim(-0.5, len(ordered_indices_sep2) - 0.5)
    ax_sep2.grid(True, alpha=0.3, axis='x', zorder=0)
    ax_sep2.tick_params(axis='x', labelsize=26)
    ax_sep2.tick_params(axis='y', labelsize=28)
    
    # Add count labels on bars
    for i, count in enumerate(ordered_counts_sep2):
        ax_sep2.text(count, i, f' {count}', va='center', fontsize=25, fontweight='bold', zorder=3)
    
    plt.tight_layout()
    
    plot2_path = os.path.join(output_dir, 'carbon_environment_distribution.png')
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved carbon_environment_distribution.png to: {plot2_path}")
    plot2_path = os.path.join(output_dir, 'carbon_environment_distribution.pdf')
    plt.savefig(plot2_path, bbox_inches='tight')
    print(f"✓ Saved carbon_environment_distribution.pdf to: {plot2_path}")
    plt.close()

    # =========================================================================
    # Lineshape Metrics: distribution (left) + 2×2 scatter + legend (right)
    # =========================================================================
    print("\nGenerating lineshape_metrics...")

    # Compute molecule-level CEBE: mol_cebe = atomic_be - delta_be
    df['mol_cebe'] = df['atomic_be'] - df['delta_be']

    ordered_envs, env_to_formatted = get_ordered_unique_envs(df)
    color_map = get_environment_colors(df)
    marker_map = get_environment_markers(df)

    from matplotlib.gridspec import GridSpec

    metric_info = [
        ('spectrum_energy',   'Spectrum Centroid (eV)'),
        ('spectrum_width',    'Spectrum Width (eV)'),
        ('spectrum_skewness', 'Spectrum Skewness'),
        ('spectrum_entropy',  'Spectrum Entropy'),
    ]

    axis_font_m = 26

    # Layout: outer 1×4 [distribution | spacer | 2×2 scatter | legend]
    # Nested inner 2×2 for scatter panels with independent row/col spacing
    fig_m = plt.figure(figsize=(42, 20))
    gs_outer = GridSpec(1, 4, width_ratios=[0.9, 0.10, 2.0, 0.25],
                        wspace=0.02, figure=fig_m,
                        left=0.05, right=0.98, top=0.96, bottom=0.06)

    gs_dist = gs_outer[0, 0].subgridspec(1, 1)
    # spacer column 1 is unused
    gs_scatter = gs_outer[0, 2].subgridspec(2, 2, wspace=0.12, hspace=0.02)
    gs_leg = gs_outer[0, 3].subgridspec(1, 1)

    # --- Distribution plot spanning full height (column 0) ---
    ax_dist = fig_m.add_subplot(gs_dist[0, 0])

    env_counts_sep2 = df['carbon_env_label'].value_counts().sort_values()
    ordered_indices_sep2 = env_counts_sep2.index.tolist()
    ordered_counts_sep2 = env_counts_sep2.values.tolist()
    global_max_count_sep2 = max(ordered_counts_sep2)

    for i, (idx, count) in enumerate(zip(ordered_indices_sep2, ordered_counts_sep2)):
        n_segments = 100
        x_positions = np.linspace(0, count, n_segments)
        colors_for_bar = plt.cm.viridis(x_positions / global_max_count_sep2)
        for j in range(n_segments - 1):
            x_start = x_positions[j]
            x_end = x_positions[j + 1]
            rect = plt.Rectangle((x_start, i - 0.4), x_end - x_start, 0.8,
                                 facecolor=colors_for_bar[j], edgecolor='none', zorder=1)
            ax_dist.add_patch(rect)

    ax_dist.set_yticks(range(len(ordered_indices_sep2)))
    formatted_labels_dist = [
        IDX_TO_CARBON_ENV[int(i)].replace('C_', '').replace('_', ' ')
        for i in ordered_indices_sep2
    ]
    ax_dist.set_yticklabels(formatted_labels_dist, fontsize=axis_font_m)
    ax_dist.set_xlabel('Count', fontsize=axis_font_m, fontweight='bold')
    ax_dist.set_xlim(0, max(ordered_counts_sep2) * 1.12)
    ax_dist.set_ylim(-0.5, len(ordered_indices_sep2) - 0.5)
    ax_dist.grid(True, alpha=0.3, axis='x', zorder=0)
    ax_dist.tick_params(axis='x', labelsize=axis_font_m)
    ax_dist.tick_params(axis='y', labelsize=axis_font_m)

    for i, count in enumerate(ordered_counts_sep2):
        ax_dist.text(count, i, f' {count}', va='center', fontsize=axis_font_m, fontweight='bold', zorder=3)

    # --- 2×2 scatter panels ---
    axes_m = [
        fig_m.add_subplot(gs_scatter[0, 0]),
        fig_m.add_subplot(gs_scatter[0, 1]),
        fig_m.add_subplot(gs_scatter[1, 0]),
        fig_m.add_subplot(gs_scatter[1, 1]),
    ]

    for ax, (col, ylabel) in zip(axes_m, metric_info):
        for env_idx in ordered_envs:
            mask = df['carbon_env_label'] == env_idx
            ax.scatter(
                df.loc[mask, 'mol_cebe'],
                df.loc[mask, col],
                c=[color_map[env_idx]],
                marker=marker_map[env_idx],
                alpha=0.45, s=80, edgecolors='none',
                label=env_to_formatted[env_idx],
            )
        ax.set_ylabel(ylabel, fontsize=axis_font_m, fontweight='bold')
        ax.grid(True, alpha=0.3, linewidth=1.0)
        ax.tick_params(axis='both', labelsize=axis_font_m)

    # Only bottom-row panels get CEBE x-axis label; top-row hides x tick labels
    for ax in axes_m[2:]:
        ax.set_xlabel('CEBE (eV)', fontsize=axis_font_m, fontweight='bold')
    for ax in axes_m[:2]:
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelbottom=False)
    # Trim CEBE axis to max 300 eV
    for ax in axes_m:
        ax.set_xlim(right=300)

    # --- Vertical legend (column 2 of outer grid) ---
    ax_leg = fig_m.add_subplot(gs_leg[0, 0])
    ax_leg.axis('off')
    handles, labels = axes_m[0].get_legend_handles_labels()
    ax_leg.legend(
        handles, labels, loc='center left',
        bbox_to_anchor=(0, 0.5),
        fontsize=axis_font_m, ncol=1, framealpha=0.95, edgecolor='black',
        markerscale=3,
    )

    for ext in ('png', 'pdf'):
        path = os.path.join(output_dir, f'lineshape_metrics.{ext}')
        fig_m.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f"✓ Saved: {path}")
    plt.close(fig_m)

    # =========================================================================
    # Spectrum Centroid 1×2 panel
    # =========================================================================
    print("\nGenerating spectrum_centroid...")

    axis_font = 30
    legend_font = 23
    
    # Use nested GridSpec for independent spacing control
    fig_centroid = plt.figure(figsize=(30, 14))
    # Main grid: 2 sections (left subplot+colorbar, right subplot+legend)
    gs_main = GridSpec(1, 2, width_ratios=[1.1, 1.25], wspace=0.10)
    
    # Left section: subplot + colorbar (tight spacing)
    gs_left = gs_main[0, 0].subgridspec(1, 2, width_ratios=[1, 0.04], wspace=0.02)
    ax_c1 = fig_centroid.add_subplot(gs_left[0, 0])
    cax = fig_centroid.add_subplot(gs_left[0, 1])
    
    # Right section: subplot + legend (tight spacing)
    gs_right = gs_main[0, 1].subgridspec(1, 2, width_ratios=[1, 0.25], wspace=0.02)
    ax_c2 = fig_centroid.add_subplot(gs_right[0, 0])
    
    alpha = 0.5
    
    # Left: Spectrum Centroid vs Molecule CEBE, colored by Electronegativity
    scatter_c1 = ax_c1.scatter(
        df['mol_cebe'],
        df['spectrum_energy'],
        c=df['e_neg_score'],
        cmap='plasma',
        alpha=alpha,
        s=150,
        edgecolors='none',
        linewidth=0.7
    )
    ax_c1.set_xlabel('CEBE (eV)', fontsize=axis_font, fontweight='bold')
    ax_c1.set_ylabel('Spectrum Centroid (eV)', fontsize=axis_font, fontweight='bold')
    ax_c1.grid(True, alpha=0.3, linewidth=1.5)
    ax_c1.tick_params(axis='both', labelsize=axis_font)
    cbar_c1 = plt.colorbar(scatter_c1, cax=cax)
    cbar_c1.set_label('Env. Electronegativity', fontsize=axis_font, fontweight='bold', rotation=270, labelpad=35)
    cbar_c1.ax.tick_params(labelsize=axis_font)
    
    # Right: Spectrum Centroid vs Molecule CEBE, colored by Carbon Environment
    for env_idx in ordered_envs:
        env_data = df[df['carbon_env_label'] == env_idx]
        ax_c2.scatter(
            env_data['mol_cebe'],
            env_data['spectrum_energy'],
            c=[color_map[env_idx]],
            marker=marker_map[env_idx],
            alpha=alpha,
            s=150,
            edgecolors='none',
            linewidth=0.7
        )
    
    ax_c2.set_xlabel('CEBE (eV)', fontsize=axis_font, fontweight='bold')
    ax_c2.grid(True, alpha=0.3, linewidth=1.5)
    ax_c2.tick_params(axis='both', labelsize=axis_font)
    ax_c2.set_yticklabels([])  # Remove y-axis labels for right plot
    # Trim CEBE axis to max 300 eV
    ax_c1.set_xlim(right=300)
    ax_c2.set_xlim(right=300)

    # Create legend in the dedicated space
    ax_legend = fig_centroid.add_subplot(gs_right[0, 1])
    ax_legend.axis('off')  # Hide axes
    
    legend_handles_c = []
    legend_labels_c = []
    for env_idx in ordered_envs:
        legend_handles_c.append(plt.scatter([], [], c=[color_map[env_idx]],
                                            marker=marker_map[env_idx],
                                            s=150, edgecolors='none'))
        legend_labels_c.append(env_to_formatted[env_idx])
    
    legend_c = ax_legend.legend(
        legend_handles_c, legend_labels_c,
        loc='center left',
        bbox_to_anchor=(0, 0.5),
        fontsize=legend_font,
        ncol=1,
        framealpha=0.95,
        edgecolor='black',
        #title='Carbon Environments',
        title_fontsize=legend_font
    )
    
    centroid_path = os.path.join(output_dir, 'spectrum_centroid.png')
    plt.savefig(centroid_path, dpi=300, bbox_inches='tight')
    centroid_pdf_path = os.path.join(output_dir, 'spectrum_centroid.pdf')
    plt.savefig(centroid_pdf_path, bbox_inches='tight')
    print(f"✓ Saved spectrum_centroid.png to: {centroid_path}")
    print(f"✓ Saved spectrum_centroid.pdf to: {centroid_pdf_path}")
    plt.close()
    
    return df


def main():
    """Main analysis function."""
    data_path = str(PROJECT_ROOT / 'data' / 'processed' / 'cnn_auger_calc_carbon.pkl')
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        sys.exit(1)
    
    print(f"Loading carbon dataframe from: {data_path}")
    df = cdf.load_carbon_dataframe(data_path)
    print(f"✓ Loaded {len(df)} carbon atoms")
    
    print("\nCreating publication plots...")
    create_plots(df)
    
    print("\n" + "=" * 80)
    print("✓ Plots complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
