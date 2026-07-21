import numpy as np
from pathlib import Path
import json

from ase import Atoms
from dscribe.descriptors import SOAP
    
from sklearn.metrics import mean_absolute_error, r2_score

from augernet import DATA_RAW_DIR

def detect_atom_types(*data_groups):
    s = set()
    for data in data_groups:
        for d in data:
            s.update(d["symbols"])
    return sorted(s)

def soap_input_and_be_output(atom_types, soap_params, data, n_jobs=4):
    #Initiate dscribe soap constructor
    soap = SOAP(species=atom_types, periodic=False, sparse=False, **soap_params)
    #Create ASE molecules  
    systems = [Atoms(symbols=d["symbols"], positions=d["positions"]) for d in data]
    # Only make SOAP descriptors for carbons with BE labels
    centers = [d["cidx"] for d in data]
    D = soap.create(systems, centers=centers, n_jobs=n_jobs)
    X = np.vstack(D)
    # y in data dict already selected with cidx 
    y = np.concatenate([d["y"] for d in data])
    # row-index map for data efficiencey scan    
    name2rows, off = {}, 0
    for dat, d in zip(data, D):
        name2rows[dat["name"]] = np.arange(off, off + len(d)); off += len(d)
    return X, y, name2rows

def metrics(pred, y):
    err = y - pred
    return dict(pred=pred, MAE=float(mean_absolute_error(y, pred)),
                RMSE=float(np.sqrt(np.mean(err ** 2))),
                R2=float(r2_score(y, pred)), STD=float(np.std(err)),
                MAX=float(np.max(np.abs(err))))

#######################################################
## augernet_data_result.py specific functions
#######################################################

def _read_list(path):
    return [ln.strip() for ln in open(path) if ln.strip()]

def _parse_xyz_files(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    syms, pos = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        syms.append("".join(c for c in p[0] if c.isalpha()))
        pos.append([float(p[1]), float(p[2]), float(p[3])])
    return syms, np.asarray(pos, float)

def load_augernet_data():
    data_raw = Path(DATA_RAW_DIR)
    def load_set(folder, mol_list):
        data = []
        for m in mol_list:
            xyz, out = data_raw / folder / f"{m}.xyz", data_raw / folder / f"{m}_out.txt"
            syms, pos = _parse_xyz_files(xyz)
            cebe = np.loadtxt(out).reshape(-1)
            cidx = [i for i, (s, v) in enumerate(zip(syms, cebe)) if s == "C" and v != -1.0]
            if cidx:
                data.append(dict(name=m, symbols=syms, positions=pos, cidx=cidx,
                                 y=np.array([cebe[i] for i in cidx]),
                                 natoms=len(syms)))
        return data
    return dict(
        calc=load_set("calc_cebe",    _read_list(data_raw / "calc_cebe" / "mol_list.txt")),
        exp_val=load_set("exp_cebe",  _read_list(data_raw / "exp_cebe" / "mol_list_val.txt")),
        exp_eval=load_set("exp_cebe", _read_list(data_raw / "exp_cebe" / "mol_list_eval.txt")),
    )

def augernet_butina_split(fold, n_folds, cutoff):

    from augernet.build_molecular_graphs import _mol_from_xyz_order, get_butina_clusters
    from sklearn.model_selection import GroupKFold

    smiles = []
    data_raw = Path(DATA_RAW_DIR)
    mol_list =  _read_list(data_raw / "calc_cebe" / "mol_list.txt") 
    for name in mol_list:
        _, _, _, smi = _mol_from_xyz_order(str(data_raw / "calc_cebe" / f"{name}.xyz"),
                                           labeled_atoms=False)
        smiles.append(smi)

    clusters = get_butina_clusters(smiles, cutoff=cutoff)
    folds = list(GroupKFold(n_splits=n_folds).split(np.arange(len(mol_list)), groups=clusters))
    train, val = folds[fold - 1]
    train_list, val_list = train.tolist(), val.tolist()
    return mol_list, train_list, val_list 


#######################################################
## reproduce_porcelli_et_al.py1 specific functions
#######################################################

def _parse_xyz_and_be_from_string(xyz_string):
    """Porcelli et al. xyz atom lines are 'element x y z BE' (BE='Nan' if not target)."""
    lines = xyz_string.strip().splitlines()
    n = int(lines[0].split()[0])
    symbol, pos, be = [], [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        if len(p) < 4:
            continue
        symbol.append("".join(c for c in p[0] if c.isalpha()))
        pos.append([float(p[1]), float(p[2]), float(p[3])])
        try:
            be.append(float(p[4]) if len(p) >= 5 else np.nan)
        except ValueError:
            be.append(np.nan)
    return symbol, np.asarray(pos, float), np.asarray(be, float)

def load_data_from_json(json_path):
    with open(json_path, 'r') as f:
        data_json = json.load(f)
    data = []
    for item in data_json:
        symbol, pos, be = _parse_xyz_and_be_from_string(item['xyz_file'])
        cidx = [i for i, (s, b) in enumerate(zip(symbol, be)) if s == "C" and not np.isnan(b) and b != -1.0]
        if cidx:
            data.append(dict(name=item['name'], symbols=symbol, positions=pos,
                             cidx=cidx, y=be[cidx], natoms=len(symbol)))
    return data


#######################################################
## Evulation and plotting routines 
#######################################################


def conformal_quantile(residuals, alpha):
    """
    Compute split conformal prediction quantile.
    
    Parameters
    ----------
    residuals : np.ndarray
        Absolute residuals from validation set.
    alpha : float
        Error probability (1 - confidence level).
    
    Returns
    -------
    float
        Quantile threshold q_hat for conformal bands.
    """
    n = len(residuals)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    sorted_residuals = np.sort(residuals)
    if k > n:
        print(f"Split CP Warning: k={k} exceeds number of residuals n={n}. Returning infinity.")
        return np.inf
    return sorted_residuals[k]


def _rstats(pred, exp):
    """Calculate R2, MAE, STD for scatter plot statistics."""
    mae = float(mean_absolute_error(exp, pred))
    r2 = float(r2_score(exp, pred))
    std = float(np.std(exp - pred))
    return dict(r2=r2, mae=mae, std=std)


def _scatter_panel(ax, all_pred, all_exp, rstats, scatter_s, LINE_WIDTH,
                   STATS_FONT, AXIS_FONT, TICK_FONT, subplot=None, mol_sizes=None,
                   vmin=None, vmax=None, cmap="YlOrRd", size_threshold=16,
                   below_color="#0072B2", pred_label="Predicted CEBE (eV)", band=0.68,
                   scp_value=None, show_band=True, show_scp=True):
    import matplotlib.colors as mcolors
    if mol_sizes is not None:
        mask_below = mol_sizes <= size_threshold
        mask_above = ~mask_below
        ax.scatter(all_pred[mask_below], all_exp[mask_below], alpha=0.6, s=scatter_s,
                   color=below_color, edgecolors=None, linewidth=0.2, zorder=3,
                   label=f"≤{size_threshold} atoms")
        cmap_vmin = vmin if vmin is not None else mol_sizes[mask_above].min()
        cmap_vmax = vmax if vmax is not None else mol_sizes[mask_above].max()
        norm = mcolors.Normalize(vmin=cmap_vmin, vmax=cmap_vmax)
        sc = ax.scatter(all_pred[mask_above], all_exp[mask_above], alpha=0.8, s=scatter_s,
                        c=mol_sizes[mask_above], cmap=cmap, norm=norm,
                        edgecolors="black", linewidth=0.4, zorder=4)
    else:
        sc = ax.scatter(all_pred, all_exp, alpha=0.4, s=scatter_s, edgecolors=None,
                        linewidth=0.2, color=below_color, zorder=3)
    lo = min(all_exp.min(), all_pred.min())
    hi = max(all_exp.max(), all_pred.max())
    pad = (hi - lo) * 0.03
    diag = np.array([lo - pad, hi + pad])
    
    # Only draw shaded band if show_band is True
    if show_band:
        ax.fill_between(diag, diag - band, diag + band, color="#301934", alpha=0.2, zorder=1)
    
    ax.plot(diag, diag, "k:", linewidth=LINE_WIDTH, alpha=0.7, zorder=2)
    
    # Build text annotation with optional split-CP value
    stats_text = (f"R$^{{2}}$ = {rstats['r2']:.2f}\nMAE = {rstats['mae']:.2f} eV\n"
                  f"STD = {rstats['std']:.2f} eV")
    if show_scp and scp_value is not None:
        stats_text += f"\nS-CP = {scp_value:.2f} eV"
    
    ax.text(0.05, 0.95,
            stats_text,
            ha="left", va="top", transform=ax.transAxes, fontsize=STATS_FONT,
            fontweight="bold",
            bbox=dict(boxstyle="round", edgecolor="grey", facecolor="white",
                      alpha=0.85, pad=0.5), zorder=5)
    ax.set_xlabel(pred_label, fontsize=AXIS_FONT, fontweight="bold")
    if subplot is None or subplot == "(a)":
        ax.set_ylabel("Experimental CEBE (eV)", fontsize=AXIS_FONT, fontweight="bold")
    else:
        ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=TICK_FONT)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.grid(True, alpha=0.3, linewidth=1.0, zorder=0); ax.set_axisbelow(True)
    if subplot:
        ax.text(-0.12, 1.05, subplot, transform=ax.transAxes,
                fontsize=AXIS_FONT + 2, fontweight="bold", va="top")
    return ax, sc


def write_cebe_labels(pred_data, true_data, mol_symbols, output_dir, file_stem):
    """
    Write CEBE evaluation results to a labels file.
    
    Produces a text file with per-atom predicted vs true CEBE values, organized by molecule.
    Format mirrors evaluate_cebe_model.py output.
    
    Parameters
    ----------
    pred_data : np.ndarray
        Predicted CEBE values (per carbon atom).
    true_data : np.ndarray
        True (experimental) CEBE values (per carbon atom).
    mol_symbols : list of dicts
        Each dict has 'name' and 'symbols' keys. Used to organize output by molecule.
    output_dir : str or Path
        Directory for output file.
    file_stem : str
        Base filename (labels file will be {file_stem}_labels.txt).
    
    Returns
    -------
    Path
        Path to the written labels file.
    """
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    label_path = output_dir / f"{file_stem}_labels.txt"
    
    with open(label_path, 'w') as f:
        f.write("# CEBE Evaluation Results (SOAP-KRR)\n")
        f.write("# Columns: atom_symbol  true_BE(eV)  pred_BE(eV)  error(eV)\n")
        f.write("# Non-carbon atoms and missing values marked with '--'\n")
        f.write("#\n")
        
        pred_idx = 0
        for mol_info in mol_symbols:
            mol_name = mol_info['name']
            syms = mol_info['symbols']
            cidx = mol_info.get('cidx', [])
            
            f.write(f"# --- {mol_name} ---\n")
            
            c_count = 0
            for atom_idx, sym in enumerate(syms):
                if atom_idx in cidx:
                    # This is a carbon atom with a prediction
                    true_be = float(true_data[pred_idx])
                    pred_be = float(pred_data[pred_idx])
                    error = pred_be - true_be
                    f.write(f"{sym:>3s}    {true_be:10.4f}    {pred_be:10.4f}    {error:10.4f}\n")
                    pred_idx += 1
                    c_count += 1
                else:
                    # Non-carbon or carbon with no prediction
                    f.write(f"{sym:>3s}    {'--':>10s}    {'--':>10s}    {'--':>10s}\n")
            
            f.write("\n")
    
    return label_path


def two_panel_scatter(res, out_dir, tag="augernet",
                      pred_label="SOAP/KRR Predicted CEBE (eV)", band=0.68):
    """band = split-CP half-width (q_hat) of THIS model, shaded around y=x."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import seaborn as sns

    pv, ev, sv = res["exp_val"]["pred"], res["exp_val"]["y"], res["exp_val"]["natoms"]
    scp_val = res["exp_val"].get("scp", None)
    pe, ee, se = res["exp_eval"]["pred"], res["exp_eval"]["y"], res["exp_eval"]["natoms"]
    scp_eval = res["exp_eval"].get("scp", None)
    rstats_val, rstats_eval = _rstats(pv, ev), _rstats(pe, ee)

    above = np.concatenate([sv[sv > 16], se[se > 16]])
    vmin = above.min() if len(above) else 17
    vmax = above.max() if len(above) else 45

    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 10, 'axes.linewidth': 1.2,
                         'xtick.direction': 'in', 'ytick.direction': 'in',
                         'xtick.major.width': 1.0, 'xtick.major.width': 1.0,
                         'xtick.major.size': 5, 'ytick.major.size': 5,
                         'legend.frameon': True})
    AXIS_FONT, TICK_FONT, STATS_FONT, LINE_WIDTH, scatter_s = 16, 17, 16, 2.2, 60

    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, wspace=0.01)

    ax_l = fig.add_subplot(gs[0, 0])
    ax_l, sc = _scatter_panel(ax_l, pv, ev, rstats_val, scatter_s, LINE_WIDTH,
                              STATS_FONT, AXIS_FONT, TICK_FONT, subplot="(a)",
                              mol_sizes=sv, vmin=vmin, vmax=vmax, pred_label=pred_label,
                              band=band, scp_value=scp_val, show_band=False, show_scp=False)
    ax_l.text(0.95, 0.05, "Exp. Validation", ha="right", va="bottom",
              transform=ax_l.transAxes, fontsize=STATS_FONT, fontweight="bold",
              bbox=dict(boxstyle="round", edgecolor="grey", facecolor="white",
                        alpha=0.85, pad=0.5), zorder=5)

    ax_r = fig.add_subplot(gs[0, 1])
    ax_r, sc_eval = _scatter_panel(ax_r, pe, ee, rstats_eval, scatter_s, LINE_WIDTH,
                                   STATS_FONT, AXIS_FONT, TICK_FONT, subplot="(b)",
                                   mol_sizes=se, vmin=vmin, vmax=vmax, pred_label=pred_label,
                                   band=band, scp_value=scp_eval, show_band=True, show_scp=True)
    ax_r.text(0.95, 0.05, "Exp. Evaluation", ha="right", va="bottom",
              transform=ax_r.transAxes, fontsize=STATS_FONT, fontweight="bold",
              bbox=dict(boxstyle="round", edgecolor="grey", facecolor="white",
                        alpha=0.85, pad=0.5), zorder=5)

    cbar = fig.colorbar(sc_eval, ax=[ax_l, ax_r], shrink=0.85, pad=0.02)
    cbar.set_label("Molecule Size (atoms)", fontsize=AXIS_FONT, fontweight="bold")
    cbar.ax.tick_params(labelsize=TICK_FONT)

    png = Path(out_dir) / f"2_panel_scatter_{tag}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(Path(out_dir) / f"2_panel_scatter_{tag}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png