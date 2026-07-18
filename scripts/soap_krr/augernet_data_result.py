import numpy as np
from pathlib import Path
import itertools
from sklearn.kernel_ridge import KernelRidge
import json

from soap_krr_utils import (
    detect_atom_types, soap_input_and_be_output, metrics,
    load_augernet_data, augernet_butina_split,
    load_data_from_json 
    )

OUT_DIR = Path("soap_krr_outputs")
OUT_DIR.mkdir(exist_ok=True)

SOAP_PARAM_GRID = dict(rcut=[4.0, 6.0, 8.0], nmax=[6, 8, 10],
                       lmax=[4, 6, 8], sigma=[0.025, 0.05, 0.1])
KRR_PARAM_GRID  = dict(gamma=list(np.logspace(-3, 1, 9)), alpha=list(np.logspace(-11, -1, 6)))

SOAP_PARAMS = dict(r_cut=4.0, n_max=8, l_max=6, sigma=0.025)
KRR_PARAMS  = dict(kernel="rbf", alpha=1e-11, gamma=1.0)

# OMP
N_JOBS  = 1
#Split seed
PARAM_SEED  = 42
FOLD    = 5
N_FOLDS = 10
CUTOFF  = 0.65
SELECT_MAX = 3000 # sub sample train data for faster param search

DSCAN_SEED = [0, 1, 2, 3]
DATA_FRAC  = [0.1, 0.25, 0.5, 0.75, 1.0]

# RUN_TYPE Options: 
# 'train': Uses SOAP_PARAMS and KRR_PARAMS to train single model
# 'param': Uses SOAP_PARAM_GRID and KRR_PARAM_GRID for hyper param search 
#           calc-val used to select KRR_PARAMS, exp-val used to select SOAP_PARAMS 
# 'dscan': Uses SOAP_PARAMS and KRR_PARAMS for data efficiencey scan, samples 4 seeds
RUN_TYPE = 'train' 

# Need to have prevsiously run scripts/prepare_data.py to download data from zenodo to data/raw
data = load_augernet_data()
pes_data = load_data_from_json('ws22_geom.json')

# Data info
atom_types = detect_atom_types(data["calc"], data["exp_val"], data["exp_eval"], pes_data)
for nm in ("calc", "exp_val", "exp_eval"):
    print(f"  {nm:9s}: {len(data[nm])} mols, {sum(len(d['cidx']) for d in data[nm])} C")

print(f"  WS22 PES Data: {len(pes_data)} mols, {sum(len(d['cidx']) for d in pes_data)} C")
# butina slit on calc data mol_list to copy AugerNet EGNN split
mol_list, calc_train_mol_idx, calc_val_mol_idx = augernet_butina_split(FOLD, N_FOLDS, CUTOFF)

# Split calc data into train/val molecule sets
data['calc_train'] = [data['calc'][i] for i in calc_train_mol_idx]
data['calc_val'] = [data['calc'][i] for i in calc_val_mol_idx]

X = {}
y = {}
name2rows  = {}

############################################
# Train one model: 
# Apply to val, eval and non-equil data
###########################################

if RUN_TYPE == 'train':

    for key in ("calc_train", "calc_val", "exp_val", "exp_eval"):
        X[key], y[key], _ = soap_input_and_be_output(atom_types, SOAP_PARAMS, data[key], N_JOBS)

    print(f"\nSOAP descriptors generated:")
    print(f" Calc. Train: {len(X['calc_train'])} C atoms")
    print(f" Calc. Val:   {len(X['calc_val'])} C atoms\n")

    mu = y['calc_train'].mean() 
    model = KernelRidge(kernel=KRR_PARAMS["kernel"], alpha=KRR_PARAMS["alpha"],
                            gamma=KRR_PARAMS["gamma"]).fit(X['calc_train'], y['calc_train']-mu)
    results = {}
    for key in ("calc_val", "exp_val", "exp_eval"):
        results[key] = metrics(model.predict(X[key])+mu, y[key])
        print(f"{key}:")
        print(f"\tMAE = {results[key]['MAE']:.3f} eV  |  RMSE {results[key]['RMSE']:.3f} | R2 {results[key]['R2']:.3f}  | "
            f"MAX {results[key]['MAX']:.3f} | STD {results[key]['STD']:.3f}\n")

    X_pes, y_pes, _  = soap_input_and_be_output(atom_types, SOAP_PARAMS, pes_data, N_JOBS)
    pes_preds = model.predict(X_pes) + mu

    #write pes results to .txt file
    pes_out_file = OUT_DIR /  "soap_krr_pes_labels.txt"
    with open(pes_out_file, "w") as f:
        f.write("# CEBE Predictions (SOAP-KRR)\n")
        f.write(f"# Model: SOAP-KRR  rcut={SOAP_PARAMS['r_cut']} nmax={SOAP_PARAMS['n_max']} ")
        f.write(f"lmax={SOAP_PARAMS['l_max']} sigma={SOAP_PARAMS['sigma']} ")
        f.write(f"alpha={KRR_PARAMS['alpha']} gamma={KRR_PARAMS['gamma']}\n")
        f.write("# Columns: atom_symbol  pred_BE(eV)   (non-carbon atoms starred)\n")
        f.write("#\n")
        
        pred_idx = 0
        for mol_data in pes_data:
            mol_name = mol_data["name"]
            syms = mol_data["symbols"]
            cidx = mol_data["cidx"]
            
            f.write(f"# --- {mol_name} ---\n")
            c_idx = 0
            for atom_idx, sym in enumerate(syms):
                if sym == "C":
                    f.write(f"  C      {pes_preds[pred_idx]:.4f}\n")
                    pred_idx += 1
                else:
                    f.write(f"  {sym}*     0.0000\n")
            f.write("\n")
    
    print(f"Wrote {pes_out_file}")




###########################################
# Hyperparam search
###########################################

if RUN_TYPE == 'param':

    soap_settings = list(itertools.product(SOAP_PARAM_GRID["rcut"], SOAP_PARAM_GRID["nmax"], SOAP_PARAM_GRID["lmax"], SOAP_PARAM_GRID["sigma"]))
    kernel_settings = list(itertools.product(KRR_PARAM_GRID["gamma"], KRR_PARAM_GRID["alpha"]))

    total_fits = len(soap_settings) * len(kernel_settings)
    done_fits = 0
    print(f"  total KRR fits to run: {total_fits}\n")

    best = None
    for si, (rcut, nmax, lmax, sigma) in enumerate(soap_settings, 1):

        sel = data['calc_train']
        if SELECT_MAX and sum(len(r["cidx"]) for r in sel) > SELECT_MAX:
            rng = np.random.RandomState(PARAM_SEED)
            order = rng.permutation(len(sel))
            sel, n = [], 0
            for i in order:
                sel.append(data['calc_train'][i]); n += len(data['calc_train'][i]["cidx"])
                if n >= SELECT_MAX:
                    break

        sp = dict(r_cut=float(rcut), n_max=int(nmax), l_max=int(lmax), sigma=float(sigma))
        Xtr, ytr, _  = soap_input_and_be_output(atom_types, sp, sel, N_JOBS)
        Xcv, ycv, _  = soap_input_and_be_output(atom_types, sp, data['calc_val'], N_JOBS)
        Xev, yev, _  = soap_input_and_be_output(atom_types, sp, data['exp_val'],  N_JOBS)

        mu = ytr.mean() 
        # Use calc-val for KKR params
        best_krr = None
        for gamma, alpha in kernel_settings:
            model = KernelRidge(kernel=KRR_PARAMS["kernel"], alpha=alpha, gamma=gamma).fit(Xtr, ytr-mu)
            done_fits += 1
            calc_val_mae = metrics(model.predict(Xcv)+mu, ycv)["MAE"]
            if best_krr is None or calc_val_mae < best_krr["calc_val_mae"]:
                best_krr = dict(gamma=float(gamma), alpha=float(alpha),
                                calc_val_mae=float(calc_val_mae), model=model)

        # Use Exp-val for SOAP params
        exp_val_mae = metrics(best_krr["model"].predict(Xev)+mu, yev)["MAE"]
        if best is None or exp_val_mae < best["exp_val_mae"]:
            best = dict(soap=sp, gamma=best_krr["gamma"], alpha=best_krr["alpha"],
                        calc_val_mae=best_krr["calc_val_mae"], exp_val_mae=float(exp_val_mae))
            print(f"  [{si}/{len(soap_settings)}] {sp}  calc-val {best_krr['calc_val_mae']:.3f} "
                  f"| exp-val {exp_val_mae:.3f}  (a={best_krr['alpha']:.1e}, g={best_krr['gamma']:.1e})  *")
        elif si % 25 == 0:
            print(f"  [{si}/{len(soap_settings)}] ... best exp-val {best['exp_val_mae']:.3f}")
        
        # live progress across the whole search
        print(f"  progress: {si}/{len(soap_settings)} SOAP settings | "
              f"{done_fits}/{total_fits} fits ({100*done_fits/total_fits:4.1f}%)",
              end="\r", flush=True)
    print()  # finish the printout progress line

    # remake SOAP and KRR with best params and apply to fulll train data
    Xtr, ytr, _ = soap_input_and_be_output(atom_types, best['soap'], data['calc_train'], N_JOBS)
    mu = ytr.mean()
    final = KernelRidge(kernel=KRR_PARAMS["kernel"], alpha=best['alpha'], gamma=best['gamma']).fit(Xtr, ytr-mu)

    print("\nFinal model (winner refit on full calc_train):")
    for key in ("calc_val", "exp_val", "exp_eval"):
        Xk, yk, _ = soap_input_and_be_output(atom_types, best['soap'], data[key], N_JOBS)
        results = metrics(final.predict(Xk)+mu, yk)
        print(f"  {key:9s} MAE {results['MAE']:.3f} | RMSE {results['RMSE']:.3f} | R2 {results['R2']:.3f} | MAX {results['MAX']:.3f}")

    print(f"\n# paste into the RUN_TYPE='train' block:")
    print(f"SOAP_PARAMS = {best['soap']}")
    print(f"KRR_PARAMS  = dict(kernel='rbf', alpha={best['alpha']}, gamma={best['gamma']})")


###########################################
# Data Efficiencey Scan
###########################################

if RUN_TYPE == 'dscan':

    for key in ("calc_train", "calc_val", "exp_val", "exp_eval"):
        X[key], y[key], name2rows[key] = soap_input_and_be_output(atom_types, SOAP_PARAMS, data[key], N_JOBS)

    train_pool = calc_train_mol_idx
    total_results = []

    for frac in DATA_FRAC:
        seeds = [DSCAN_SEED[0]] if frac == 1.0 else DSCAN_SEED
        for seed in seeds:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(train_pool))
            k = max(1, int(round(frac * len(train_pool))))
            keep = [train_pool[perm[i]] for i in range(k)]
            keep_names = [mol_list[i] for i in keep]

            tr_rows_list = []
            for mol_name in keep_names:
                if mol_name in name2rows['calc_train']:
                    tr_rows_list.append(name2rows['calc_train'][mol_name])
            if not tr_rows_list:
                 print(f"  WARNING: No matching molecules for frac={frac}, seed={seed}")
                 continue

            tr_rows = np.concatenate(tr_rows_list)
            Xtr, ytr = X['calc_train'][tr_rows], y['calc_train'][tr_rows]
            mu = ytr.mean() 
            model = KernelRidge(kernel=KRR_PARAMS["kernel"], alpha=KRR_PARAMS["alpha"], 
                                gamma=KRR_PARAMS["gamma"]).fit(Xtr, ytr-mu)

            calc_val_results = metrics(model.predict(X["calc_val"])+mu, y["calc_val"])
            exp_val_results  = metrics(model.predict(X["exp_val"])+mu,  y["exp_val"])
            exp_eval_results = metrics(model.predict(X["exp_eval"])+mu, y["exp_eval"])

            print(f"  frac={frac:<4} seed={seed}  n_mol={len(keep):<4} "
                  f"n_C={len(tr_rows):<5} exp-eval MAE={exp_eval_results['MAE']:.3f}"
                  f"exp-val MAE={exp_val_results['MAE']:.3f} calc-val MAE={calc_val_results['MAE']:.3f}")

            total_results.append(dict(train_frac=frac, 
                                        n_train_mols=len(keep),
                                        calc_val_results=calc_val_results,
                                        exp_val_results=exp_val_results,
                                        exp_eval_results=exp_eval_results
                                    ))
            

    # ---- save results (for a separate plotting script) ----
    import pandas as pd

    flat_results = []
    for r in total_results:
        flat_results.append(dict(
            train_frac=r['train_frac'],
            n_train_mols=r['n_train_mols'],
            calc_val_MAE=r['calc_val_results']['MAE'],
            exp_val_MAE=r['exp_val_results']['MAE'],
            exp_eval_MAE=r['exp_eval_results']['MAE']
        ))

    df = pd.DataFrame(flat_results)
    csv = OUT_DIR / "data_efficiency_results.csv"
    df.to_csv(csv, index=False)

    summary = (df.groupby("train_frac")
                 .agg(n_train_mols=("n_train_mols", "first"),
                      calc_val_MAE_mean=("calc_val_MAE", "mean"),
                      calc_val_MAE_std=("calc_val_MAE",  "std"),
                      exp_val_MAE_mean=("exp_val_MAE",   "mean"),
                      exp_val_MAE_std=("exp_val_MAE",    "std"),
                      exp_eval_MAE_mean=("exp_eval_MAE", "mean"),
                      exp_eval_MAE_std=("exp_eval_MAE",  "std"))
                 .reset_index())

    print("\nLearning curve (exp-eval MAE, mean +/- std over seeds):")
    for _, r in summary.iterrows():
        exp_eval_std = 0 if np.isnan(r.exp_eval_MAE_std) else r.exp_eval_MAE_std
        exp_val_std  = 0 if np.isnan(r.exp_val_MAE_std)  else r.exp_val_MAE_std
        calc_val_std = 0 if np.isnan(r.calc_val_MAE_std) else r.calc_val_MAE_std
        print(f"  {int(r.n_train_mols):>4} mols ({r.train_frac:.2f}) : "
              f"exp-eval {r.exp_eval_MAE_mean:.3f} +/- {exp_eval_std:.3f} | "
              f"exp-val {r.exp_val_MAE_mean:.3f} +/- {exp_val_std:.3f} | "
              f"calc-val {r.calc_val_MAE_mean:.3f} +/- {calc_val_std:.3f} eV")

    (OUT_DIR / "data_efficiency_config.json").write_text(json.dumps(dict(
        soap=SOAP_PARAMS, krr=KRR_PARAMS, 
        n_folds=N_FOLDS, train_fold=FOLD, butina_cutoff=CUTOFF,
        fracs=DATA_FRAC, seeds=DSCAN_SEED,
        n_pool_mols=len(train_pool), n_val_mols=len(calc_val_mol_idx),
        n_exp_val_C=int(len(y["exp_val"])), n_exp_eval_C=int(len(y["exp_eval"]))), indent=2))

    print(f"\nSaved {csv.name} + data_efficiency_config.json to {OUT_DIR})")