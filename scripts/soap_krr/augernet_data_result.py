import numpy as np
import itertools
from sklearn.kernel_ridge import KernelRidge

from soap_krr_utils import (
    detect_atom_types, soap_input_and_be_output, metrics,
    load_augernet_data, augernet_butina_split 
    )

SOAP_PARAM_GRID = dict(rcut=[4.0, 6.0, 8.0], nmax=[6, 8, 10],
                      lmax=[4, 6, 8], sigma=[0.025, 0.05, 0.1])
KRR_PARAM_GRID  = dict(gamma=list(np.logspace(-3, 1, 9)), alpha=list(np.logspace(-11, -1, 6)))

#SOAP_PARAMS = dict(r_cut=5.0, n_max=8, l_max=4, sigma=0.1)
#KRR_PARAMS  = dict(kernel="rbf", gamma=0.1, alpha=1e-7)
SOAP_PARAMS = dict(r_cut=8.0, n_max=8, l_max=4, sigma=0.05)
KRR_PARAMS  = dict(kernel="rbf", alpha=1e-11, gamma=0.25)

# OMP
N_JOBS  = 4
#Split seed
SEED    = 42
FOLD    = 5
N_FOLDS = 10
CUTOFF  = 0.65
SELECT_MAX = 3000 # sub sample train data for faster param search

# RUN_TYPE Options: 
# 'train': uses SOAP_PARAMS and KRR_PARAMS 
# 'param': uses SOAP_PARAM_GRID and KRR_PARAM_GRID 
RUN_TYPE = 'param' 

# Need to have prevsiously run scripts/prepare_data.py to download data from zenodo to data/raw
data = load_augernet_data()

# Data info
atom_types = detect_atom_types(data["calc"], data["exp_val"], data["exp_eval"])
for nm in ("calc", "exp_val", "exp_eval"):
    print(f"  {nm:9s}: {len(data[nm])} mols, {sum(len(r['cidx']) for r in data[nm])} C")

# butina slit on calc data mol_list to copy AugerNet EGNN split
calc_train_mol_idx, calc_val_mol_idx = augernet_butina_split(FOLD, N_FOLDS, CUTOFF)

# Split calc data into train/val molecule sets
data['calc_train'] = [data['calc'][i] for i in calc_train_mol_idx]
data['calc_val'] = [data['calc'][i] for i in calc_val_mol_idx]

X = {}
y = {}
natoms = {}
names  = {}

if RUN_TYPE == 'train':
    for key in ("calc_train", "calc_val", "exp_val", "exp_eval"):
        X[key], y[key], natoms[key], names[key] = soap_input_and_be_output(atom_types, SOAP_PARAMS, data[key], N_JOBS)

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
            rng = np.random.RandomState(SEED)
            order = rng.permutation(len(sel))
            sel, n = [], 0
            for i in order:
                sel.append(data['calc_train'][i]); n += len(data['calc_train'][i]["cidx"])
                if n >= SELECT_MAX:
                    break

        sp = dict(r_cut=float(rcut), n_max=int(nmax), l_max=int(lmax), sigma=float(sigma))
        Xtr, ytr, _, _ = soap_input_and_be_output(atom_types, sp, sel, N_JOBS)
        Xcv, ycv, _, _ = soap_input_and_be_output(atom_types, sp, data['calc_val'],   N_JOBS)
        Xev, yev, _, _ = soap_input_and_be_output(atom_types, sp, data['exp_val'],    N_JOBS)

        mu = ytr.mean() 
        # Use calc-val for KKR params
        best_krr = None
        for gamma, alpha in kernel_settings:
            model = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma).fit(Xtr, ytr-mu)
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
    Xtr, ytr, _, _ = soap_input_and_be_output(atom_types, best['soap'], data['calc_train'], N_JOBS)
    mu = ytr.mean()
    final = KernelRidge(kernel="rbf", alpha=best['alpha'], gamma=best['gamma']).fit(Xtr, ytr-mu)

    print("\nFinal model (winner refit on full calc_train):")
    for key in ("calc_val", "exp_val", "exp_eval"):
        Xk, yk, _, _ = soap_input_and_be_output(atom_types, best['soap'], data[key], N_JOBS)
        r = metrics(final.predict(Xk)+mu, yk)
        print(f"  {key:9s} MAE {r['MAE']:.3f} | RMSE {r['RMSE']:.3f} | R2 {r['R2']:.3f} | MAX {r['MAX']:.3f}")

    print(f"\n# paste into the RUN_TYPE='train' block:")
    print(f"SOAP_PARAMS = {best['soap']}")
    print(f"KRR_PARAMS  = dict(kernel='rbf', alpha={best['alpha']}, gamma={best['gamma']})")