import numpy as np

from ase import Atoms
from dscribe.descriptors import SOAP
    
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

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
    X = np.vstack(soap.create(systems, centers=centers, n_jobs=n_jobs))
    # y in data dict already selected with cidx 
    y = np.concatenate([d["y"] for d in data])
    natoms = np.concatenate([np.full(len(d["cidx"]), d["natoms"]) for d in data])
    names = np.concatenate([[d["name"]] * len(d["cidx"]) for d in data])
    return X, y, natoms, names

def krr_single_or_cv(X, y, kernel, alphas, gammas, folds, standardize, seed=0, n_jobs=2):
    """5-fold-CV grid over (alpha, gamma). Returns (alpha, gamma, cv_MAE).
    Uses a SHUFFLED KFold: without shuffling, alphabetically-sorted molecules form
    contiguous folds that group similar chemistry and inflate the CV error.
    With singleton alpha/gamma grids this just reports the CV MAE of that pair."""
    if standardize:
        X = StandardScaler().fit_transform(X)
    yc = y - y.mean()  # MAE is offset-invariant, so CV ranking is unaffected
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    gs = GridSearchCV(KernelRidge(kernel=kernel),
                      {"alpha": alphas, "gamma": gammas},
                      scoring="neg_mean_absolute_error", cv=cv, n_jobs=n_jobs)
    gs.fit(X, yc)
    return float(gs.best_params_["alpha"]), float(gs.best_params_["gamma"]), float(-gs.best_score_)


def metrics(pred, y):
    err = y - pred
    return dict(pred=pred, MAE=float(mean_absolute_error(y, pred)),
                RMSE=float(np.sqrt(np.mean(err ** 2))),
                R2=float(r2_score(y, pred)), STD=float(np.std(err)),
                MAX=float(np.max(np.abs(err))))