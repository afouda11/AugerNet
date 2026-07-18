import json
from pathlib import Path
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split

from soap_krr_utils import (
    detect_atom_types, soap_input_and_be_output, metrics
    )

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

def _load_data_from_json(json_path):
    with open(json_path, 'r') as f:
        data_json = json.load(f)
    data = []
    for item in data_json:
        symbol, pos, be = _parse_xyz_and_be_from_string(item['xyz_file'])
        cidx = [i for i, (s, b) in enumerate(zip(symbol, be)) if s == "C" and not np.isnan(b)]
        if cidx:
            data.append(dict(name=item['name'], symbols=symbol, positions=pos,
                             cidx=cidx, y=be[cidx], natoms=len(symbol)))
    return data

# Values from J. Chem. Phys. 162, 244101 (2025) 
SOAP_PARAMS = dict(r_cut=8.0, n_max=8, l_max=4, sigma=0.05)
KRR_PARAMS  = dict(kernel="rbf", alpha=1e-11, gamma=0.25)
TEST_FRAC   = 0.3

# OMP
N_JOBS = 4
#Split seed
SEED = 42

# Read data from json and make data dict
data = _load_data_from_json('photoemission_v1_ML-C1S-XYZ.json')

# Data info
atom_types = detect_atom_types(data)
n_c = sum(len(d["cidx"]) for d in data)
print(f"Loaded {len(data)} molecules: {n_c} carbons and atom types: {atom_types}")

# Generate X SOAP matrix and y BE vector
X, y, *_ = soap_input_and_be_output(atom_types, SOAP_PARAMS, data, N_JOBS)
print("Built SOAP input descriptors")
#Random 70/30 test train split
Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_FRAC, shuffle=True, random_state=SEED)

print(f"Data shuffled and split with {TEST_FRAC} test fraction random seed: {SEED}")
print(f"Running KRR model with:\n", 
      f"\t kernel: {KRR_PARAMS['kernel']}\n",
      f"\t gamma: {KRR_PARAMS['gamma']}\n",
      f"\t alpha: {KRR_PARAMS['alpha']}\n")

mu = ytr.mean()
model = KernelRidge(kernel=KRR_PARAMS["kernel"], alpha=KRR_PARAMS["alpha"],
                        gamma=KRR_PARAMS["gamma"]).fit(Xtr, ytr-mu)

results = metrics(model.predict(Xte)+mu, yte)

print(f"Test-set MAE = {results['MAE']:.3f} eV  |  RMSE {results['RMSE']:.3f}  R2 {results['R2']:.3f}  "
          f"MAX {results['MAX']:.3f}   (Porcelli et al. reported 0.12 eV)\n")

print("Agreement won't be exact due to random test/train split seed value difference.\n")