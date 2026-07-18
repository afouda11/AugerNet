import numpy as np
from pathlib import Path

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
    X = np.vstack(soap.create(systems, centers=centers, n_jobs=n_jobs))
    # y in data dict already selected with cidx 
    y = np.concatenate([d["y"] for d in data])
    natoms = np.concatenate([np.full(len(d["cidx"]), d["natoms"]) for d in data])
    names = np.concatenate([[d["name"]] * len(d["cidx"]) for d in data])
    return X, y, natoms, names

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
    return train_list, val_list 

