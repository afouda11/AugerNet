"""
AugerNet 
=============================================================

1) Molecular graph generation and GNN prediction of: 
  a) core-electron binding energy (CEBE) 
  b) Auger-Electron spectroscopy (AES) 

2) CNN classifications of local bond environments from AES+CEBE

Usage
-----
  # As a CLI:
  python -m augernet --config /path/to/config.yml
  
  # Results and models written to cwd

  # Programmatically:
  from augernet.config import load_config
  from augernet.train_driver import run
  cfg = load_config('configs/cebe_default.yml')
  run(cfg)
"""
import os

__version__ = '0.1.0'

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(os.path.dirname(PACKAGE_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DATA_RAW_DIR = os.path.join(DATA_DIR, 'raw')
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
