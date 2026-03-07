"""
AugerNet — Machine Learning for Auger-Decay Spectroscopy
=========================================================

Provides:
  • GNN models for core-electron binding energy (CEBE) prediction
  • GNN models for Auger spectrum prediction (stick + fitted)
  • CNN models for carbon environment classification

Usage
-----
  # As a CLI:
  python -m augernet --model cebe-gnn --mode train --config configs/cebe_default.yml

  # Programmatically:
  from augernet.config import load_config
  from augernet.train_driver import run
  cfg = load_config('configs/cebe_default.yml')
  run(cfg)
"""
import os

__version__ = '0.1.0'

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')