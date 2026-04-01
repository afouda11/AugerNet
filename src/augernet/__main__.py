"""
AugerNet 
=============================================================

1) Molecular graph generation and GNN prediction of: 
a) core-electron binding energy (CEBE) 
and
b) Auger-Electron spectroscopy (AES) 

2) CNN local bond environment classification from AES+CEBE

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

from __future__ import annotations

import argparse

def main():
    parser = argparse.ArgumentParser(
        prog='augernet',
        description='AugerNet — training, evaluation and predictions of:\n'
                    '1) GNNs predictions of: \n'
                    '  a) core-electron binding energy (CEBE) \n'
                    '  b) Auger-Electron spectroscopy (AES) \n'
                    '2) CNN local bond environment classification from AES+CEBE\n'
                    '\n'
                    'Modes: cv | train | param | evaluate | predict',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--config', '-c', type=str, default=None, required=True,
        help='Path to a YAML config file (e.g. /path/to/config.yml)',
    )

    args = parser.parse_args() 

    # Load config 
    from augernet.config import load_config
    cfg = load_config(args.config)

    # Run
    from augernet.train_driver import run
    run(cfg)

if __name__ == '__main__':
    main()
