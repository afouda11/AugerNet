"""
AugerNet CLI Entry Point
========================

Usage:
  python -m augernet --config /path/to/config.yml

All options and parameters defined in user config.yml file, see README.md

"""

from __future__ import annotations

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        prog='augernet',
        description='AugerNet — training, evaluation and predictions of GNNs for CEBEs.\n'
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
