"""
Checkpoint sidecars — normalisation constants and build provenance
==================================================================

A trained checkpoint is not self-describing.  The constants used to normalise
its targets and inputs are fitted on the training molecules of one fold, and the
settings used to build those inputs live in a YAML that may since have changed.
``evaluate`` and ``predict`` have no training split from which to re-derive
either.

Each backend therefore writes ``{model_stem}_norm.json`` beside the ``.pth`` at
train time and reads it back at inference time.  A missing sidecar is an error:
there is deliberately no dataset-wide fallback, because normalising or
re-broadening with constants the model was not trained against yields
plausible-looking numbers that are silently wrong.

This module owns only the mechanics — path, read, write, comparison.  What goes
*into* a sidecar is each backend's business:

    backend_gnn : CEBE mean/std, Auger maxI, node-feature stats, spectrum grid
    backend_cnn : delta_be mean/std, CarbonDataset build params, architecture

Kept in one place so the two backends cannot drift apart on the file naming or
on how a mismatch is reported.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  Path / read / write
# ─────────────────────────────────────────────────────────────────────────────

def norm_sidecar_path(model_path: str) -> str:
    """Path of the sidecar belonging to *model_path*.

    ``/…/model_fold3.pth`` -> ``/…/model_fold3_norm.json``
    """
    return f"{os.path.splitext(model_path)[0]}_norm.json"


def to_jsonable(value: Any) -> Any:
    """Recursively convert tuples to lists.

    ``ARCHITECTURE_PRESETS`` uses tuples; ``json.dump`` writes them as arrays and
    ``json.load`` returns lists, so ``(5, 10, 15) != [5, 10, 15]`` would read as
    a config mismatch on every single load.  Normalising both sides through this
    makes the round-trip compare equal.
    """
    if isinstance(value, (tuple, list)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    return value


def save_norm_sidecar(model_path: str, norm: Dict[str, Any]) -> str:
    """Write *norm* beside *model_path*.  Returns the path written."""
    path = norm_sidecar_path(model_path)
    with open(path, 'w') as fh:
        json.dump(to_jsonable(norm), fh, indent=2)
    return path


def load_norm_sidecar(model_path: str, *,
                      require: Sequence[str] = ()) -> Dict[str, Any]:
    """Load the sidecar for *model_path*, or raise.

    Parameters
    ----------
    require : sequence of str
        Top-level keys that must be present, so a sidecar written by a
        different backend (or an older version) fails loudly rather than
        halfway through inference.
    """
    path = norm_sidecar_path(model_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Normalization stats for the given fold are not available.\n"
            f"  Expected sidecar: {path}\n"
            f"  Model:            {model_path}\n"
            f"  The constants are fitted on the training molecules of the fold "
            f"and written beside the checkpoint at train time.  They cannot be "
            f"reconstructed from the model alone, and there is no dataset-wide "
            f"fallback.  Re-train the fold, or supply its "
            f"{os.path.basename(path)}."
        )
    with open(path) as fh:
        norm = json.load(fh)

    missing = [k for k in require if k not in norm]
    if missing:
        raise ValueError(
            f"Malformed normalisation sidecar: {path}\n"
            f"  Missing required block(s): {', '.join(missing)}\n"
            f"  Present: {', '.join(sorted(norm)) or '(empty)'}\n"
            f"  This usually means the sidecar belongs to a different model "
            f"type, or predates the block being asked for."
        )
    return norm


# ─────────────────────────────────────────────────────────────────────────────
#  Config-vs-checkpoint comparison
# ─────────────────────────────────────────────────────────────────────────────

def collect_mismatches(
    pairs: Iterable[Tuple[str, Any, Any]],
) -> List[str]:
    """Compare ``(label, config_value, checkpoint_value)`` triples.

    A ``checkpoint_value`` of ``None`` is skipped — that setting was not
    recorded by this sidecar, so there is nothing to check against and an older
    sidecar does not start failing.  Both sides go through ``to_jsonable`` so a
    tuple in the config matches the list it was serialised as.
    """
    problems: List[str] = []
    for label, config_value, checkpoint_value in pairs:
        if checkpoint_value is None:
            continue
        if to_jsonable(config_value) != to_jsonable(checkpoint_value):
            problems.append(
                f"  {label}: config {config_value!r} "
                f"vs checkpoint {checkpoint_value!r}"
            )
    return problems


def raise_on_mismatch(problems: Sequence[str], *, model_path: str,
                      context: str = 'config') -> None:
    """Raise a single collected error, or return quietly if there are none."""
    if not problems:
        return
    raise ValueError(
        f"{context} does not match the checkpoint it is being run with:\n"
        + "\n".join(problems)
        + f"\n\nThe checkpoint was trained with these settings; running it with "
          f"others\nproduces silently wrong values.  Correct the YAML, or point "
          f"model_path at a\ncheckpoint trained the way this config describes.\n"
          f"  Sidecar: {norm_sidecar_path(model_path)}"
    )
