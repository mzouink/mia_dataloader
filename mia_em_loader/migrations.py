"""Version migrations for the crop database JSON format.

Each migration transforms a dict from version N to version N+1.
The chain is always forward-only. Add new migrations at the bottom.

To add a new migration:
    1. Write a function ``_migrate_vN_to_vN1(data: dict) -> dict``
    2. Add it to ``_MIGRATIONS`` with key N
    3. Bump ``CURRENT_VERSION``
"""

from __future__ import annotations

import logging
from typing import Callable, Dict

logger = logging.getLogger(__name__)

CURRENT_VERSION = 2


def _migrate_v1_to_v2(data: dict) -> dict:
    """Move top-level ``norms`` into each crop's ``norm_params``.

    v1 had a redundant top-level ``norms`` dict keyed by dataset name,
    plus ``norm_params`` already embedded in each crop.  v2 drops the
    top-level ``norms`` — each crop is fully self-contained.

    For v1 files where crops might lack ``norm_params`` (shouldn't happen,
    but be defensive), we inject from the top-level ``norms``.
    """
    norms = data.pop("norms", {})
    for crop in data.get("crops", []):
        if "norm_params" not in crop or crop["norm_params"] is None:
            dataset = crop.get("dataset_name", "")
            if dataset in norms:
                crop["norm_params"] = norms[dataset]
            else:
                logger.warning(
                    f"Crop {dataset}/{crop.get('crop_id')} has no norm_params "
                    f"and dataset not found in top-level norms during migration"
                )
    data["version"] = 2
    return data


# Registry: version -> function that upgrades dict from that version to the next
_MIGRATIONS: Dict[int, Callable[[dict], dict]] = {
    1: _migrate_v1_to_v2,
}


def migrate(data: dict) -> dict:
    """Apply all necessary migrations to bring *data* to CURRENT_VERSION.

    Modifies and returns *data* in place.
    """
    v = data.get("version", 1)
    if v > CURRENT_VERSION:
        raise ValueError(
            f"Database version {v} is newer than supported ({CURRENT_VERSION}). "
            f"Please upgrade mia_em_loader."
        )
    while v < CURRENT_VERSION:
        fn = _MIGRATIONS.get(v)
        if fn is None:
            raise ValueError(
                f"No migration from version {v} to {v + 1}. "
                f"This is a bug — please report it."
            )
        logger.info(f"Migrating crop database v{v} -> v{v + 1}")
        data = fn(data)
        v += 1
    data["version"] = CURRENT_VERSION
    return data
