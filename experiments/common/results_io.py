"""Cached / resumable result IO. Writes one row at a time so failed runs
can be resumed without recomputing finished rows."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable, Mapping


def append_csv_row(path: str | os.PathLike, row: Mapping, fieldnames: Iterable[str]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def existing_keys(path: str | os.PathLike, key_cols: Iterable[str]) -> set[tuple]:
    path = Path(path)
    if not path.exists():
        return set()
    keys = set()
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            keys.add(tuple(row[k] for k in key_cols))
    return keys


def save_json(path: str | os.PathLike, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | os.PathLike, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)
