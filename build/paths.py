"""Configuration for the build stage.

config/rooms.json   the dataset definition (rooms, takes, exclusions, spans);
                    committed, machine-independent.
config/paths.json   where the raw data and the CHiME ATFs live on THIS machine;
                    not committed -- copy config/paths.example.json and edit.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / "config" / "rooms.json"))
_local = ROOT / "config" / "paths.json"
if not _local.exists():
    raise FileNotFoundError(f"{_local} is missing: copy config/paths.example.json and set raw_root "
                            f"and chime_atf_pkl for this machine")
CFG.update(json.load(open(_local)))
RAW = Path(CFG["raw_root"])
