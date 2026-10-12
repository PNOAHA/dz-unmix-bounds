# -*- coding: utf-8 -*-
"""
Constants-only STUB of dz-unmix-cg/dz_unmixing_experiment.py.

Purpose: let bayesian_realdata.py import (DEFAULT_SOURCES, T_GRID) on the
air-gapped L20 container WITHOUT uploading the full first-paper core (which pulls
sklearn). Values MIRROR dz-unmix-cg exactly — that file is NOT modified.

Used only by validate_gpu.py via:  DZ_CORE_DIR=<this folder> python validate_gpu.py
Kept in a subfolder so it never shadows the real core in local runs.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Source:
    name: str
    mean_age: float
    sigma: float
    geo_label: str


# mirror of dz-unmix-cg DEFAULT_SOURCES (S1 130/15, S2 280/30, S3 1900/100)
DEFAULT_SOURCES = [
    Source("S1_MesozoicArc", 130.0, 15.0, "MesozoicArc"),
    Source("S2_PaleozoicBase", 280.0, 30.0, "PaleozoicBase"),
    Source("S3_ArcheanCraton", 1900.0, 100.0, "ArcheanCraton"),
]

# mirror of dz-unmix-cg T_GRID: 50..2500 Ma, step 5 (imported by bayesian_realdata; unused in sampler)
T_GRID = np.arange(50.0, 2500.0 + 1e-9, 5.0)
