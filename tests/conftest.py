"""Pytest configuration shared by all test modules.

Makes the repository root importable so ``import pso3d`` works from a plain checkout
(the package is also installable with ``pip install -e .``).
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
