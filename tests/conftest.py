"""Shared fixtures for the fast ML unit tests.

Guards: these tests must never load model weights. Importing torch / transformers / mlx_vlm raises, and
onnxruntime.InferenceSession is replaced with a stub that raises, so an accidental model load fails loudly
instead of competing with long CPU jobs.
"""
from __future__ import annotations

import importlib.abc
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

NORMS_PATH = os.path.join(ROOT, "app", "rules", "norms.yaml")
BLOCKED = ("torch", "transformers", "mlx_vlm", "mlx")


class _BlockHeavyML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ImportError(f"{fullname} is blocked in unit tests (no model loading)")
        return None


if not any(isinstance(f, _BlockHeavyML) for f in sys.meta_path):
    sys.meta_path.insert(0, _BlockHeavyML())


def _no_session(*a, **k):
    raise RuntimeError("onnxruntime.InferenceSession is blocked in unit tests (no model loading)")


try:
    import onnxruntime as _ort

    _ort.InferenceSession = _no_session
except ImportError:  # pragma: no cover
    pass


@pytest.fixture(scope="session")
def norms() -> dict:
    import yaml

    with open(NORMS_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
