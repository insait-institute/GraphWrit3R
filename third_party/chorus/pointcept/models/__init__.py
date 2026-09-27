"""Minimal model registry for vendored Chorus inference."""

from .builder import build_model
from .default import LangPretrainerMultiTeacher
from .point_transformer_v3 import *

__all__ = ["build_model", "LangPretrainerMultiTeacher"]

