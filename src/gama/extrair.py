# -*- coding: utf-8 -*-
"""Fachada (até a leva 8): a régua mora em `extratores/regua.py` (ExtratorRegua)."""
from __future__ import annotations

from .extratores.regua import ExtratorRegua
from .span import Span  # noqa: F401  (reexportado)


def extrair(texto: str) -> list:
    return ExtratorRegua().extrair(texto)
