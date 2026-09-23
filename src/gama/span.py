# -*- coding: utf-8 -*-
"""Span de citação — a moeda comum entre extratores, resolver e classificador."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    inicio: int
    fim: int
    trecho: str
    tipo: str        # jurisprudencia | lei
    forma: str       # cnj | processo | sumula | tema | artigo | vaga
    digitos: str = ""
    confianca: float | None = None   # probabilidade do extrator, quando houver
