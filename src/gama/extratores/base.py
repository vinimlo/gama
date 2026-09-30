# -*- coding: utf-8 -*-
"""Contrato de um extrator: texto -> spans de citação tipados."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..span import Span


class Extrator(ABC):
    nome: str

    @abstractmethod
    def extrair(self, texto: str) -> list[Span]:
        """Spans sem sobreposição, ordenados por início, offsets em codepoints."""
