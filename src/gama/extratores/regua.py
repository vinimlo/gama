# -*- coding: utf-8 -*-
"""A régua: o extrator por regras do baseline.

Não é a solução — é o número que o extrator neural precisa bater (1,09848 no dev
set, ajustado a ele; falha em 5 de 6 sondas de ruído documentado).
"""
from __future__ import annotations

from ..extrair import extrair
from ..span import Span
from .base import Extrator


class ExtratorRegua(Extrator):
    nome = "regua"

    def extrair(self, texto: str) -> list[Span]:
        return extrair(texto)
