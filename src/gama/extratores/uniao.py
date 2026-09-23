# -*- coding: utf-8 -*-
"""União: spans do extrator neural + spans da régua que o modelo deixou de fora.

A régua tem precisão alta nas formas conhecidas (100% no dev; é a checagem "citação
sem rótulo" que o goldenset inteiro passou); o modelo cobre forma nova e ruído. Onde
os dois se cruzam, vale o do modelo — borda e tipo aprendidos. Se a união ganha do
neural sozinho é pergunta da validação por dobras (métrica oficial), não suposição.
"""
from __future__ import annotations

from ..span import Span
from .neural import ExtratorNeural
from .regua import ExtratorRegua


class ExtratorUniao:
    def __init__(self, pasta):
        self.neural = ExtratorNeural(pasta)
        self.regua = ExtratorRegua()

    def extrair(self, texto: str) -> list[Span]:
        spans = list(self.neural.extrair(texto))
        for r in self.regua.extrair(texto):
            if not any(r.inicio < s.fim and s.inicio < r.fim for s in spans):
                spans.append(r)
        return sorted(spans, key=lambda s: s.inicio)
