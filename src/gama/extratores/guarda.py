# -*- coding: utf-8 -*-
"""Guarda do extrator neural para texto fora do molde da organização.

Duas regras, medidas em `wiki/experimentos/2026-09-23_otimizacao-medida.md`:

1. VAGA colada a outro span sai. Na redação da organização a referência vaga é uma frase
   própria ("julgado do STJ proferido em 2021 pela relatoria de ..."), depois de fim de
   frase: a mais próxima de outra citação está a 4 caracteres (". O "). Em ementa real,
   "Rel. Min. Fulano, julgado em ..." colado a um precedente numerado é o rabo dele.
2. Span com confiança < 0,95 sai; no lugar entra o span da régua que o cruza, se houver e se
   não cruzar um span confiante. Nenhum acerto no estilo da organização fica abaixo de 0,98
   (34.171 spans em dev, estresse e final_v1); os fragmentos de ementa real ("Rel", "2011",
   ")") ficam, com confiança mediana entre 0,85 e 0,94.

No estilo da organização as duas regras não mudam nenhum documento (4.626 medidos); em 305
ementas reais, o F1 de extração vai de 0,605 a 0,808. Diferente da união (D-007), a régua
só entra onde o modelo hesita, e no molde ele não hesita.
"""
from __future__ import annotations

from ..span import Span
from .neural import ExtratorNeural
from .regua import ExtratorRegua

VAGA_COLADA = 2            # caracteres entre a VAGA e o span vizinho
CONFIANCA_MINIMA = 0.95


def _confiante(s: Span) -> bool:
    return s.confianca is None or s.confianca >= CONFIANCA_MINIMA


def _cruza(a: Span, b: Span) -> bool:
    return a.inicio < b.fim and b.inicio < a.fim


def guardar(spans: list[Span], regua: list[Span]) -> list[Span]:
    """Spans do modelo e da régua, já aparados -> spans que ficam, ordenados."""
    spans = [s for s in spans if s.forma != "vaga" or all(
        max(o.inicio - s.fim, s.inicio - o.fim) > VAGA_COLADA for o in spans if o is not s)]
    fortes = [s for s in spans if _confiante(s)]
    fracos = [s for s in spans if not _confiante(s)]
    extra = [r for r in regua
             if any(_cruza(r, w) for w in fracos) and not any(_cruza(r, s) for s in fortes)]
    return sorted(fortes + extra, key=lambda s: s.inicio)


class ExtratorGuardado:
    """O extrator de produção: o modelo, com a guarda e a régua de reserva."""
    nome = "neural"

    def __init__(self, pasta):
        self.neural = ExtratorNeural(pasta)
        self.regua = ExtratorRegua()

    def extrair(self, texto: str) -> list[Span]:
        from ..pipeline import aparar              # tardio: o pipeline importa os extratores

        def aparados(ss):
            return [s for s in (aparar(x, texto) for x in ss) if s]
        return guardar(aparados(self.neural.extrair(texto)), aparados(self.regua.extrair(texto)))
