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

from ..span import Span, aparar_todos, cruza, distancia

VAGA_COLADA = 2            # caracteres entre a VAGA e o span vizinho
CONFIANCA_MINIMA = 0.95


def confiante(s: Span, minimo: float) -> bool:
    return s.confianca is None or s.confianca >= minimo


def sem_vaga_colada(spans: list[Span], limite: int) -> list[Span]:
    """Regra 1: VAGA a até `limite` caracteres de outro span sai."""
    return [s for s in spans if s.forma != "vaga" or all(
        distancia(s, o) > limite for o in spans if o is not s)]


def trocar_fracos_pela_regua(spans: list[Span], regua: list[Span], minimo: float) -> list[Span]:
    """Regra 2: span com confiança < `minimo` sai; entra o span da régua que o cruza, se
    não cruzar um span confiante. Devolve ordenado por início."""
    fortes = [s for s in spans if confiante(s, minimo)]
    fracos = [s for s in spans if not confiante(s, minimo)]
    extra = [r for r in regua
             if any(cruza(r, w) for w in fracos) and not any(cruza(r, s) for s in fortes)]
    return sorted(fortes + extra, key=lambda s: s.inicio)


def guardar(spans: list[Span], regua: list[Span]) -> list[Span]:
    """Spans do modelo e da régua, já aparados -> spans que ficam, ordenados.

    A ordem das regras é a medida na D-008: a VAGA é comparada também com os spans fracos
    que a regra 2 vai tirar. Os limites são lidos a cada chamada: `bench.controles.avaliar`
    troca CONFIANCA_MINIMA durante a medida."""
    return trocar_fracos_pela_regua(sem_vaga_colada(spans, VAGA_COLADA), regua, CONFIANCA_MINIMA)


class ExtratorGuardado:
    """O extrator de produção: o modelo, com a guarda e a régua de reserva."""
    nome = "neural"

    def __init__(self, neural, regua):
        self.neural = neural
        self.regua = regua

    def extrair(self, texto: str) -> list[Span]:
        return guardar(aparar_todos(self.neural.extrair(texto), texto),
                       aparar_todos(self.regua.extrair(texto), texto))
