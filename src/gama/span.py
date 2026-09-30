# -*- coding: utf-8 -*-
"""Span de citação — a moeda comum entre extratores, resolver e classificador."""
from __future__ import annotations

from dataclasses import dataclass, replace

# Nenhum span do gabarito termina em .,;: (192/192, wiki/conceitos/convencoes-de-borda.md).
PONTUACAO_FINAL = ".,;:"


class Intervalo:
    """Posição no texto: quem herda tem `inicio` e `fim` (fim exclusivo, em codepoints)."""

    def cruza(self, outro: Intervalo) -> bool:
        """Os dois dividem ao menos um caractere; encostar (fim == início) não é cruzar."""
        return self.inicio < outro.fim and outro.inicio < self.fim

    def distancia(self, outro: Intervalo) -> int:
        """Caracteres entre o fim de um e o início do outro; negativa quando se cruzam."""
        return max(outro.inicio - self.fim, self.inicio - outro.fim)

    def iou(self, outro: Intervalo) -> float:
        """Interseção sobre união, em caracteres; 0 quando não se cruzam."""
        i = max(0, min(self.fim, outro.fim) - max(self.inicio, outro.inicio))
        return i / (max(self.fim, outro.fim) - min(self.inicio, outro.inicio)) if i else 0.0


@dataclass(frozen=True)
class Span(Intervalo):
    inicio: int
    fim: int
    trecho: str
    tipo: str        # jurisprudencia | lei
    forma: str       # cnj | processo | sumula | tema | artigo | vaga
    digitos: str = ""
    confianca: float | None = None   # probabilidade do extrator, quando houver

    def confiante(self, minimo: float) -> bool:
        """Sem confiança (a régua) conta como confiante."""
        return self.confianca is None or self.confianca >= minimo

    def aparado(self, texto: str) -> Span | None:
        """Borda na convenção do gabarito para QUALQUER extrator (a régua também): sem espaço
        Unicode nas pontas, sem .,;: no fim (revisão independente, rodada 2, achado 1). None se
        sobrar nada; o próprio span, se nada mudar."""
        a, b = self.inicio, self.fim
        while a < b and texto[a].isspace():
            a += 1
        while b > a and (texto[b - 1].isspace() or texto[b - 1] in PONTUACAO_FINAL):
            b -= 1
        if b <= a:
            return None
        return self if (a, b) == (self.inicio, self.fim) else replace(self, inicio=a, fim=b, trecho=texto[a:b])

    @staticmethod
    def aparar_todos(spans, texto: str) -> list[Span]:
        """`aparado` em cada span; os que ficam vazios saem."""
        return [s for s in (x.aparado(texto) for x in spans) if s]


# ------------------------------------------------------------------ fachadas (até a leva 8)
# Nomes antigos, ainda importados pelo laboratório (bench/, treino/).

def cruza(a: Span, b: Span) -> bool:
    return a.cruza(b)


def distancia(a: Span, b: Span) -> int:
    return a.distancia(b)


def aparar(span: Span, texto: str) -> Span | None:
    return span.aparado(texto)


def aparar_todos(spans, texto: str) -> list[Span]:
    return Span.aparar_todos(spans, texto)
