# -*- coding: utf-8 -*-
"""Span de citação — a moeda comum entre extratores, resolver e classificador."""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Span:
    inicio: int
    fim: int
    trecho: str
    tipo: str        # jurisprudencia | lei
    forma: str       # cnj | processo | sumula | tema | artigo | vaga
    digitos: str = ""
    confianca: float | None = None   # probabilidade do extrator, quando houver


def cruza(a: Span, b: Span) -> bool:
    """Os dois dividem ao menos um caractere."""
    return a.inicio < b.fim and b.inicio < a.fim


def distancia(a: Span, b: Span) -> int:
    """Caracteres entre o fim de um e o início do outro; negativa quando se cruzam."""
    return max(b.inicio - a.fim, a.inicio - b.fim)


def aparar(span: Span, texto: str) -> Span | None:
    """Borda na convenção do gabarito para QUALQUER extrator (a régua também): sem espaço
    Unicode nas pontas, sem .,;: no fim (revisão independente, rodada 2, achado 1). None se sobrar nada."""
    a, b = span.inicio, span.fim
    while a < b and texto[a].isspace():
        a += 1
    while b > a and (texto[b - 1].isspace() or texto[b - 1] in ".,;:"):
        b -= 1
    if b <= a:
        return None
    return span if (a, b) == (span.inicio, span.fim) else replace(span, inicio=a, fim=b, trecho=texto[a:b])


def aparar_todos(spans, texto: str) -> list[Span]:
    """`aparar` em cada span; os que ficam vazios saem."""
    return [s for s in (aparar(x, texto) for x in spans) if s]
