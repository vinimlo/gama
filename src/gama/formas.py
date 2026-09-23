# -*- coding: utf-8 -*-
"""Roteamento de um span para o caminho de resolução certo.

O extrator neural devolve só o span e o rótulo (JURIS, LEI, VAGA). O resolver
precisa saber se é súmula, tema, CNJ, número clássico ou artigo — cada um resolve
por um índice diferente. Isto decide a partir do TEXTO do span, que já foi
delimitado pelo extrator: não procura citação no documento.
"""
from __future__ import annotations

import re

from .span import Span


def forma(trecho: str) -> str:
    t = trecho.lower()
    sem_ano = re.sub(r"\b(?:19|20)\d{2}\b", " ", t)
    if re.search(r"relatoria|rel\.\s*min|relator", t) and not re.search(r"\d", sem_ano):
        return "vaga"
    if re.match(r"\s*[s5]\s*[uú]\s*[mr]", t):          # Súmula, 5úmula, Súrnula
        return "sumula"
    # Tema, Temã, Terna — palavra inteira: "Terceiro AgR na Rcl" não é tema.
    if re.match(r"\s*te(?:m|rn)[aã]\b", t):
        return "tema"
    if re.match(r"\s*art", t):
        return "artigo"
    if re.search(r"\d{1,7}\s*-{1,2}\s*\d{2}\s*[.\s]*\d{4}", t):
        return "cnj"
    return "processo"


def span_de(texto: str, inicio: int, fim: int, rotulo: str,
            confianca: float | None = None) -> Span:
    """Monta um Span a partir de um rótulo do extrator (JURIS | LEI | VAGA)."""
    trecho = texto[inicio:fim]
    if rotulo == "VAGA":
        fm = "vaga"
    elif rotulo == "LEI":
        fm = "artigo"
    else:
        fm = forma(trecho)
        if fm in ("vaga", "artigo"):                   # o rótulo do modelo manda no tipo
            fm = "processo"
    tipo = "lei" if fm == "artigo" else "jurisprudencia"
    return Span(inicio, fim, trecho, tipo, fm, "", confianca)
