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


class DetectorDeForma:
    """Forma do span (sumula, tema, artigo, cnj, processo, vaga) pelo texto, e o Span
    montado a partir do rótulo do extrator."""
    _SEM_ANO = re.compile(r"\b(?:19|20)\d{2}\b")
    _RELATOR = re.compile(r"relatoria|rel\.\s*min|relator")
    _SUMULA = re.compile(r"\s*[s5]\s*[uú]\s*[mr]")              # Súmula, 5úmula, Súrnula
    # Tema, Temã, Terna — palavra inteira: "Terceiro AgR na Rcl" não é tema.
    _TEMA = re.compile(r"\s*te(?:m|rn)[aã]\b")
    _ARTIGO = re.compile(r"\s*art")
    _CNJ = re.compile(r"\d{1,7}\s*-{1,2}\s*\d{2}\s*[.\s]*\d{4}")

    def forma(self, trecho: str) -> str:
        t = trecho.lower()
        if self._RELATOR.search(t) and not re.search(r"\d", self._SEM_ANO.sub(" ", t)):
            return "vaga"
        if self._SUMULA.match(t):
            return "sumula"
        if self._TEMA.match(t):
            return "tema"
        if self._ARTIGO.match(t):
            return "artigo"
        if self._CNJ.search(t):
            return "cnj"
        return "processo"

    def span(self, texto: str, inicio: int, fim: int, rotulo: str, confianca: float | None = None) -> Span:
        """Monta um Span a partir de um rótulo do extrator (JURIS | LEI | VAGA)."""
        trecho = texto[inicio:fim]
        if rotulo == "VAGA":
            fm = "vaga"
        elif rotulo == "LEI":
            fm = "artigo"
        else:
            fm = self.forma(trecho)
            if fm in ("vaga", "artigo"):                   # o rótulo do modelo manda no tipo
                fm = "processo"
        tipo = "lei" if fm == "artigo" else "jurisprudencia"
        return Span(inicio, fim, trecho, tipo, fm, "", confianca)


# ------------------------------------------------------------------ fachadas (até a leva 8)

_DETECTOR = DetectorDeForma()


def forma(trecho: str) -> str:
    return _DETECTOR.forma(trecho)


def span_de(texto: str, inicio: int, fim: int, rotulo: str,
            confianca: float | None = None) -> Span:
    return _DETECTOR.span(texto, inicio, fim, rotulo, confianca)
