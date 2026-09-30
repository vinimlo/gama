# -*- coding: utf-8 -*-
"""Forma de um span pelo texto, e o Span montado a partir do rótulo do extrator."""
import pytest

from gama.formas import forma, span_de


@pytest.mark.parametrize("trecho,f", [
    ("julgado do STJ de 2021, relatoria de Nancy Andrighi", "vaga"),
    ("Rel. Min. Luis Felipe Salomão, julgado em 12/2020", "processo"),   # há número além do ano
    ("precedente da Terceira Turma, Rel. Min. X", "vaga"),
    ("relator o Ministro X, em 2019", "vaga"),
    ("Súmula 7 do STJ", "sumula"), ("5úmula 7", "sumula"), ("Súrnula 7", "sumula"), ("  súmula 7", "sumula"),
    ("Tema 725", "tema"), ("Terna 725", "tema"), ("Temã 725", "tema"),
    ("Terceiro AgR na Rcl 1", "processo"),                             # "te" + "rc": não é tema
    ("Temas 725", "processo"),                                         # palavra inteira
    ("art. 5º da CF", "artigo"), ("Artigo 373 do CPC", "artigo"),
    ("AI 0603026-69.2018.6.09.0000", "cnj"), ("RR 1835 -- 06 . 2010", "cnj"), ("7000101-61 2019", "cnj"),
    ("REsp 1.234.567/SP", "processo"),
    ("REsp 1.234.567/SP, Rel. Min. X", "processo"),
])
def test_forma(trecho, f):
    assert forma(trecho) == f


TEXTO = "Ver REsp 1/SP e o art. 5º da CF, e Súmula 7, julgado do STJ, relatoria de X."


def _em(trecho):
    a = TEXTO.index(trecho)
    return a, a + len(trecho)


@pytest.mark.parametrize("trecho,rotulo,f,tipo", [
    ("REsp 1/SP", "JURIS", "processo", "jurisprudencia"),
    ("Súmula 7", "JURIS", "sumula", "jurisprudencia"),
    ("art. 5º da CF", "JURIS", "processo", "jurisprudencia"),        # o rótulo manda no tipo
    ("julgado do STJ, relatoria de X", "JURIS", "processo", "jurisprudencia"),
    ("art. 5º da CF", "LEI", "artigo", "lei"),
    ("REsp 1/SP", "LEI", "artigo", "lei"),
    ("julgado do STJ, relatoria de X", "VAGA", "vaga", "jurisprudencia"),
    ("REsp 1/SP", "VAGA", "vaga", "jurisprudencia"),
])
def test_span_de(trecho, rotulo, f, tipo):
    a, b = _em(trecho)
    s = span_de(TEXTO, a, b, rotulo, 0.9)
    assert (s.inicio, s.fim, s.trecho, s.forma, s.tipo, s.digitos, s.confianca) == (a, b, trecho, f, tipo, "", 0.9)


def test_span_de_sem_confianca():
    assert span_de(TEXTO, 4, 13, "JURIS").confianca is None
