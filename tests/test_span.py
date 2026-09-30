# -*- coding: utf-8 -*-
"""Span: posição relativa entre spans, confiança e aparo de borda."""
import pytest

from gama.span import Span


def _em(inicio, fim, conf=None):
    return Span(inicio, fim, "x" * (fim - inicio), "jurisprudencia", "processo", "", conf)


def _s(texto, trecho, conf=None):
    a = texto.index(trecho)
    return Span(a, a + len(trecho), trecho, "jurisprudencia", "processo", "", conf)


@pytest.mark.parametrize("a,b,cruza", [
    ((0, 10), (5, 15), True),
    ((0, 10), (0, 10), True),
    ((0, 10), (2, 4), True),                # um dentro do outro
    ((0, 10), (10, 20), False),             # encostar não é cruzar
    ((0, 10), (11, 20), False),
])
def test_cruza_e_simetrico(a, b, cruza):
    x, y = _em(*a), _em(*b)
    assert x.cruza(y) == y.cruza(x) == cruza


@pytest.mark.parametrize("a,b,distancia", [
    ((0, 10), (12, 20), 2),
    ((0, 10), (10, 20), 0),
    ((0, 10), (5, 20), -5),                 # negativa quando se cruzam
])
def test_distancia_e_simetrica(a, b, distancia):
    x, y = _em(*a), _em(*b)
    assert x.distancia(y) == y.distancia(x) == distancia


@pytest.mark.parametrize("conf,fica", [(None, True), (0.95, True), (0.9499, False), (1.0, True)])
def test_confiante(conf, fica):
    assert _em(0, 1, conf).confiante(0.95) == fica


def test_aparado_devolve_o_mesmo_span_quando_nada_muda():
    t = "Ver REsp 1/SP."
    s = _s(t, "REsp 1/SP")
    assert s.aparado(t) is s


@pytest.mark.parametrize("bruto,aparado", [
    (" REsp 1/SP;", "REsp 1/SP"),                 # NBSP no início, ; no fim
    ("REsp 1/SP. ", "REsp 1/SP"),                 # espaço largo depois do ponto
    ("REsp 1/SP ,:", "REsp 1/SP"),
])
def test_aparado_tira_espaco_unicode_e_pontuacao_final(bruto, aparado):
    t = f"Ver {bruto} depois"
    s = _s(t, bruto).aparado(t)
    assert (s.trecho, t[s.inicio:s.fim]) == (aparado, aparado)


def test_aparado_preserva_os_outros_campos():
    t = "Ver  REsp 1/SP. depois"
    s = Span(3, 15, t[3:15], "jurisprudencia", "processo", "1", 0.97).aparado(t)
    assert (s.tipo, s.forma, s.digitos, s.confianca) == ("jurisprudencia", "processo", "1", 0.97)


def test_aparado_vazio_e_none():
    t = "Ver ;  REsp."
    assert _s(t, " ;  ").aparado(t) is None


def test_aparar_todos_tira_os_vazios():
    t = "Ver ;  REsp 1/SP."
    juris = _s(t, "REsp 1/SP.")
    assert Span.aparar_todos([_s(t, " ;  "), juris], t) == [juris.aparado(t)]
