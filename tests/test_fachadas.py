# -*- coding: utf-8 -*-
"""Nomes antigos que o laboratório (bench/, treino/) ainda importa, vivos até a leva 8.

Cada fachada só delega: aqui se confere que ela chega à classe com os mesmos limites.
Sai inteiro junto com as fachadas, quando os consumidores migrarem.
"""
from gama import pipeline, span
from gama.extratores import carregar, guarda
from gama.extratores.regua import ExtratorRegua
from gama.span import Span


def _em(inicio, fim, forma="processo", conf=1.0):
    return Span(inicio, fim, "x" * (fim - inicio), "jurisprudencia", forma, "", conf)


def test_span():
    a, b = _em(0, 10), _em(12, 20)
    assert (span.cruza(a, b), span.distancia(a, b)) == (False, 2)
    t = " REsp. "
    s = Span(0, 7, t, "jurisprudencia", "processo")
    assert span.aparar(s, t) == s.aparado(t) == pipeline.aparar(s, t)
    assert span.aparar_todos([s], t) == Span.aparar_todos([s], t)


def test_guarda(monkeypatch):
    fraco, vaga = _em(0, 10, conf=0.7), _em(12, 30, forma="vaga")
    regua = [_em(0, 12, conf=None)]
    assert guarda.confiante(fraco, 0.5) and not guarda.confiante(fraco, 0.95)
    assert guarda.sem_vaga_colada([fraco, vaga], 2) == [fraco]
    assert guarda.sem_vaga_colada([fraco, vaga], 1) == [fraco, vaga]
    assert guarda.trocar_fracos_pela_regua([fraco], regua, 0.95) == regua
    assert guarda.trocar_fracos_pela_regua([fraco], regua, 0.5) == [fraco]
    assert guarda.guardar([fraco], regua) == regua
    monkeypatch.setattr(guarda, "CONFIANCA_MINIMA", 0.5)
    assert guarda.guardar([fraco], regua) == [fraco]


def test_carregar():
    assert isinstance(carregar("regua"), ExtratorRegua)
