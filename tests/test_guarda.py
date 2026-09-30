# -*- coding: utf-8 -*-
"""Guarda do extrator neural: as duas regras, com spans montados à mão."""
from gama.extratores.guarda import guardar
from gama.span import Span


def _s(texto, trecho, tipo="jurisprudencia", forma="processo", conf=1.0, apos=0):
    a = texto.index(trecho, apos)
    return Span(a, a + len(trecho), trecho, tipo, forma, "", conf)


def test_vaga_colada_a_precedente_sai():
    t = "Precedentes: REsp 1.234.567/SP, Rel. Min. Nancy Andrighi, julgado em 2021."
    juris = _s(t, "REsp 1.234.567/SP")
    vaga = _s(t, "Rel. Min. Nancy Andrighi, julgado em 2021", forma="vaga", conf=0.99)
    assert guardar([juris, vaga], []) == [juris]


def test_vaga_depois_de_fim_de_frase_fica():
    """No molde da organização a vaga é frase própria: a mais próxima está a 4 caracteres."""
    t = "Aplica-se o REsp 1.307.026/BA.\nO acórdão do STF julgado em 2024 sob relatoria de Cristiano Zanin."
    juris = _s(t, "REsp 1.307.026/BA")
    vaga = _s(t, "acórdão do STF julgado em 2024 sob relatoria de Cristiano Zanin", forma="vaga")
    assert guardar([juris, vaga], []) == [juris, vaga]


def test_span_inseguro_da_lugar_a_regua():
    t = "Nos termos do art. 927, parágrafo único, do Código Civil."
    fragmento = _s(t, "927", tipo="lei", forma="artigo", conf=0.83)
    regua = _s(t, "art. 927, parágrafo único, do Código Civil", tipo="lei", forma="artigo", conf=None)
    assert guardar([fragmento], [regua]) == [regua]


def test_span_inseguro_sem_regua_sai():
    t = "DJe 19/08/2019."
    assert guardar([_s(t, "DJe 19/08/2019", conf=0.7)], []) == []


def test_regua_nao_entra_onde_o_modelo_tem_certeza():
    t = "Ver REsp 1.234.567/SP e o fragmento solto 2011."
    forte = _s(t, "REsp 1.234.567/SP")
    fraco = _s(t, "2011", conf=0.6)
    regua_forte = _s(t, "REsp 1.234.567", conf=None)
    assert guardar([forte, fraco], [regua_forte]) == [forte]


def test_confianca_no_limite_fica():
    t = "Súmula 7 do STJ."
    s = _s(t, "Súmula 7 do STJ", forma="sumula", conf=0.95)
    assert guardar([s], []) == [s]
