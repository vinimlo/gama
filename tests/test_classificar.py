# -*- coding: utf-8 -*-
"""Classe e confiança a partir da resolução (D-002, D-006).

A tabela calibrada (calibracao.json) é trocada por uma fixa em cada teste: aqui se mede a
regra de decisão, não os números da calibração da vez.
"""
import pytest

from gama import classificar as mod
from gama.classificar import CONFIANCA, CONFIANCA_PADRAO, Citacao, classificar, faixa
from gama.resolver import Resolucao
from gama.span import Span

TABELA = {"processo|real|alta": 0.99, "processo|real_ambiguo|alta": 0.4, "vaga|incompleta|alta": 0.97}


@pytest.fixture(autouse=True)
def tabela_fixa(monkeypatch):
    monkeypatch.setattr(mod, "TABELA", dict(TABELA))


def _sp(conf=None, forma="processo"):
    return Span(3, 12, "REsp 1/SP", "jurisprudencia", forma, "", conf)


@pytest.mark.parametrize("conf,f", [(None, "regra"), (0.98, "alta"), (1.0, "alta"), (0.9799, "baixa"), (0.0, "baixa")])
def test_faixa(conf, f):
    assert faixa(conf) == f


def test_um_candidato_e_real_com_o_id():
    c = classificar(_sp(), Resolucao([123], "processo"))
    assert (c.classificacao, c.id_canonico, c.balde) == ("real", "123", ("processo", "real", "regra"))
    assert (c.inicio, c.fim, c.trecho, c.tipo) == (3, 12, "REsp 1/SP", "jurisprudencia")


def test_nenhum_candidato_e_inventada():
    c = classificar(_sp(), Resolucao([], "sumula"))
    assert (c.classificacao, c.id_canonico, c.balde) == ("inventada", None, ("sumula", "inventada", "regra"))


def test_vaga_e_incompleta_sem_id():
    c = classificar(_sp(forma="vaga"), Resolucao([], "vaga"))
    assert (c.classificacao, c.id_canonico, c.confianca) == ("incompleta", None, CONFIANCA[("vaga", "incompleta")])


def test_vaga_com_candidatos_continua_incompleta():
    c = classificar(_sp(), Resolucao(["x"], "vaga"))
    assert (c.classificacao, c.id_canonico) == ("incompleta", None)


@pytest.mark.parametrize("via", ["processo", "sumula", "dispositivo"])
def test_empate_vira_real_com_o_primeiro_e_confianca_baixa(via):
    """D-002: 2+ candidatos numa citação numerada -> real com o primeiro, no balde do empate."""
    c = classificar(_sp(), Resolucao(["b", "a"], via, 1))
    assert (c.classificacao, c.id_canonico) == ("real", "b")
    assert (c.balde, c.confianca) == (("processo", "real_ambiguo", "regra"), 0.50)


@pytest.mark.parametrize("via,classe,ids", [
    ("processo", "real", ["x"]), ("processo", "inventada", []),
    ("sumula", "real", ["x"]), ("sumula", "inventada", []),
    ("dispositivo", "real", ["x"]), ("dispositivo", "inventada", []),
])
def test_sem_calibracao_vale_o_prior(via, classe, ids):
    assert classificar(_sp(), Resolucao(ids, via)).confianca == CONFIANCA[(via, classe)]


def test_via_sem_prior_vale_o_padrao():
    assert classificar(_sp(), Resolucao(["x"], "outra")).confianca == CONFIANCA_PADRAO == 0.70


def test_calibracao_substitui_o_prior_no_balde_medido():
    assert classificar(_sp(0.99), Resolucao(["x"], "processo")).confianca == 0.99
    assert classificar(_sp(0.99), Resolucao(["x", "y"], "processo")).confianca == 0.4
    assert classificar(_sp(0.99, "vaga"), Resolucao([], "vaga")).confianca == 0.97
    # balde não medido: faixa baixa, ou via sem entrada na tabela
    assert classificar(_sp(0.5), Resolucao(["x"], "processo")).confianca == CONFIANCA[("processo", "real")]
    assert classificar(_sp(0.99), Resolucao([], "processo")).confianca == CONFIANCA[("processo", "inventada")]


def test_tabela_lida_a_cada_chamada(monkeypatch):
    monkeypatch.setattr(mod, "TABELA", {"processo|real|regra": 0.11})
    assert classificar(_sp(), Resolucao(["x"], "processo")).confianca == 0.11


def test_para_json():
    c = Citacao(3, 12, "REsp 1/SP", "jurisprudencia", "real", "123", 0.912345, ("processo", "real", "regra"))
    assert c.para_json(7) == {
        "id": "c7", "inicio": 3, "fim": 12, "trecho": "REsp 1/SP", "tipo": "jurisprudencia",
        "classificacao": "real", "resolucao": {"fonte": "jusbrasil", "id_canonico": "123"}, "confianca": 0.9123,
    }


def test_para_json_sem_id_nao_tem_resolucao():
    c = Citacao(0, 1, "x", "jurisprudencia", "inventada", None, 0.8)
    assert c.para_json(1)["resolucao"] is None


def test_balde_nao_entra_na_comparacao():
    a = Citacao(0, 1, "x", "lei", "real", "1", 0.9, ("a",))
    assert a == Citacao(0, 1, "x", "lei", "real", "1", 0.9, ("b",))
