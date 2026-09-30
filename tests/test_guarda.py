# -*- coding: utf-8 -*-
"""Guarda do extrator neural: as duas regras, com spans montados à mão.

Só pelo contrato público — `guardar(spans, regua)`, os limites do módulo e o extrator que
`carregar("neural")` entrega —, para valer contra qualquer arrumação interna da guarda.
"""
import pytest

from gama.extratores import guarda
from gama.extratores.guarda import guardar
from gama.pipeline import aparar
from gama.span import Span


def _s(texto, trecho, tipo="jurisprudencia", forma="processo", conf=1.0, apos=0):
    a = texto.index(trecho, apos)
    return Span(a, a + len(trecho), trecho, tipo, forma, "", conf)


def _em(inicio, fim, forma="processo", conf=1.0, tipo="jurisprudencia"):
    """Span só por offsets: as regras olham posição, forma e confiança, não o texto."""
    return Span(inicio, fim, "x" * (fim - inicio), tipo, forma, "", conf)


# ------------------------------------------------------------------ regra 1: VAGA colada

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


@pytest.mark.parametrize("inicio_vaga,fica", [(12, False), (13, True)])
def test_vaga_a_dois_caracteres_sai_a_tres_fica(inicio_vaga, fica):
    """Limite estrito: a `VAGA_COLADA` (2) caracteres ainda é colada; a 3, não."""
    juris = _em(0, 10)
    vaga = _em(inicio_vaga, 40, forma="vaga")
    assert guardar([juris, vaga], []) == ([juris, vaga] if fica else [juris])


def test_vaga_colada_antes_do_precedente_tambem_sai():
    """A distância vale para os dois lados: VAGA que termina colada no início do outro."""
    vaga = _em(0, 20, forma="vaga")
    juris = _em(21, 30)
    assert guardar([vaga, juris], []) == [juris]


def test_vaga_sobreposta_sai():
    juris = _em(10, 30)
    vaga = _em(25, 50, forma="vaga")
    assert guardar([juris, vaga], []) == [juris]


def test_duas_vagas_coladas_saem_as_duas():
    assert guardar([_em(0, 20, forma="vaga"), _em(21, 40, forma="vaga")], []) == []


def test_vaga_sozinha_fica():
    vaga = _em(5, 40, forma="vaga")
    assert guardar([vaga], []) == [vaga]


def test_regra_da_vaga_so_vale_para_vaga():
    """Dois precedentes numerados colados ("REsp 1/SP e REsp 2/RJ", espaço comido) ficam."""
    a, b = _em(0, 10), _em(11, 20)
    assert guardar([a, b], []) == [a, b]


def test_vaga_colada_a_span_fraco_sai_mesmo_que_o_fraco_saia():
    """Ordem medida na D-008: a VAGA é comparada também com os spans que a regra 2 vai
    tirar. Trocar a ordem das regras faz a VAGA sobreviver aqui."""
    fraco = _em(0, 10, conf=0.6)
    vaga = _em(11, 40, forma="vaga", conf=0.99)
    assert guardar([fraco, vaga], []) == []


# ------------------------------------------------------------------ regra 2: span fraco

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


def test_regua_que_cruza_fraco_e_forte_nao_entra():
    """Cruzar um fraco não basta: se também cruza um confiante, o do modelo manda."""
    forte = _em(0, 20)
    fraco = _em(22, 30, conf=0.6)
    regua = _em(15, 30, conf=None)
    assert guardar([forte, fraco], [regua]) == [forte]


def test_regua_encostada_no_forte_nao_o_cruza():
    """Encostar (fim == início) não é cruzar: a régua entra no lugar do fraco."""
    forte = _em(0, 10)
    fraco = _em(12, 15, conf=0.6)
    regua = _em(10, 15, conf=None)
    assert guardar([forte, fraco], [regua]) == [forte, regua]


def test_regua_que_nao_cruza_fraco_nao_entra():
    """A régua é reserva, não união (D-007): sem fraco a substituir, ela não entra."""
    fraco = _em(0, 10, conf=0.6)
    assert guardar([fraco], [_em(20, 30, conf=None)]) == []
    assert guardar([], [_em(20, 30, conf=None)]) == []


def test_regua_que_cruza_dois_fracos_entra_uma_vez():
    regua = _em(0, 30, conf=None)
    assert guardar([_em(0, 10, conf=0.6), _em(20, 30, conf=0.7)], [regua]) == [regua]


def test_confianca_no_limite_fica():
    t = "Súmula 7 do STJ."
    s = _s(t, "Súmula 7 do STJ", forma="sumula", conf=0.95)
    assert guardar([s], []) == [s]


def test_span_sem_confianca_conta_como_confiante():
    s = _em(0, 10, conf=None)
    assert guardar([s], []) == [s]


def test_vaga_fraca_sai_e_a_regua_entra_no_lugar():
    vaga = _em(0, 40, forma="vaga", conf=0.7)
    regua = _em(0, 35, forma="vaga", conf=None)
    assert guardar([vaga], [regua]) == [regua]


# ------------------------------------------------------------------ saída

def test_sem_spans_sai_vazio():
    assert guardar([], []) == []


def test_saida_ordenada_por_inicio():
    """A régua que entra antes de um forte e a entrada fora de ordem saem por início."""
    forte = _em(50, 60)
    fraco = _em(0, 5, conf=0.6)
    regua = _em(0, 12, conf=None)
    outro = _em(20, 30)
    assert guardar([forte, outro, fraco], [regua]) == [regua, outro, forte]


def test_limites_lidos_a_cada_chamada(monkeypatch):
    """`bench.controles.avaliar` troca os limites do módulo durante a medida."""
    fraco = _em(0, 10, conf=0.7)
    monkeypatch.setattr(guarda, "CONFIANCA_MINIMA", 0.5)
    assert guardar([fraco], []) == [fraco]

    juris, vaga = _em(0, 10), _em(15, 40, forma="vaga")
    assert guardar([juris, vaga], []) == [juris, vaga]
    monkeypatch.setattr(guarda, "VAGA_COLADA", 5)
    assert guardar([juris, vaga], []) == [juris]


# ------------------------------------------------------------------ o extrator de produção

def _extrator_guardado(monkeypatch, do_modelo, da_regua):
    """`carregar("neural")` com o modelo e a régua trocados por respostas fixas.

    Troca as classes em todo módulo que as tenha importado, para não depender de onde a
    guarda as constrói (nem de torch)."""
    from gama import extratores
    from gama.extratores import neural, regua

    class Modelo:
        def __init__(self, *_a, **_k):
            pass

        def extrair(self, texto):
            return list(do_modelo)

    class Regua:
        nome = "regua"

        def extrair(self, texto):
            return list(da_regua)

    for mod in (extratores, guarda, neural):
        if hasattr(mod, "ExtratorNeural"):
            monkeypatch.setattr(mod, "ExtratorNeural", Modelo)
    for mod in (extratores, guarda, regua):
        if hasattr(mod, "ExtratorRegua"):
            monkeypatch.setattr(mod, "ExtratorRegua", Regua)
    return extratores.carregar("neural", "/sem/pesos")


def test_carregar_neural_entrega_o_extrator_guardado(monkeypatch):
    ext = _extrator_guardado(monkeypatch, [], [])
    assert ext.nome == "neural"
    assert isinstance(ext, guarda.ExtratorGuardado)


def test_extrator_guardado_apara_antes_de_guardar(monkeypatch):
    """Borda com espaço e vírgula sai antes das regras; a régua entra no lugar do fragmento."""
    t = "Nos termos do art. 927, parágrafo único, do Código Civil, e REsp 1.234.567/SP."
    fragmento = _s(t, "927, ", tipo="lei", forma="artigo", conf=0.83)
    juris = _s(t, " REsp 1.234.567/SP.")
    regua = _s(t, "art. 927, parágrafo único, do Código Civil,", tipo="lei", forma="artigo", conf=None)
    ext = _extrator_guardado(monkeypatch, [fragmento, juris], [regua])
    assert [s.trecho for s in ext.extrair(t)] == [
        "art. 927, parágrafo único, do Código Civil", "REsp 1.234.567/SP"]


def test_extrator_guardado_descarta_span_que_o_aparo_esvazia(monkeypatch):
    t = "Ver ;  REsp 1/SP."
    so_borda = _s(t, " ;  ", conf=None)
    juris = _s(t, "REsp 1/SP")
    ext = _extrator_guardado(monkeypatch, [so_borda, juris], [_s(t, ";", conf=None)])
    assert ext.extrair(t) == [juris]


def test_aparo_devolve_o_mesmo_span_quando_nada_muda():
    t = "Ver REsp 1/SP."
    s = _s(t, "REsp 1/SP")
    assert aparar(s, t) is s


@pytest.mark.parametrize("bruto,aparado", [
    (" REsp 1/SP;", "REsp 1/SP"),                 # NBSP no início, ; no fim
    ("REsp 1/SP. ", "REsp 1/SP"),                 # espaço largo depois do ponto
    ("REsp 1/SP ,:", "REsp 1/SP"),
])
def test_aparo_tira_espaco_unicode_e_pontuacao_final(bruto, aparado):
    t = f"Ver {bruto} depois"
    s = aparar(_s(t, bruto), t)
    assert (s.trecho, t[s.inicio:s.fim]) == (aparado, aparado)
