# -*- coding: utf-8 -*-
"""Guarda do extrator neural: as duas regras, com spans montados à mão.

`GUARDA` é a de produção (limites padrão). O extrator guardado e o catálogo recebem
modelo e régua falsos: nada aqui carrega torch nem pesos.
"""
import pytest

from gama.extratores import CatalogoDeExtratores, guarda
from gama.extratores.guarda import ExtratorGuardado, Guarda
from gama.extratores.regua import ExtratorRegua
from gama.extratores.uniao import ExtratorUniao
from gama.span import Span

GUARDA = Guarda()


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
    assert GUARDA.aplicar([juris, vaga], []) == [juris]


def test_vaga_depois_de_fim_de_frase_fica():
    """No molde da organização a vaga é frase própria: a mais próxima está a 4 caracteres."""
    t = "Aplica-se o REsp 1.307.026/BA.\nO acórdão do STF julgado em 2024 sob relatoria de Cristiano Zanin."
    juris = _s(t, "REsp 1.307.026/BA")
    vaga = _s(t, "acórdão do STF julgado em 2024 sob relatoria de Cristiano Zanin", forma="vaga")
    assert GUARDA.aplicar([juris, vaga], []) == [juris, vaga]


@pytest.mark.parametrize("inicio_vaga,fica", [(12, False), (13, True)])
def test_vaga_a_dois_caracteres_sai_a_tres_fica(inicio_vaga, fica):
    """Limite estrito: a `VAGA_COLADA` (2) caracteres ainda é colada; a 3, não."""
    juris = _em(0, 10)
    vaga = _em(inicio_vaga, 40, forma="vaga")
    assert GUARDA.aplicar([juris, vaga], []) == ([juris, vaga] if fica else [juris])


def test_vaga_colada_antes_do_precedente_tambem_sai():
    """A distância vale para os dois lados: VAGA que termina colada no início do outro."""
    vaga = _em(0, 20, forma="vaga")
    juris = _em(21, 30)
    assert GUARDA.aplicar([vaga, juris], []) == [juris]


def test_vaga_sobreposta_sai():
    juris = _em(10, 30)
    vaga = _em(25, 50, forma="vaga")
    assert GUARDA.aplicar([juris, vaga], []) == [juris]


def test_duas_vagas_coladas_saem_as_duas():
    assert GUARDA.aplicar([_em(0, 20, forma="vaga"), _em(21, 40, forma="vaga")], []) == []


def test_vaga_sozinha_fica():
    vaga = _em(5, 40, forma="vaga")
    assert GUARDA.aplicar([vaga], []) == [vaga]


def test_regra_da_vaga_so_vale_para_vaga():
    """Dois precedentes numerados colados ("REsp 1/SP e REsp 2/RJ", espaço comido) ficam."""
    a, b = _em(0, 10), _em(11, 20)
    assert GUARDA.aplicar([a, b], []) == [a, b]


def test_vaga_colada_a_span_fraco_sai_mesmo_que_o_fraco_saia():
    """Ordem medida na D-008: a VAGA é comparada também com os spans que a regra 2 vai
    tirar. Trocar a ordem das regras faz a VAGA sobreviver aqui."""
    fraco = _em(0, 10, conf=0.6)
    vaga = _em(11, 40, forma="vaga", conf=0.99)
    assert GUARDA.aplicar([fraco, vaga], []) == []


# ------------------------------------------------------------------ regra 2: span fraco

def test_span_inseguro_da_lugar_a_regua():
    t = "Nos termos do art. 927, parágrafo único, do Código Civil."
    fragmento = _s(t, "927", tipo="lei", forma="artigo", conf=0.83)
    regua = _s(t, "art. 927, parágrafo único, do Código Civil", tipo="lei", forma="artigo", conf=None)
    assert GUARDA.aplicar([fragmento], [regua]) == [regua]


def test_span_inseguro_sem_regua_sai():
    t = "DJe 19/08/2019."
    assert GUARDA.aplicar([_s(t, "DJe 19/08/2019", conf=0.7)], []) == []


def test_regua_nao_entra_onde_o_modelo_tem_certeza():
    t = "Ver REsp 1.234.567/SP e o fragmento solto 2011."
    forte = _s(t, "REsp 1.234.567/SP")
    fraco = _s(t, "2011", conf=0.6)
    regua_forte = _s(t, "REsp 1.234.567", conf=None)
    assert GUARDA.aplicar([forte, fraco], [regua_forte]) == [forte]


def test_regua_que_cruza_fraco_e_forte_nao_entra():
    """Cruzar um fraco não basta: se também cruza um confiante, o do modelo manda."""
    forte = _em(0, 20)
    fraco = _em(22, 30, conf=0.6)
    regua = _em(15, 30, conf=None)
    assert GUARDA.aplicar([forte, fraco], [regua]) == [forte]


def test_regua_encostada_no_forte_nao_o_cruza():
    """Encostar (fim == início) não é cruzar: a régua entra no lugar do fraco."""
    forte = _em(0, 10)
    fraco = _em(12, 15, conf=0.6)
    regua = _em(10, 15, conf=None)
    assert GUARDA.aplicar([forte, fraco], [regua]) == [forte, regua]


def test_regua_que_nao_cruza_fraco_nao_entra():
    """A régua é reserva, não união (D-007): sem fraco a substituir, ela não entra."""
    fraco = _em(0, 10, conf=0.6)
    assert GUARDA.aplicar([fraco], [_em(20, 30, conf=None)]) == []
    assert GUARDA.aplicar([], [_em(20, 30, conf=None)]) == []


def test_regua_que_cruza_dois_fracos_entra_uma_vez():
    regua = _em(0, 30, conf=None)
    assert GUARDA.aplicar([_em(0, 10, conf=0.6), _em(20, 30, conf=0.7)], [regua]) == [regua]


def test_confianca_no_limite_fica():
    t = "Súmula 7 do STJ."
    s = _s(t, "Súmula 7 do STJ", forma="sumula", conf=0.95)
    assert GUARDA.aplicar([s], []) == [s]


def test_span_sem_confianca_conta_como_confiante():
    s = _em(0, 10, conf=None)
    assert GUARDA.aplicar([s], []) == [s]


def test_vaga_fraca_sai_e_a_regua_entra_no_lugar():
    vaga = _em(0, 40, forma="vaga", conf=0.7)
    regua = _em(0, 35, forma="vaga", conf=None)
    assert GUARDA.aplicar([vaga], [regua]) == [regua]


# ------------------------------------------------------------------ saída

def test_sem_spans_sai_vazio():
    assert GUARDA.aplicar([], []) == []


def test_saida_ordenada_por_inicio():
    """A régua que entra antes de um forte e a entrada fora de ordem saem por início."""
    forte = _em(50, 60)
    fraco = _em(0, 5, conf=0.6)
    regua = _em(0, 12, conf=None)
    outro = _em(20, 30)
    assert GUARDA.aplicar([forte, outro, fraco], [regua]) == [regua, outro, forte]


def test_limites_da_instancia():
    fraco = _em(0, 10, conf=0.7)
    assert Guarda(confianca_minima=0.5).aplicar([fraco], []) == [fraco]
    juris, vaga = _em(0, 10), _em(15, 40, forma="vaga")
    assert GUARDA.aplicar([juris, vaga], []) == [juris, vaga]
    assert Guarda(vaga_colada=5).aplicar([juris, vaga], []) == [juris]


def test_limites_padrao():
    assert (GUARDA.vaga_colada, GUARDA.confianca_minima) == (2, 0.95)


def test_padrao_le_os_limites_do_modulo_a_cada_chamada(monkeypatch):
    """`bench.controles.avaliar` troca CONFIANCA_MINIMA durante a medida."""
    monkeypatch.setattr(guarda, "CONFIANCA_MINIMA", 0.5)
    monkeypatch.setattr(guarda, "VAGA_COLADA", 7)
    assert (Guarda.padrao().confianca_minima, Guarda.padrao().vaga_colada) == (0.5, 7)


# ------------------------------------------------------------------ o extrator de produção

class _Fixo:
    """Extrator falso: devolve sempre os mesmos spans."""
    def __init__(self, spans):
        self.spans = spans

    def extrair(self, texto):
        return list(self.spans)


def test_extrator_guardado_apara_antes_de_guardar():
    """Borda com espaço e vírgula sai antes das regras; a régua entra no lugar do fragmento."""
    t = "Nos termos do art. 927, parágrafo único, do Código Civil, e REsp 1.234.567/SP."
    fragmento = _s(t, "927, ", tipo="lei", forma="artigo", conf=0.83)
    juris = _s(t, " REsp 1.234.567/SP.")
    regua = _s(t, "art. 927, parágrafo único, do Código Civil,", tipo="lei", forma="artigo", conf=None)
    ext = ExtratorGuardado(_Fixo([fragmento, juris]), _Fixo([regua]))
    assert [s.trecho for s in ext.extrair(t)] == [
        "art. 927, parágrafo único, do Código Civil", "REsp 1.234.567/SP"]


def test_extrator_guardado_descarta_span_que_o_aparo_esvazia():
    t = "Ver ;  REsp 1/SP."
    so_borda = _s(t, " ;  ", conf=None)
    juris = _s(t, "REsp 1/SP")
    ext = ExtratorGuardado(_Fixo([so_borda, juris]), _Fixo([_s(t, ";", conf=None)]))
    assert ext.extrair(t) == [juris]


def test_extrator_guardado_usa_a_guarda_injetada():
    fraco = _em(0, 10, conf=0.7)
    t = "x" * 20
    assert ExtratorGuardado(_Fixo([fraco]), _Fixo([])).extrair(t) == []
    assert ExtratorGuardado(_Fixo([fraco]), _Fixo([]), Guarda(confianca_minima=0.5)).extrair(t) == [fraco]


def test_extrator_guardado_sem_guarda_le_os_limites_na_extracao(monkeypatch):
    """Montado antes da troca do limite, extrai com o limite trocado (o que o bench faz)."""
    fraco = _em(0, 10, conf=0.7)
    ext = ExtratorGuardado(_Fixo([fraco]), _Fixo([]))
    monkeypatch.setattr(guarda, "CONFIANCA_MINIMA", 0.5)
    assert ext.extrair("x" * 20) == [fraco]


# ------------------------------------------------------------------ união

def test_uniao_soma_da_regua_so_o_que_o_modelo_nao_cruza():
    modelo = [_em(20, 30, conf=0.6)]
    regua = [_em(0, 10, conf=None), _em(25, 40, conf=None), _em(30, 35, conf=None)]
    ext = ExtratorUniao(_Fixo(modelo), _Fixo(regua))
    assert ext.extrair("x" * 50) == [regua[0], modelo[0], regua[2]]


# ------------------------------------------------------------------ catálogo

class _Modelo:
    def __init__(self, pasta):
        self.pasta = pasta

    def extrair(self, texto):
        return []


@pytest.fixture
def catalogo(monkeypatch):
    """O catálogo com o modelo falso no lugar do ExtratorNeural (sem torch nem pesos)."""
    from gama.extratores import neural
    monkeypatch.setattr(neural, "ExtratorNeural", _Modelo)
    return CatalogoDeExtratores("/sem/pesos")


def test_catalogo_monta_cada_extrator(catalogo):
    assert isinstance(catalogo.carregar("regua"), ExtratorRegua)
    assert isinstance(catalogo.carregar("neural-cru"), _Modelo)
    guardado = catalogo.carregar("neural")
    assert isinstance(guardado, ExtratorGuardado) and guardado.nome == "neural"
    assert isinstance(guardado.neural, _Modelo) and guardado.neural.pasta == "/sem/pesos"
    assert isinstance(guardado.regua, ExtratorRegua) and guardado.guarda is None
    uniao = catalogo.carregar("uniao")
    assert isinstance(uniao, ExtratorUniao) and isinstance(uniao.neural, _Modelo)


def test_catalogo_conhece_os_nomes_do_pipeline(catalogo):
    for nome in CatalogoDeExtratores.NOMES:
        catalogo.carregar(nome)


def test_catalogo_recusa_nome_desconhecido(catalogo):
    with pytest.raises(ValueError, match="desconhecido"):
        catalogo.carregar("gliner")
