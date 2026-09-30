# -*- coding: utf-8 -*-
"""Número próprio de cada ficha, pelo cabeçalho do seu tribunal, e a cadeia de classe.

Os cabeçalhos imitam os formatos reais descritos em cabecalho.py. O acervo inteiro é
coberto por test_cobertura.py (marca dados); aqui cada leitor é exercitado sozinho.
"""
import pytest

from gama.cabecalho import (CadeiaDeClasse, LeitorDeCabecalho, LeitorDeNumeroProprio, LeitorGenerico, LeitorSTJ,
                            LeitorTSE, NumeroProprio)

LEITOR = LeitorDeNumeroProprio()

STM = "SUPERIOR TRIBUNAL MILITAR\nAPELAÇÃO Nº 7000101-61.2019.7.00.0000\nRELATOR: MIN. X"
TSE_ANTIGO = ("TRIBUNAL SUPERIOR ELEITORAL\nACÓRDÃO\nAGRAVO DE INSTRUMENTO Nº 10.353 "
              "( 38964-78.2008.6.00.0000) - CLASSE 6 - SÃO PAULO")
TSE_ANTIGO_SEM_CNJ = "TRIBUNAL SUPERIOR ELEITORAL\nACÓRDÃO\nRECURSO ESPECIAL ELEITORAL Nº 25.105 - CLASSE 32"
TSE_CNJ = "TRIBUNAL SUPERIOR ELEITORAL\nACÓRDÃO\nRECURSO ESPECIAL ELEITORAL Nº 378-82.2016.6.05.0151 - CLASSE 32"
TSE_CNJ_FROUXO = ("TRIBUNAL SUPERIOR ELEITORAL\nACÓRDÃO\nAGRAVO REGIMENTAL NO RECURSO ESPECIAL ELEITORAL "
                  "N o 685-65. 2016.6.11.0055")
STF = "PRIMEIRA TURMA AG.REG. NO RECURSO EXTRAORDINÁRIO 1.276.977 SÃO PAULO RELATOR : MIN. Y"
STJ = "AgRg no RECURSO ESPECIAL Nº 1.205.500 - SC (2010⁄0146585-7)\nRELATOR : MINISTRO A"
TST = ("Vistos, relatados e discutidos estes autos de Embargos em Recurso de Revista "
       "nº TST-ED-ED-E- ED-ED-ARR-1575-04.2016.5.20.0001, em que é Embargante X.")
TST_FRACO = "Vistos estes autos de Agravo nº Ag-AIRR-100-10.2019.5.01.0001, em que é Agravante X."
# Formato raro: o cabeçalho de sempre não está lá, mas há "<CLASSE> Nº <número>" adiante.
RARO = "Documento exportado\n" + "x" * 400 + "\nHABEAS CORPUS Nº 123.456 texto"


@pytest.mark.parametrize("tribunal,texto,chaves,cadeia,fonte,numero", [
    ("STF", STF, ["1276977"], ("AgR", "RE"), "stf", "1.276.977"),
    ("STJ", STJ, ["1205500"], ("AgR", "REsp"), "stj", "1.205.500"),
    ("STM", STM, ["70001016120197000000"], ("APL",), "stm", "7000101-61.2019.7.00.0000"),
    ("TST", TST, ["15750420165200001"], ("ED", "ED", "E", "ED", "ED", "ARR"), "tst:autos",
     "1575-04.2016.5.20.0001"),
    ("TST", TST_FRACO, ["1001020195010001"], ("Ag", "AIRR"), "tst:autos_fraco", "100-10.2019.5.01.0001"),
])
def test_cabecalho_de_cada_tribunal(tribunal, texto, chaves, cadeia, fonte, numero):
    p = LEITOR.ler(texto, tribunal)
    assert (p.chaves, p.cadeia, p.fonte, p.numero) == (chaves, cadeia, fonte, numero)


def test_tse_antigo_indexa_o_numero_antigo_e_o_cnj():
    p = LEITOR.ler(TSE_ANTIGO, "TSE")
    assert (p.chaves, p.cadeia, p.fonte) == (["10353", "389647820086000000"], ("AI",), "tse:antigo")
    assert (p.numero, p.classe) == ("38964-78.2008.6.00.0000", "AGRAVO DE INSTRUMENTO")


def test_tse_antigo_sem_cnj():
    p = LEITOR.ler(TSE_ANTIGO_SEM_CNJ, "TSE")
    assert (p.chaves, p.cadeia, p.fonte, p.numero) == (["25105"], ("REspe",), "tse:antigo", "25.105")


def test_tse_cnj_direto_vem_antes_do_numero_antigo():
    """Com a ordem inversa, "Nº 378-82.2016..." casava o padrão antigo em "378" (chave truncada)."""
    p = LEITOR.ler(TSE_CNJ, "TSE")
    assert (p.chaves, p.cadeia, p.fonte) == (["3788220166050151"], ("REspe",), "tse:cnj")


def test_tse_cnj_com_espaco_e_marca_de_ocr():
    p = LEITOR.ler(TSE_CNJ_FROUXO, "TSE")
    assert (p.chaves, p.cadeia, p.fonte) == (["6856520166110055"], ("AgR", "REspe"), "tse:cnj")


def test_tst_sem_sigla_conhecida_le_a_classe_por_extenso():
    texto = ("discutidos estes autos de Recurso de Revista nº XX-1835-06.2010.5.15.0042, "
             "em que é Recorrente X.")
    assert LEITOR.ler(texto, "TST").cadeia == ("RR",)


def test_tribunal_em_minusculas():
    p = LEITOR.ler(STJ, "stj")
    assert (p.chaves, p.fonte) == (["1205500"], "stj")


@pytest.mark.parametrize("tribunal", ["STJ", "STF", "STM", "TSE", "XYZ"])
def test_formato_raro_cai_no_leitor_generico(tribunal):
    p = LEITOR.ler(RARO, tribunal)
    assert (p.chaves, p.cadeia, p.fonte, p.numero) == (["123456"], ("HC",), "generico", "123.456")


def test_tst_nao_usa_o_leitor_generico():
    """No TST o primeiro "CLASSE Nº" costuma ser um processo CITADO."""
    p = LEITOR.ler(RARO, "TST")
    assert (p.chaves, p.fonte) == ([], "tst:sem_padrao")


@pytest.mark.parametrize("tribunal,fonte", [
    ("STF", "stf:sem_padrao"), ("STJ", "stj:sem_padrao"), ("STM", "stm:sem_padrao"),
    ("TSE", "tse:sem_padrao"), ("TST", "tst:sem_padrao"),
    ("XYZ", "tribunal_desconhecido"), (None, "tribunal_desconhecido"), ("", "tribunal_desconhecido"),
])
def test_sem_padrao_nem_generico_fica_a_fonte_do_tribunal(tribunal, fonte):
    p = LEITOR.ler("texto sem cabeçalho nenhum", tribunal)
    assert (p.chaves, p.cadeia, p.fonte, p.numero, p.classe) == ([], (), fonte, "", "")


def test_cabecalho_do_stj_abre_o_documento():
    """STJ: a classe e o número estão no começo; mais adiante, só o genérico acha."""
    p = LEITOR.ler("x" * 400 + STJ, "STJ")
    assert p.fonte == "generico"


# ------------------------------------------------------------------ cadeia de classe

@pytest.mark.parametrize("texto,cadeia", [
    ("EMB.DECL. NO AG.REG. EM MANDADO DE SEGURANÇA", ("ED", "AgR", "MS")),
    ("TST-ED-ED-E-ED-ED-ARR", ("ED", "ED", "E", "ED", "ED", "ARR")),
    ("AGRAVO EM RECURSO ESPECIAL", ("AREsp",)),                    # a mais longa primeiro
    ("AGRAVO DE INSTRUMENTO EM RECURSO DE REVISTA", ("AI", "RR")),
    ("SEGUNDO AG.REG. NO RECURSO EXTRAORDINÁRIO", ("AgR2", "RE")),
    ("Terceiros EDcl no REsp", ("ED3", "REsp")),
    ("AgInt nosEMBARGOS DE DECLARAÇÃO", ("AgInt", "ED")),           # colado na exportação
    ("palavra desconhecida", ()),
    ("", ()),
])
def test_cadeia_de_classe(texto, cadeia):
    assert CadeiaDeClasse.ler(texto) == cadeia


def test_ordinal_vale_so_para_o_elo_seguinte():
    assert CadeiaDeClasse.ler("Segundo AgRg no AgRg no REsp") == ("AgR2", "AgR", "REsp")


def test_cadeia_e_uma_tupla():
    c = CadeiaDeClasse.ler("AgRg no REsp")
    assert isinstance(c, tuple) and c == ("AgR", "REsp") and {c: 1}[("AgR", "REsp")] == 1


@pytest.mark.parametrize("cadeia,sem_ordinal,primeiro", [
    (("AgR2", "REsp"), ("AgR", "REsp"), ("AgR",)),
    (("ED3", "AgR2", "RE"), ("ED", "AgR", "RE"), ("ED",)),
    ((), (), ()),
])
def test_sem_ordinal_e_primeiro_elo(cadeia, sem_ordinal, primeiro):
    c = CadeiaDeClasse(cadeia)
    assert (c.sem_ordinal(), c.primeiro_elo()) == (sem_ordinal, primeiro)
    assert isinstance(c.sem_ordinal(), CadeiaDeClasse) and isinstance(c.primeiro_elo(), CadeiaDeClasse)


@pytest.mark.parametrize("lida,ficha,cabe", [
    (("Ag", "REsp"), ("AgR", "REsp"), True),             # cada elo lido é prefixo
    (("Ag", "REsp"), ("AgInt", "AgR", "REsp"), True),     # subsequência
    (("E",), ("EDv",), True),
    (("REsp", "AgR"), ("AgR", "REsp"), False),            # ordem importa
    (("AgR", "REsp"), ("REsp",), False),
    ((), ("REsp",), True),
])
def test_cabe_em(lida, ficha, cabe):
    assert CadeiaDeClasse(lida).cabe_em(ficha) == cabe


# ------------------------------------------------------------------ leitores

def test_cada_leitor_le_so_o_seu_formato():
    assert LeitorTSE().ler(TSE_CNJ).fonte == "tse:cnj"
    assert LeitorSTJ().ler(TSE_CNJ).fonte == "stj:sem_padrao"
    assert LeitorGenerico().ler("nada").fonte == "sem_padrao"


class _Fixo(LeitorDeCabecalho):
    def __init__(self, fonte, chaves=()):
        self.fonte, self.chaves, self.lidos = fonte, list(chaves), 0

    def ler(self, texto):
        self.lidos += 1
        return NumeroProprio(list(self.chaves), fonte=self.fonte)


def test_leitores_injetados():
    tribunal, generico = _Fixo("x", ["1"]), _Fixo("g", ["2"])
    leitor = LeitorDeNumeroProprio({"ABC": tribunal}, generico)
    assert leitor.ler("t", "abc").fonte == "x"
    assert generico.lidos == 0                                    # achou: não tenta o genérico
    assert leitor.ler("t", "STJ").fonte == "g"                     # tribunal sem leitor


def test_generico_so_quando_o_tribunal_nao_acha():
    generico = _Fixo("g")
    leitor = LeitorDeNumeroProprio({"ABC": _Fixo("x")}, generico, sem_generico=frozenset({"ABC"}))
    assert (leitor.ler("t", "ABC").fonte, generico.lidos) == ("x", 0)
    leitor = LeitorDeNumeroProprio({"ABC": _Fixo("x")}, generico, sem_generico=frozenset())
    assert (leitor.ler("t", "ABC").fonte, generico.lidos) == ("x", 1)   # genérico vazio: fica o do tribunal
