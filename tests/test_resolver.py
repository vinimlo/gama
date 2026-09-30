# -*- coding: utf-8 -*-
"""Resolver sobre um índice montado à mão — sem ./dados.

O span chega com a forma já decidida (`formas.py` tem teste próprio): aqui só se mede
o caminho de cada forma até os ids. Os mesmos casos contra o acervo real estão em
test_resolver_formas.py (marca dados).
"""
import pytest

from gama.cabecalho import CadeiaDeClasse, NumeroProprio
from gama.indice import Indice, IndiceDeDispositivos, IndiceDeSumulas
from gama.normalizar import NumeroDeProcesso
from gama.resolver import (ClasseProcessual, DesempatePorCadeia, DesempatePorTribunal, Estrategia, Resolucao,
                           ResolucaoDeProcesso, Resolvedor)
from gama.span import Span


def _sp(trecho, forma, digitos=""):
    tipo = "lei" if forma == "artigo" else "jurisprudencia"
    return Span(0, len(trecho), trecho, tipo, forma, digitos)


def _indice(processos=(), sumulas=None, dispositivos=None):
    """processos: (id, número como no cabeçalho, tribunal, cadeia de classe)."""
    idx = Indice(sumulas=IndiceDeSumulas(sumulas or {}), dispositivos=IndiceDeDispositivos(dispositivos or {}))
    for doc_id, numero, tribunal, cadeia in processos:
        idx.processos.adicionar(doc_id, NumeroProprio([NumeroDeProcesso.do_bruto(numero).chave],
                                                      CadeiaDeClasse(cadeia)))
        idx.meta[doc_id] = (tribunal, 2020, "Relator")
    return idx


def resolver(span, idx):
    return Resolvedor(idx).resolver(span)


def _ids(trecho, forma, idx):
    return resolver(_sp(trecho, forma), idx).ids


# ------------------------------------------------------------------ vaga

def test_vaga_nao_consulta_o_indice():
    r = resolver(_sp("julgado do STJ de 2021, relatoria de Nancy Andrighi", "vaga"), _indice())
    assert (r.ids, r.via) == ([], "vaga")


# ------------------------------------------------------------------ súmula

SUMULAS = {("211", "STJ", False): "s211", ("10", "STF", True): "v10", ("7", "STJ", False): "s7"}


@pytest.mark.parametrize("trecho", [
    "Súmula 211 do STJ", "Súmula nº 211 do STJ", "SÚMULA 211 do STJ",
    "5úmula 211 do STJ",                     # S -> 5
    "Súrnula 211 do STJ",                    # m -> rn
    "SúmuIa 211 do STJ",                     # l -> I
    "Súmula 2l1 do STJ",                     # 1 -> l no número
    "Súm. 211 do STJ",
    "Súmula 211/STJ",
])
def test_sumula_resolve_com_ruido(trecho):
    r = resolver(_sp(trecho, "sumula"), _indice(sumulas=SUMULAS))
    assert (r.ids, r.via, r.ambiguidade) == (["s211"], "sumula", 0)


def test_sumula_com_zero_a_esquerda():
    assert _ids("Súmula 07 do STJ", "sumula", _indice(sumulas=SUMULAS)) == ["s7"]


@pytest.mark.parametrize("trecho", [
    "Súmula Vinculante 10", "Súmula Vinculante nº 10 do STF",
    "Súmula Vinculantc 10 do STF", "SÚMULA Vinculãnte 10", "Súm. Vineulante 10",
])
def test_vinculante_resolve_com_ruido(trecho):
    assert _ids(trecho, "sumula", _indice(sumulas=SUMULAS)) == ["v10"]


def test_palavra_com_v_depois_de_sumula_nao_faz_vinculante():
    assert _ids("Súmula vigente 7 do STJ", "sumula", _indice(sumulas=SUMULAS)) == ["s7"]


def test_comum_nao_resolve_para_vinculante():
    assert _ids("Súmula 10 do STF", "sumula", _indice(sumulas=SUMULAS)) == []


def test_sumula_de_outro_tribunal_e_inventada():
    assert _ids("Súmula 211 do TSE", "sumula", _indice(sumulas=SUMULAS)) == []


def test_sumula_inexistente():
    assert _ids("Súmula 979 do STF", "sumula", _indice(sumulas=SUMULAS)) == []


def test_sumula_sem_numero():
    r = resolver(_sp("Súmula do STJ", "sumula"), _indice(sumulas=SUMULAS))
    assert (r.ids, r.via) == ([], "sumula")


def test_sumula_sem_tribunal_com_empate_conta_a_ambiguidade():
    idx = _indice(sumulas={("7", "STJ", False): "s7stj", ("7", "STF", False): "s7stf"})
    r = resolver(_sp("Súmula 7", "sumula"), idx)
    assert (sorted(r.ids), r.ambiguidade) == (["s7stf", "s7stj"], 1)


# ------------------------------------------------------------------ dispositivo

DISPOSITIVOS = {
    ("373", "13105"): "cpc373", ("5", "CF"): "cf5", ("1", "LC64"): "lc64-1",
    ("14", "8078"): "cdc14", ("186", "10406"): "cc186", ("5", "8078"): "cdc5",
}


@pytest.mark.parametrize("trecho,esperado", [
    ("art. 373 do CPC", "cpc373"),
    ("art. 373, I, do Código de Processo Civil", "cpc373"),
    ("art. 373 da Lei nº 13.105/2015", "cpc373"),
    ("art. 373 da Lei 13105/2015", "cpc373"),
    ("art. 5º da Constituição Federal", "cf5"),
    ("art. 5º, XXXV, da CF", "cf5"),
    ("artigo 5º da Carta Magna", "cf5"),
    ("art. 1º da LC nº 64/1990", "lc64-1"),
    ("art. 1º da LC 64/90", "lc64-1"),
    ("art. 1º, I, g, da Lei Complementar nº 64/1990", "lc64-1"),
    ("art. 14 do Código de Defesa do Consumidor", "cdc14"),
    ("art. 14 do CDC", "cdc14"),
    ("art. 186 do Código Civil", "cc186"),
])
def test_dispositivo_resolve(trecho, esperado):
    r = resolver(_sp(trecho, "artigo"), _indice(dispositivos=DISPOSITIVOS))
    assert (r.ids, r.via) == ([esperado], "dispositivo")


@pytest.mark.parametrize("trecho,esperado", [
    ("art. 373 da Lei nº 131O5/2015", "cpc373"),               # O -> 0 no número da lei
    ("art. 373, II, da Lei nº 13. 105/2015", "cpc373"),        # espaço depois do ponto
    ("art. 1º da Lei Cornplernentar nº 64/1990", "lc64-1"),    # m -> rn na palavra-chave
    ("art. 14 do Códig0 de Defesa do Consumidor", "cdc14"),
    ("artigô 14 d0 Código de Defcsa do\nConsumidor", "cdc14"),
    ("Art.\n186 da Lêi nº 10.4O6/2002", "cc186"),
])
def test_dispositivo_resolve_com_ruido(trecho, esperado):
    assert _ids(trecho, "artigo", _indice(dispositivos=DISPOSITIVOS)) == [esperado]


def test_apelido_mais_longo_vence():
    """"Código de Processo Civil" antes de "CLT": o apelido mais longo que casar decide."""
    assert _ids("art. 373 do Código de Processo Civil e da CLT", "artigo", _indice(dispositivos=DISPOSITIVOS)) == [
        "cpc373"]


def test_lei_municipal_nao_vira_federal():
    """Mesmo número do CDC, mas lei municipal: resolver seria falso real (τ)."""
    assert _ids("art. 5º da Lei Municipal nº 8.078/1990", "artigo", _indice(dispositivos=DISPOSITIVOS)) == []


@pytest.mark.parametrize("trecho", [
    "art. 373 do CC",                          # artigo existe, em outra lei
    "art. 999 do CPC",                         # lei existe, artigo não
    "art. 10 do Regimento Interno",            # lei desconhecida
    "Lei nº 13.105/2015",                      # sem artigo
])
def test_dispositivo_sem_correspondente(trecho):
    r = resolver(_sp(trecho, "artigo"), _indice(dispositivos=DISPOSITIVOS))
    assert (r.ids, r.via) == ([], "dispositivo")


def test_tema_vai_pelo_caminho_do_dispositivo():
    """Tema de repercussão geral não está no acervo: sem artigo, sai sem candidato."""
    r = resolver(_sp("Tema 725 da repercussão geral", "tema"), _indice(dispositivos=DISPOSITIVOS))
    assert (r.ids, r.via) == ([], "dispositivo")


# ------------------------------------------------------------------ processo

def test_processo_unico():
    idx = _indice([("resp", "1.234.567", "STJ", ("REsp",))])
    r = resolver(_sp("REsp 1.234.567/SP", "processo"), idx)
    assert (r.ids, r.via, r.ambiguidade) == (["resp"], "processo", 0)


@pytest.mark.parametrize("trecho", [
    "REsp 1.234.567/SP", "REsp nº 1234567 - SP", "RECURSO ESPECIAL Nº 1.234.567",
    "AgInt no REsp 1.234.567/SP",              # "AgInt" não pode virar dígito pelo OCR
    "REsp 1.234.S67/SP",                       # S -> 5
    "REsp 1.234.\n567/SP",
])
def test_processo_resolve_pelo_nucleo_numerico(trecho):
    assert _ids(trecho, "processo", _indice([("resp", "1.234.567", "STJ", ())])) == ["resp"]


def test_processo_inexistente():
    r = resolver(_sp("Rcl 88.178/RS", "processo"), _indice([("resp", "1.234.567", "STJ", ())]))
    assert (r.ids, r.ambiguidade) == ([], 0)


def test_cnj_resolve():
    idx = _indice([("tse", "0603026-69.2018.6.09.0000", "TSE", ("AI",))])
    assert _ids("AI 0603026-69.2018.6.09.0000", "cnj", idx) == ["tse"]


def test_cnj_com_classe_grudada_pelo_ocr():
    """'AgR-A1 0603026-…': o 'A1' vira dígito e o núcleo passa de 20; ficam os 20 últimos."""
    idx = _indice([("tse", "0603026-69.2018.6.09.0000", "TSE", ("AgR", "AI"))])
    assert _ids("AgR-A1 0603026-69.2018.6.09.0000", "cnj", idx) == ["tse"]


def test_cnj_inventado_com_classe_grudada_continua_inventado():
    idx = _indice([("tse", "0603026-69.2018.6.09.0000", "TSE", ("AgR", "AI"))])
    assert _ids("AgR-A1 0603026-69.2018.6.09.0001", "cnj", idx) == []


def test_digitos_do_extrator_quando_o_trecho_nao_tem_numero():
    idx = _indice([("resp", "1.234.567", "STJ", ())])
    assert resolver(_sp("REsp", "processo", digitos="1234567"), idx).ids == ["resp"]


# ------------------------------------------------------------------ empate de processo

MESMO_NUMERO = [
    ("principal", "1.599.372", "STJ", ("REsp",)),
    ("agint", "1.599.372", "STJ", ("AgInt", "REsp")),
    ("edv", "1.599.372", "STJ", ("AgInt", "EDv", "REsp")),
]


@pytest.mark.parametrize("trecho,esperado", [
    ("REsp 1.599.372/PR", "principal"),                                     # cadeia igual
    ("AgInt no REsp 1.599.372/PR", "agint"),
    ("AgInt nos EDv no REsp 1.599.372/PR", "edv"),
    ("Agravo Interno nos Embargos de Divergência no Recurso Especial nº 1.599.372", "edv"),
])
def test_empate_desfeito_pela_cadeia_de_classe(trecho, esperado):
    r = resolver(_sp(trecho, "processo"), _indice(MESMO_NUMERO))
    assert (r.ids, r.ambiguidade) == ([esperado], 0)


def test_ordinal_separa_agravos_do_mesmo_processo():
    """Nível 1: "AgRg" e "Segundo AgRg" do mesmo REsp. Sem ordinal as duas são iguais e
    a citação cabe nas duas; só a cadeia exata separa."""
    idx = _indice([("agr2", "123.456", "STJ", ("AgR2", "REsp")), ("agr", "123.456", "STJ", ("AgR", "REsp"))])
    assert _ids("AgRg no REsp 123.456/SP", "processo", idx) == ["agr"]
    assert _ids("Segundo AgRg no REsp 123.456/SP", "processo", idx) == ["agr2"]


def test_empate_desfeito_sem_ordinal():
    """Nível 2: a citação diz "Segundo AgRg" (AgR2) e a ficha, só "AgRg". As duas fichas
    abrem com AgR (o primeiro elo não separa) e a cadeia com ordinal não cabe em nenhuma:
    só a comparação sem ordinal decide."""
    idx = _indice([("resp", "123.456", "STJ", ("AgR", "REsp")), ("aresp", "123.456", "STJ", ("AgR", "AREsp"))])
    assert _ids("Segundo AgRg no REsp 123.456/SP", "processo", idx) == ["resp"]


def test_empate_desfeito_pelo_primeiro_elo():
    """Nível 3: a citação traz só o recurso mais externo ("EDcl no REsp"). Ela cabe nas
    duas fichas, de comprimentos diferentes do dela; decide o primeiro elo."""
    idx = _indice([("ed", "123.456", "STJ", ("ED", "AgR", "REsp")),
                   ("agint", "123.456", "STJ", ("AgInt", "ED", "REsp"))])
    assert _ids("EDcl no REsp 123.456/SP", "processo", idx) == ["ed"]


def test_empate_desfeito_pela_cadeia_que_cabe():
    """Nível 4: "Ag" (abreviação que o leitor não expande) cabe em "AgR" e em "AgInt AgR";
    vence a ficha de mesmo comprimento."""
    idx = _indice([("agr", "123.456", "STJ", ("AgR", "REsp")),
                   ("agint", "123.456", "STJ", ("AgInt", "AgR", "REsp"))])
    assert _ids("Ag no REsp 123.456/SP", "processo", idx) == ["agr"]


@pytest.mark.parametrize("trecho", ["REsp 1.111.111/SP", "Recurso Especial nº 1.111.111"])
def test_empate_desfeito_pelo_tribunal_da_classe(trecho):
    """Sem cadeia no índice, a classe da citação deduz o tribunal: REsp é do STJ."""
    idx = _indice([("stf", "1.111.111", "STF", ()), ("stj", "1.111.111", "STJ", ())])
    r = resolver(_sp(trecho, "processo"), idx)
    assert (r.ids, r.ambiguidade) == (["stj"], 0)


def test_tribunal_da_classe_estreita_sem_decidir():
    idx = _indice([("stf", "1.111.111", "STF", ()), ("stj1", "1.111.111", "STJ", ()),
                   ("stj2", "1.111.111", "STJ", ())])
    r = resolver(_sp("REsp 1.111.111/SP", "processo"), idx)
    assert (r.ids, r.ambiguidade) == (["stj1", "stj2"], 1)


def test_empate_sem_criterio_devolve_todos():
    """Sem cadeia nem classe: o empate fica, e o classificar decide (D-002)."""
    idx = _indice([("a", "1.111.111", "STF", ()), ("b", "1.111.111", "STJ", ())])
    r = resolver(_sp("nº 1.111.111", "processo"), idx)
    assert (r.ids, r.via, r.ambiguidade) == (["a", "b"], "processo", 1)


# ------------------------------------------------------------------ peças do resolvedor

EMPATE_STJ = [("stf", "1.111.111", "STF", ()), ("stj1", "1.111.111", "STJ", ()), ("stj2", "1.111.111", "STJ", ())]


@pytest.mark.parametrize("trecho,classe,tribunal", [
    ("AgInt no Recurso Especial nº 1", "resp", "STJ"),
    ("Agravo em Recurso Especial 1", "aresp", "STJ"),            # a frase mais longa primeiro
    ("RR-1835-06.2010", "rr", "TST"),
    ("Rcl. 88.178", "rcl", "STF"),
    ("nº 1.111.111", None, None),
])
def test_classe_processual(trecho, classe, tribunal):
    assert (ClasseProcessual.ler(trecho), ClasseProcessual.tribunal(trecho)) == (classe, tribunal)


def test_desempate_por_tribunal():
    idx, d = _indice(EMPATE_STJ), DesempatePorTribunal()
    assert d.estreitar("REsp 1.111.111", ["stf", "stj1", "stj2"], idx) == ["stj1", "stj2"]
    assert d.estreitar("nº 1.111.111", ["stf", "stj1"], idx) == ["stf", "stj1"]          # sem classe
    assert d.estreitar("RR 1.111.111", ["stf", "stj1"], idx) == ["stf", "stj1"]          # nenhum do TST


def test_desempate_por_cadeia_nao_decidido_devolve_todos():
    idx = _indice(MESMO_NUMERO)
    todos = ["principal", "agint", "edv"]
    assert DesempatePorCadeia().estreitar("nº 1.599.372", todos, idx) == todos           # sem cadeia lida
    assert DesempatePorCadeia().estreitar("RR 1.599.372", todos, idx) == todos           # não cabe em nenhuma
    assert DesempatePorCadeia().estreitar("AgInt no REsp 1.599.372", todos, idx) == ["agint"]


def test_processo_sem_desempates_mantem_o_empate():
    idx = _indice(MESMO_NUMERO)
    r = ResolucaoDeProcesso(desempates=[]).resolver(_sp("AgInt no REsp 1.599.372", "processo"), idx)
    assert (r.ids, r.ambiguidade) == (["principal", "agint", "edv"], 2)


def test_desempates_em_ordem_ate_sobrar_um():
    """O segundo desempate só roda se o primeiro não decidiu."""
    chamados = []

    class Anota(DesempatePorTribunal):
        def estreitar(self, trecho, candidatos, idx):
            chamados.append(list(candidatos))
            return super().estreitar(trecho, candidatos, idx)

    idx = _indice(EMPATE_STJ)
    r = ResolucaoDeProcesso([Anota(), Anota()]).resolver(_sp("REsp 1.111.111", "processo"), idx)
    assert (r.ids, chamados) == (["stj1", "stj2"], [["stf", "stj1", "stj2"], ["stj1", "stj2"]])
    chamados.clear()
    ResolucaoDeProcesso([Anota()]).resolver(_sp("REsp 1.111.111", "processo"), _indice(EMPATE_STJ[:1]))
    assert chamados == []                                                                  # um só: nada a desempatar


@pytest.mark.parametrize("trecho,chave", [
    ("REsp 1.234.567/SP", "1234567"),
    ("AgR-A1 0603026-69.2018.6.09.0000", "6030266920186090000"),
    ("REsp", ""),
])
def test_chave_do_processo(trecho, chave):
    assert ResolucaoDeProcesso.chave(trecho) == chave


def test_resolvedor_escolhe_a_estrategia_pela_forma():
    class Fixa(Estrategia):
        via = "fixa"

        def resolver(self, span, idx):
            return Resolucao(["x"], self.via)

    r = Resolvedor(_indice(), {"sumula": Fixa()})
    assert r.resolver(_sp("Súmula 7", "sumula")).via == "fixa"
    assert r.resolver(_sp("REsp 1", "processo")).via == "processo"                         # sem entrada: processo
    assert r.resolver(_sp("art. 5º da CF", "artigo")).via == "processo"
