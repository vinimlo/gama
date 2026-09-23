# -*- coding: utf-8 -*-
"""Regressões do resolver achadas pelo gerador sintético.

Cada caso veio de uma citação renderizada a partir de uma ficha real cujo rótulo
é conhecido por construção, e que o resolver classificava errado.
"""
import pathlib
import sys

import pytest

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from gama.classificar import classificar  # noqa: E402
from gama.formas import forma, span_de  # noqa: E402
from gama.indice import construir  # noqa: E402
from gama.resolver import resolver  # noqa: E402

DB = RAIZ / "dados" / "desafio1_bracis.db"


@pytest.fixture(scope="module")
def idx():
    return construir(DB)


def _classe(idx, trecho, rotulo="JURIS"):
    texto = f"Cita-se o {trecho}, no ponto."
    sp = span_de(texto, 10, 10 + len(trecho), rotulo)
    c = classificar(sp, resolver(sp, idx))
    return c.classificacao, c.id_canonico


def test_ordinal_por_extenso_nao_vira_tema():
    assert forma("Terceiro AgR na Rcl nº 62425/SP") == "processo"
    assert forma("Tema 2.680 da repercussão geral") == "tema"
    assert forma("Temã 2.680 da repercussão geral") == "tema"


def test_ordinal_por_extenso_resolve(idx):
    assert _classe(idx, "Terceiro AgR na Rcl nº 62425/SP") == ("real", "2428195275")


def test_sumula_de_outro_tribunal_e_inventada(idx):
    """A 211 existe no STJ; 'Súmula 211 do TSE' é outra súmula, que não existe."""
    assert _classe(idx, "Súmula 211 do STJ") == ("real", "1289710776")
    assert _classe(idx, "SÚMULA 211 do TSE") == ("inventada", None)


def test_lc_abreviada_resolve(idx):
    assert _classe(idx, "art. 1º da LC nº 64/1990", "LEI") == ("real", "11304039")
    assert _classe(idx, "art. 1º da LC 64/90", "LEI") == ("real", "11304039")


@pytest.mark.parametrize("trecho,esperado", [
    ("art. 7º, XXVIII, 'a', da Constltuição Federal", "10641213"),
    ("Art. 373 do Códig0 de Pr0cesso Civil", "28893055"),
    ("artlgo 290,\nII, do Código Penal  Militar", "10590194"),
    ("artigo  477 da Consolldação\ndas Leis do Trabalho", "10710324"),
    ("artigo 7º, XXIX, da Constituição Fedcral", "10641213"),
])
def test_lei_com_ruido_de_ocr_resolve(idx, trecho, esperado):
    """Ruído no NOME do diploma não pode derrubar a resolução (esqueleto OCR)."""
    assert _classe(idx, trecho, "LEI") == ("real", esperado)


@pytest.mark.parametrize("trecho,esperado", [
    ("SúmuIa\n211/STJ", "1289710776"),
    ("Súmulã\n211/STJ", "1289710776"),
    ("Súrn. 443/STJ", "1289711022"),
    ("SÚMULA VincuIante 10 do  STF", "1289712966"),
    ("5úmula 211 do STJ", "1289710776"),
    ("Súmula 2l1 do STJ", "1289710776"),
])
def test_sumula_com_ruido_de_ocr_resolve(idx, trecho, esperado):
    assert _classe(idx, trecho) == ("real", esperado)


def test_sumula_inventada_com_ruido_continua_inventada(idx):
    assert _classe(idx, "SúmuIa 979 do STF") == ("inventada", None)


@pytest.mark.parametrize("trecho,esperado", [
    ("art. 373\nda Lei nº 131O5/2015", "28893055"),
    ("art. 14, II, da Lei nº 807B/1990", "10606184"),
    ("art. 373, II,  da Lei nº 13. 105/2D15", "28893055"),
    ("ãrt. 7º, XXI,\n'ã', da CF", "10641213"),
    ("art. 1º, § 4º-A, dã Lei Complêmcntar nº 64/199D", "11304039"),
    ("art. 1º, I, 'g', da Lei Complementar nº 64/1990", "11304039"),
])
def test_numero_da_lei_com_ruido_resolve(idx, trecho, esperado):
    assert _classe(idx, trecho, "LEI") == ("real", esperado)


def test_vinculante_com_i_trocado(idx):
    assert _classe(idx, "SÚMULA\nVlnculante 10") == ("real", "1289712966")


# Ruído na PALAVRA-CHAVE (goldenset v3, 22/09): o resolver tem que aguentar o mesmo
# ruído que o extrator aprendeu a aguentar.
@pytest.mark.parametrize("trecho,esperado", [
    ("art 1º da Lei Cornplernentar nº 64/1990", "11304039"),
    ("art. 1º, V, da Lei Cornplcrnentar nº 64/19q0", "11304039"),
    ("art. 1º da Lei Cornp1cmentar nº 64/1990", "11304039"),
    ("Art. 276 da\nLei nº 473T/1965", "10577194"),
    ("artigô 14 d0 Código de Defcsa do\nConsumidor", "10606184"),
])
def test_lei_com_ruido_na_palavra_chave(idx, trecho, esperado):
    assert _classe(idx, trecho, "LEI") == ("real", esperado)


@pytest.mark.parametrize("trecho", [
    "Súrnula Vinculantc\n10 do  5TF", "SÚMULA Vinculãnte 10 do STF", "Súrn. Vineulantê 10",
    "5ÚMULA\nVinculãnte 10", "Súm.  Vineulante 10 do 5TF", "Súmula Vinculantc 10",
])
def test_vinculante_com_ruido(idx, trecho):
    assert _classe(idx, trecho) == ("real", "1289712966")


def test_cadeia_de_classe_tolera_ocr():
    from gama.cabecalho import cadeia_de_classe
    assert cadeia_de_classe("Ernb. Decl. no AgR no REsp") == cadeia_de_classe("Emb. Decl. no AgR no REsp")
    assert cadeia_de_classe("Aglnt nos EREsp") == cadeia_de_classe("AgInt nos EREsp")
    assert cadeia_de_classe("Rcc. Esp.") == ("REsp",)
    assert cadeia_de_classe("5egundo AG.REG. no ARE") == cadeia_de_classe("Segundo AG.REG. no ARE")


def test_abreviacao_com_espaco_depois_do_ponto():
    from gama.cabecalho import cadeia_de_classe
    assert cadeia_de_classe("Ag. Reg. no REsp") == ("AgR", "REsp")
    assert cadeia_de_classe("Emb. Decl. no AgR no REsp") == ("ED", "AgR", "REsp")
    assert cadeia_de_classe("AgInt nos Emb. Div. no REsp") == ("AgInt", "EDv", "REsp")
    assert cadeia_de_classe("AgInt nos EREsp no REsp") == ("AgInt", "EDv", "REsp")


def test_lei_com_circunflexo(idx):
    assert _classe(idx, "Art.\n186 da Lêi nº 10.A06/2Q02", "LEI") == ("real", "10718759")
