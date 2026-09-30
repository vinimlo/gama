# -*- coding: utf-8 -*-
"""Testes que guardam os invariantes -- todos nasceram de um bug real."""
from __future__ import annotations

import pathlib

import pytest

from gama.extrair import extrair
from gama.indice import Indice
from gama.normalizar import OCR, NumeroDeProcesso
from gama.pipeline import Pipeline

RAIZ = pathlib.Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"


@pytest.fixture(scope="session")
def idx():
    return Indice.do_banco(DADOS / "desafio1_bracis.db")


# --------------------------------------------------------------- normalizacao

@pytest.mark.parametrize("bruto,esperado", [
    ("21737l8", "2173718"),      # l -> 1
    ("1.528.4S5", "1528455"),    # S -> 5
    ("170076O", "1700760"),      # O -> 0
    ("6G.838", "66838"),         # G -> 6
    ("1.45g.779", "1459779"),    # g -> 9
])
def test_ocr_letra_vira_digito(bruto, esperado):
    assert OCR.so_digitos(bruto) == esperado


def test_chave_ignora_zeros_a_esquerda():
    assert NumeroDeProcesso.do_bruto("0600216").chave == NumeroDeProcesso.do_bruto("600216").chave


# --------------------------------------------------------------- extracao

def test_prefixo_nao_contamina_a_chave():
    """INVARIANTE: so o nucleo numerico vira chave.

    Aplicar a tabela de OCR ao trecho inteiro converte as letras do prefixo em
    digitos e produz chave fantasma. Bug silencioso que apareceu tres vezes.
    """
    spans = extrair("Cita-se o AgInt no AREsp 1576933/SP, que trata do tema.")
    assert len(spans) == 1
    assert spans[0].digitos == "1576933"
    assert "AgInt" in spans[0].trecho


def test_ponto_de_abreviatura_nao_encerra_sentenca():
    """O corte ingenuo em '. ' decepava o prefixo e custava IoU."""
    texto = "Observa-se o AgRg no Rec. Esp. n. 1.522.200 (SC), no ponto."
    spans = extrair(texto)
    assert len(spans) == 1
    assert spans[0].trecho.startswith("AgRg")


def test_numero_dos_autos_no_cabecalho_e_distrator():
    texto = ("TRIBUNAL\nProcesso nº 8133385-26.2020.5.05.4913\n\n\n"
             "No mérito, invoca-se o RR-1835-06.2010.5.15.0042 como paradigma.")
    trechos = [s.trecho for s in extrair(texto)]
    assert not any("8133385" in t for t in trechos)
    assert any("1835-06" in t for t in trechos)


def test_referencia_vaga_sem_numero():
    texto = ("Ao final.\nInvoca-se precedente do STF de 2026, "
             "da relatoria de CRISTIANO ZANIN, no ponto.")
    spans = extrair(texto)
    assert [s.forma for s in spans] == ["vaga"]


# --------------------------------------------------------------- ponta a ponta

@pytest.mark.dados
def test_dispositivos_todos_indexados(idx):
    assert len(idx.dispositivos) == 13
    assert len(idx.sumulas) == 5


@pytest.mark.dados
def test_tst_resolve_apesar_do_preambulo(idx):
    """Os acordaos do TST so citam o proprio numero por volta do char 1100."""
    cit = Pipeline(idx).processar("Cita-se o RR-1835-06.2010.5.15.0042 no ponto.")
    assert len(cit) == 1
    assert cit[0].classificacao == "real"
    assert cit[0].id_canonico == "392669042"


@pytest.mark.dados
def test_citacao_inexistente_e_inventada(idx):
    cit = Pipeline(idx).processar("Menciona-se a Rcl 88.178/RS, sem correspondente.")
    assert len(cit) == 1
    assert cit[0].classificacao == "inventada"
    assert cit[0].id_canonico is None


@pytest.mark.dados
def test_offsets_apontam_para_o_texto_original(idx):
    """inicio/fim sao a chave de juncao com o gabarito -- precisam fechar."""
    texto = (DADOS / "txt" / "gen_n2_003.txt").read_text(encoding="utf-8")
    for c in Pipeline(idx).processar(texto):
        assert texto[c.inicio:c.fim] == c.trecho


@pytest.mark.dados
def test_confianca_nunca_saturada(idx):
    """INVARIANTE: nao inflar confianca para 1,0 -- o bonus e Brier."""
    texto = (DADOS / "txt" / "gen_n1_001.txt").read_text(encoding="utf-8")
    for c in Pipeline(idx).processar(texto):
        assert 0.0 < c.confianca < 1.0
