# -*- coding: utf-8 -*-
"""Normalização: texto achatado, esqueleto de OCR e chave de número de processo.

O localizador do número no trecho (NumeroDeProcesso.do_trecho) tem os casos de ruído em test_nucleo.py.
"""
import pytest

from gama.normalizar import OCR, OCR_PARA_DIGITO, Normalizador, NumeroDeProcesso, TabelaOCR

achatar, esqueleto, sem_acento = Normalizador.achatar, Normalizador.esqueleto, Normalizador.sem_acento


def test_sem_acento():
    assert sem_acento("Súmula Ação ÇÃO côncavo") == "Sumula Acao CAO concavo"


def test_achatar():
    assert achatar("  Código  de\nProcesso\tCivil ") == "codigo de processo civil"


def test_esqueleto_iguala_as_trocas_de_ocr():
    assert esqueleto("Códig0 de Pr0cesso Civil") == esqueleto("Código de Processo Civil")
    assert esqueleto("Código de Processo Civil") == "eodlgo de proeesso elvll"
    assert esqueleto("Consolldação") == esqueleto("Consolidação")
    assert esqueleto("Ernbargos") == esqueleto("Embargos")


def test_tabela_de_ocr_so_troca_letra_por_digito():
    """A garantia da organização: dígito nunca vira outro dígito."""
    assert all(not k.isdigit() and v.isdigit() for k, v in OCR_PARA_DIGITO.items())


@pytest.mark.parametrize("bruto,digitos", [
    ("1.528.4S5/ RJ", "1528455"),
    ("21737l8", "2173718"),
    ("7000449-40.2023", "7000449402023"),
    ("", ""),
    ("xy", ""),
])
def test_so_digitos(bruto, digitos):
    assert OCR.so_digitos(bruto) == digitos


@pytest.mark.parametrize("bruto,digitos", [
    ("AgInt no RESP 21737l8 - SP", "4910521737185"),
    ("REsp 1.741.784", "51741784"),
])
def test_so_digitos_no_trecho_inteiro_cola_as_letras_do_prefixo(bruto, digitos):
    """Por isso o resolver usa NumeroDeProcesso.do_trecho, nunca so_digitos no span inteiro."""
    assert OCR.so_digitos(bruto) == digitos


@pytest.mark.parametrize("bruto,chave", [
    ("1.741.784", "1741784"),
    ("0001741784", "1741784"),
    ("000", "000"),                                     # só zeros: fica como está
    ("", ""),
])
def test_chave_processo_tira_zeros_a_esquerda(bruto, chave):
    assert NumeroDeProcesso.do_bruto(bruto).chave == chave


@pytest.mark.parametrize("bruto,numero", [
    ("2l1", "211"),
    ("10", "10"),
    ("1O", "10"),
    ("lO", None),                                       # sem dígito real não é número
    ("2x1", "21"),                                      # letra fora da tabela some
])
def test_numero_ocr(bruto, numero):
    assert OCR.numero(bruto) == numero


def test_numero_guarda_os_digitos_e_a_chave_tira_os_zeros():
    n = NumeroDeProcesso.do_bruto("0001.741-784")
    assert (n.digitos, n.chave) == ("0001741784", "1741784")
    assert NumeroDeProcesso.do_trecho("REsp 0001.741.784/SP") == NumeroDeProcesso("0001741784")
    assert NumeroDeProcesso.do_trecho("sem número").chave == ""


@pytest.mark.parametrize("pedaco,numerico", [
    ("45g", True), ("21737l8", True), ("6G", True), ("170076O", True),
    ("Rc1", False), ("n0", False), ("Especia1", False), ("OO", False), ("12x", False),
    ("lO1", False), ("AgI1", False),                  # só letras da tabela, mas mais letras que dígitos
])
def test_pedaco_numerico(pedaco, numerico):
    assert OCR.pedaco_numerico(pedaco) == numerico


def test_letras_do_digito_e_a_inversa_so_com_letras():
    inversa = OCR.letras_do_digito()
    assert inversa["1"] == ["l", "I", "i"]                       # "|" não é letra
    assert all(OCR_PARA_DIGITO[letra] == d for d, letras in inversa.items() for letra in letras)
    assert sum(map(len, inversa.values())) == sum(k.isalpha() for k in OCR_PARA_DIGITO)


def test_tabela_injetada():
    ocr = TabelaOCR({"X": "9"})
    assert (ocr.so_digitos("1X2O"), ocr.numero("1X"), ocr.pedaco_numerico("1O")) == ("192", "19", False)
    assert NumeroDeProcesso.do_trecho("REsp 1X2/SP", ocr).chave == "192"
