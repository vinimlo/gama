# -*- coding: utf-8 -*-
"""Normalização: texto achatado, esqueleto de OCR e chave de número de processo.

O localizador do número no trecho (nucleo_numerico) tem os casos de ruído em test_nucleo.py.
"""
import pytest

from gama.normalizar import OCR_PARA_DIGITO, achatar, chave_processo, esqueleto, sem_acento, so_digitos
from gama.resolver import _numero_ocr


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
    assert so_digitos(bruto) == digitos


@pytest.mark.parametrize("bruto,digitos", [
    ("AgInt no RESP 21737l8 - SP", "4910521737185"),
    ("REsp 1.741.784", "51741784"),
])
def test_so_digitos_no_trecho_inteiro_cola_as_letras_do_prefixo(bruto, digitos):
    """Por isso o resolver usa nucleo_numerico, nunca so_digitos no span inteiro."""
    assert so_digitos(bruto) == digitos


@pytest.mark.parametrize("bruto,chave", [
    ("1.741.784", "1741784"),
    ("0001741784", "1741784"),
    ("000", "000"),                                     # só zeros: fica como está
    ("", ""),
])
def test_chave_processo_tira_zeros_a_esquerda(bruto, chave):
    assert chave_processo(bruto) == chave


@pytest.mark.parametrize("bruto,numero", [
    ("2l1", "211"),
    ("10", "10"),
    ("1O", "10"),
    ("lO", None),                                       # sem dígito real não é número
    ("2x1", "21"),                                      # letra fora da tabela some
])
def test_numero_ocr(bruto, numero):
    assert _numero_ocr(bruto) == numero
