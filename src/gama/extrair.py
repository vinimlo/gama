# -*- coding: utf-8 -*-
"""Fachada (até a leva 8): a régua mora em `extratores/regua.py` (ExtratorRegua)."""
from __future__ import annotations

from .extratores.regua import ExtensorDeCitacao, ExtratorRegua, PadroesDaRegua
from .span import Span  # noqa: F401  (reexportado)


def extrair(texto: str) -> list:
    return ExtratorRegua().extrair(texto)


def _fim_cabecalho(texto: str) -> int:
    return ExtratorRegua.fim_do_cabecalho(texto)


def _e_sigla(bruto: str) -> bool:
    return ExtensorDeCitacao.e_sigla(bruto)


def _fronteira_sentenca(janela: str) -> int:
    return ExtensorDeCitacao.fronteira_sentenca(janela)


def _estende_esquerda(texto: str, inicio: int, limite: int = 80) -> int:
    return ExtensorDeCitacao(limite).esquerda(texto, inicio)


def _estende_direita(texto: str, fim: int) -> int:
    return ExtensorDeCitacao.direita(texto, fim)


def _tem_classe(prefixo: str) -> bool:
    return ExtensorDeCitacao.tem_classe(prefixo)


def _vagas(texto: str, fim_cabecalho: int) -> list:
    from .extratores.regua import DetectorDeVagas
    return DetectorDeVagas().encontrar(texto, fim_cabecalho)


VOCAB_PREFIXO = PadroesDaRegua.VOCAB_PREFIXO
