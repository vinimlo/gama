# -*- coding: utf-8 -*-
"""Sondas do ruído DOCUMENTADO pela organização, aplicado a palavras-chave.

Cada sonda troca UMA letra da palavra-chave por uma confusão da lista oficial
(m↔rn, 1↔l). No dev set nenhuma delas caiu numa palavra-chave; no conjunto cego,
com ~100 citações de nível 2 (peso 2×), vão cair. A régua falha em 5 de 6 — o
extrator neural tem que passar em todas.
"""
import os
import pathlib
import sys

import pytest

RAIZ = pathlib.Path(__file__).resolve().parent.parent
MODELOS = pathlib.Path(os.environ.get("GAMA_MODELOS", str(RAIZ / "modelos")))
sys.path.insert(0, str(RAIZ / "src"))

from gama.extratores import carregar  # noqa: E402
from gama.indice import construir  # noqa: E402
from gama.pipeline import processar  # noqa: E402

MOLDURA = ("PODER JUDICIÁRIO\nTRIBUNAL REGIONAL\n\nProcesso nº 8133385-26.2020.5.05.4913\n"
           "Relator: Desembargador PAULO HENRIQUE\n\n\nACÓRDÃO\n\n"
           "Trata-se de recurso. Contrarrazões apresentadas.\n\n"
           "No mérito, invoca-se o {cit}, no ponto.\n\nConclusão.")

SONDAS = [
    ("m->rn Sumula", "Súrnula 211 do STJ", "real"),
    ("m->rn Tema", "Terna 725 da repercussão geral", "inventada"),
    ("m->rn Reclamacao", "Reclarnação nº 66.516/RO", "inventada"),
    ("l->1 Especial", "AgInt no Recurso Especia1 nº 1.620.021/PR", "real"),
    ("l->1 relatoria", "precedente do STF de 2024, da re1atoria de Cármen Lúcia", "incompleta"),
    ("l->1 Rel.", "Rcl de 2021, Re1. Min. Rosa Weber", "incompleta"),
]
FALHA_CONHECIDA_DA_REGUA = {"m->rn Sumula", "m->rn Tema", "l->1 Especial",
                            "l->1 relatoria", "l->1 Rel."}
# Gama v1.2 (vinimlo/gama@ad06ffd) passa nas seis — o v1 falhava em "Terna 725", fechado
# no v1.2 com ruído dirigido à palavra-chave. Conjunto vazio: qualquer falha é
# regressão.
FALHA_CONHECIDA_DO_NEURAL: set = set()


def _extratores():
    yield "regua"
    modelos = MODELOS
    if (modelos / "config.json").exists() or any(modelos.glob("*/config.json")):
        yield "neural"


@pytest.fixture(scope="module")
def idx():
    return construir(RAIZ / "dados" / "desafio1_bracis.db")


def _iou(a, b):
    i = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return i / ((a[1] - a[0]) + (b[1] - b[0]) - i) if i else 0.0


@pytest.mark.parametrize("nome_extrator", list(_extratores()))
@pytest.mark.parametrize("nome,cit,esperado", SONDAS)
def test_sonda(nome_extrator, nome, cit, esperado, idx, request):
    if nome_extrator == "regua" and nome in FALHA_CONHECIDA_DA_REGUA:
        request.node.add_marker(pytest.mark.xfail(reason="régua não resiste a ruído em palavra-chave", strict=True))
    if nome_extrator == "neural" and nome in FALHA_CONHECIDA_DO_NEURAL:
        request.node.add_marker(pytest.mark.xfail(reason="Gama v1: Tema fora da faixa do treino; corrigido no v1.2", strict=True))
    extrator = carregar(nome_extrator, str(MODELOS))
    texto = MOLDURA.format(cit=cit)
    gold = (texto.index(cit), texto.index(cit) + len(cit))
    casados = [c for c in processar(texto, idx, extrator) if _iou(gold, (c.inicio, c.fim)) >= 0.5]
    assert casados, f"{nome}: nenhum span com IoU >= 0,5"
    assert casados[0].classificacao == esperado
