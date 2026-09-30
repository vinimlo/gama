# -*- coding: utf-8 -*-
"""Extrator neural (Gama): determinismo e contrato de saída. Pula sem pesos montados ou sem ./dados.

Roda com os pesos em GAMA_MODELOS (no container: ./modelos montado em /models).
"""
import csv
import os
import pathlib

import pytest

RAIZ = pathlib.Path(__file__).resolve().parent.parent
MODELOS = pathlib.Path(os.environ.get("GAMA_MODELOS", str(RAIZ / "modelos")))
DADOS = RAIZ / "dados"

pytestmark = [pytest.mark.modelos, pytest.mark.dados]


@pytest.fixture(scope="module")
def gama():
    from gama.extratores.neural import ExtratorNeural
    return ExtratorNeural(MODELOS)


def _docs(n=None):
    arqs = sorted((DADOS / "txt").glob("*.txt"))
    return arqs[:n] if n else arqs


def test_deterministico(gama):
    """Determinismo: mesma entrada, mesma saída — spans e confiança idênticos."""
    for arq in _docs(4):
        texto = arq.read_text(encoding="utf-8")
        a = [(s.inicio, s.fim, s.tipo, s.forma, s.confianca) for s in gama.extrair(texto)]
        b = [(s.inicio, s.fim, s.tipo, s.forma, s.confianca) for s in gama.extrair(texto)]
        assert a == b


def test_contrato_de_saida(gama):
    """Sem sobreposição (invalidaria a submissão) e borda na convenção do gabarito."""
    for arq in _docs():
        texto = arq.read_text(encoding="utf-8")
        spans = sorted(gama.extrair(texto), key=lambda s: s.inicio)
        for x, y in zip(spans, spans[1:]):
            assert x.fim <= y.inicio, (arq.stem, x.trecho, y.trecho)
        for s in spans:
            assert texto[s.inicio:s.fim] == s.trecho
            assert not s.trecho[0].isspace() and not s.trecho[-1].isspace()
            assert s.trecho[-1] not in ".,;:"


def test_recall_exato_no_dev(gama):
    """Guarda de regressão: o Gama acha as 192 citações do dev com borda exata."""
    gold = {}
    for r in csv.DictReader(open(DADOS / "goldenset_offsets.csv", encoding="utf-8-sig")):
        gold.setdefault(r["documento_id"], set()).add((int(r["inicio"]), int(r["fim"])))
    total = achados = 0
    for doc, spans in gold.items():
        pred = {(s.inicio, s.fim) for s in gama.extrair((DADOS / "txt" / f"{doc}.txt").read_text(encoding="utf-8"))}
        total += len(spans)
        achados += len(spans & pred)
    assert achados / total >= 0.99, (achados, total)
