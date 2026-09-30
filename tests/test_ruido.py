# -*- coding: utf-8 -*-
"""O injetor de ruído preserva o rótulo: offsets fecham e o número sobrevive."""
import csv
import pathlib
import random

import pytest

from avaliacao.cobertura import nucleo
from avaliacao.convencoes import forma
from gama.ruido import DIGITO_PARA_LETRA, aplicar_ruido
from gama.normalizar import OCR_PARA_DIGITO

RAIZ = pathlib.Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"


def _gabarito_nivel1():
    """Documentos de nível 1 (limpos) com seus spans — matéria-prima do teste."""
    por_doc = {}
    for r in csv.DictReader(open(DADOS / "goldenset_offsets.csv", encoding="utf-8-sig")):
        if r["nivel"] == "1":
            por_doc.setdefault(r["documento_id"], []).append((int(r["inicio"]), int(r["fim"])))
    return {d: ((DADOS / "txt" / f"{d}.txt").read_text(encoding="utf-8"), sp)
            for d, sp in por_doc.items()}


def test_tabela_de_ocr_e_inversa_exata():
    """Toda letra que o injetor põe no lugar de um dígito volta para ELE."""
    for dig, letras in DIGITO_PARA_LETRA.items():
        for letra in letras:
            assert OCR_PARA_DIGITO[letra] == dig


def test_intensidade_zero_nao_altera():
    texto = "Cita-se o REsp 1.741.784/PR, no ponto."
    assert aplicar_ruido(texto, [(10, 27)], random.Random(1), 0.0) == (texto, [(10, 27)])


@pytest.mark.dados
@pytest.mark.parametrize("semente", range(40))
def test_numero_de_cada_citacao_sobrevive_ao_ruido(semente):
    """INVARIANTE da organização: dígito nunca vira outro dígito."""
    rng = random.Random(semente)
    for doc, (texto, spans) in _gabarito_nivel1().items():
        novo, remap = aplicar_ruido(texto, spans, rng, intensidade=0.9)
        for (a, b), (na, nb) in zip(spans, remap):
            antes, depois = texto[a:b], novo[na:nb]
            # Só números resolvidos pelo índice de processos. Artigo de lei tem
            # vários números e resolve pelo par (artigo, lei), por outro caminho.
            if forma(antes) in ("processo", "cnj") and nucleo(antes):
                assert nucleo(depois) == nucleo(antes), (doc, antes, depois)


@pytest.mark.dados
@pytest.mark.parametrize("semente", range(20))
def test_offsets_remapeados_apontam_para_o_span_ruidoso(semente):
    """Cada span remapeado começa e termina em caractere não-branco e o número de
    spans se mantém — o rótulo continua alinhado depois de m->rn e quebras."""
    rng = random.Random(1000 + semente)
    for doc, (texto, spans) in _gabarito_nivel1().items():
        novo, remap = aplicar_ruido(texto, spans, rng, intensidade=0.9)
        assert len(remap) == len(spans)
        for na, nb in remap:
            assert 0 <= na < nb <= len(novo)
            assert not novo[na].isspace() and not novo[nb - 1].isspace()


def test_ruido_e_deterministico_por_semente():
    texto = "Invoca-se a Súmula 211 do STJ e o AgInt no REsp 1.599.910/PR."
    spans = [(12, 29), (34, 61)]
    assert aplicar_ruido(texto, spans, random.Random(7), 0.8) == \
        aplicar_ruido(texto, spans, random.Random(7), 0.8)
