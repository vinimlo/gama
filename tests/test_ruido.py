# -*- coding: utf-8 -*-
"""O injetor de ruído preserva o rótulo: offsets fecham e o número sobrevive."""
import csv
import hashlib
import json
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


# ------------------------------------------------------------------ sem ./dados

TEXTOS = [
    ("Cita-se o AgInt no REsp 1.741.784/PR, no ponto, e a Reclamação n° 66.838/BA.",
     ["AgInt no REsp 1.741.784/PR", "Reclamação n° 66.838/BA"]),
    ("Conforme o REspe nº 0600316-49.2020.6.16.0182 e a Súmula 211 do STJ, com a Lei 13.105 de 2015 e o art. 5º da CF.",
     ["REspe nº 0600316-49.2020.6.16.0182", "Súmula 211 do STJ", "art. 5º da CF"]),
    ("Julgado do STJ de 2021, da relatoria de Nancy Andrighi, sem número; o processo 21737 do ano 1999 fora dos "
     "spans e mais texto comum aqui.", ["Julgado do STJ de 2021, da relatoria de Nancy Andrighi"]),
    ("Tema 725 da repercussão geral; RR-1835-06.2010.5.15.0042; Recurso Especial 1.205.500 - SC.",
     ["Tema 725 da repercussão geral", "RR-1835-06.2010.5.15.0042", "Recurso Especial 1.205.500 - SC"]),
]


def _spans(texto, trechos):
    return [(texto.index(t), texto.index(t) + len(t)) for t in trechos]


def test_saida_congelada_por_semente():
    """A ordem das chamadas ao RNG é parte do contrato: o gerador sintético e os conjuntos
    de treino são reproduzidos pela semente. 1.600 saídas num hash, e duas por extenso."""
    h = hashlib.sha256()
    for semente in range(200):
        for texto, trechos in TEXTOS:
            for intensidade in (0.3, 0.9):
                saida = aplicar_ruido(texto, _spans(texto, trechos), random.Random(semente), intensidade)
                h.update(json.dumps(saida, ensure_ascii=False).encode())
    assert h.hexdigest() == "b5fa0f62d05e70129493cf0adb881bf56bff2eaadb3cc831fedbe745b5a384c2"
    texto, trechos = TEXTOS[0]
    assert aplicar_ruido(texto, _spans(texto, trechos), random.Random(3), 0.9) == (
        "Cita-se o AgInt no REsp\xa01741784/PR, no ponto, e a Rcclarnação n° 66838/BA.", [(10, 34), (50, 73)])
    assert aplicar_ruido(texto, _spans(texto, trechos), random.Random(5), 0.9) == (
        "Cita-se o AgInt no\nREsp 1.\n741.784/PR, no ponto, e a Reclamãção n° 6G.838/BA.", [(10, 37), (53, 76)])


@pytest.mark.parametrize("semente", range(60))
def test_invariantes_sem_dados(semente):
    """Número da citação sobrevive, CNJ não ganha letra, e os spans remapeados fecham em
    não-branco."""
    from gama.normalizar import NumeroDeProcesso
    rng = random.Random(semente)
    for texto, trechos in TEXTOS:
        spans = _spans(texto, trechos)
        novo, remap = aplicar_ruido(texto, spans, rng, 0.9)
        assert len(remap) == len(spans)
        for (a, b), (na, nb) in zip(spans, remap):
            antes, depois = texto[a:b], novo[na:nb]
            assert 0 <= na < nb <= len(novo) and not depois[0].isspace() and not depois[-1].isspace()
            if forma(antes) in ("processo", "cnj"):
                assert NumeroDeProcesso.do_trecho(depois) == NumeroDeProcesso.do_trecho(antes), (antes, depois)
            if forma(antes) == "cnj":
                numero = depois[next(i for i, c in enumerate(depois) if c.isdigit()):]
                assert not any(c.isalpha() for c in numero), depois
