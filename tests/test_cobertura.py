# -*- coding: utf-8 -*-
"""Cobertura do índice sobre o acervo inteiro — o ponto cego do dev set.

O dev set cita ~95 fichas; o conjunto cego cita outras. Ficha sem número
próprio vira `inventada` em silêncio para qualquer citação a ela.
"""
import csv
import pathlib

import pytest

from avaliacao.cobertura import construir, nucleo
from gama.cabecalho import CadeiaDeClasse

RAIZ = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.dados
def test_todo_acordao_tem_numero_proprio():
    _, prop, _ = construir()
    sem = [(d, t, p.fonte) for d, (t, p) in prop.items() if not p.chaves]
    assert sem == []


@pytest.mark.dados
def test_toda_real_de_acordao_do_gabarito_resolve_para_o_id_certo():
    _, prop, por_chave = construir()
    falhas = []
    for r in csv.DictReader(open(RAIZ / "dados" / "goldenset_offsets.csv", encoding="utf-8-sig")):
        if r["classificacao"] != "real" or r["tipo"] != "jurisprudencia":
            continue
        gid = int(r["id_canonico"])
        if gid not in prop:
            continue                                   # súmula
        cands = por_chave.get(nucleo(r["trecho"]), [])
        if cands != [gid]:
            cc = CadeiaDeClasse.ler(r["trecho"].replace("\\n", "\n"))
            if [d for d in cands if prop[d][1].cadeia == cc] != [gid]:
                falhas.append((r["trecho"], gid, cands))
    assert falhas == []


def test_cadeia_de_classe_separa_embargos_de_divergencia():
    assert CadeiaDeClasse.ler("AgInt nosEMBARGOS DE DIVERGÊNCIA EM RESP") == ("AgInt", "EDv", "REsp")
    assert CadeiaDeClasse.ler("AgInt no RECURSO ESPECIAL") == ("AgInt", "REsp")
    assert CadeiaDeClasse.ler("TST-Ag-ED-AIRR") == ("Ag", "ED", "AIRR")
