# -*- coding: utf-8 -*-
"""As regras de borda valem no gabarito inteiro.

O mesmo verificador valida o goldenset gerado sinteticamente: se o gabarito da
organização não viola nenhuma regra, um documento gerado que viole está com a
borda errada e ensinaria o modelo errado.
"""
import csv
import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from avaliacao.convencoes import violacoes_de_borda  # noqa: E402

DADOS = RAIZ / "dados"


def test_gabarito_nao_viola_nenhuma_regra_de_borda():
    textos, violacoes = {}, []
    for r in csv.DictReader(open(DADOS / "goldenset_offsets.csv", encoding="utf-8-sig")):
        d = r["documento_id"]
        if d not in textos:
            textos[d] = (DADOS / "txt" / f"{d}.txt").read_text(encoding="utf-8")
        span = textos[d][int(r["inicio"]):int(r["fim"])]
        for v in violacoes_de_borda(span):
            violacoes.append((d, v, span))
    assert violacoes == []


def test_regras_pegam_bordas_erradas():
    assert "artigo_ou_preposicao_inicial" in violacoes_de_borda("o REsp 1.741.784/PR")
    assert "pontuacao_final_dentro" in violacoes_de_borda("REsp 1.741.784/PR,")
    assert "processo_sem_uf" in violacoes_de_borda("REsp 1.741.784")
    assert violacoes_de_borda("REsp 1.741.784/PR") == []
    assert violacoes_de_borda("art. 373, I, do CPC") == []
