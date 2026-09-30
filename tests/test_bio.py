# -*- coding: utf-8 -*-
"""BIO: rótulos por token <-> spans por caractere, e as janelas do documento longo."""
import pytest

from gama.extratores import bio

B, I, FORA = bio.ID["B-JURIS"], bio.ID["I-JURIS"], bio.ID["O"]
BL, IL, BV = bio.ID["B-LEI"], bio.ID["I-LEI"], bio.ID["B-VAGA"]


def test_esquema():
    assert bio.TIPOS == ("JURIS", "LEI", "VAGA")
    assert bio.ROTULOS == ["O", "B-JURIS", "I-JURIS", "B-LEI", "I-LEI", "B-VAGA", "I-VAGA"]
    assert all(bio.ROTULOS[i] == r for r, i in bio.ID.items())
    assert bio.IGNORAR == -100


# ------------------------------------------------------------------ rotular

def test_rotular_b_no_primeiro_token_e_i_nos_seguintes():
    offsets = [(0, 0), (0, 4), (5, 6), (7, 10), (11, 14), (0, 0)]
    assert bio.rotular(offsets, [(0, 6, "JURIS"), (11, 14, "LEI")]) == [bio.IGNORAR, B, I, FORA, BL, bio.IGNORAR]


def test_rotular_token_que_cruza_a_borda_conta_como_dentro():
    assert bio.rotular([(0, 3), (3, 8)], [(4, 7, "JURIS")]) == [FORA, B]


def test_rotular_spans_fora_de_ordem():
    assert bio.rotular([(0, 2), (3, 5)], [(3, 5, "LEI"), (0, 2, "VAGA")]) == [BV, BL]


def test_rotular_token_em_dois_spans_fica_com_o_que_comeca_antes():
    assert bio.rotular([(3, 5)], [(3, 8, "LEI"), (0, 5, "JURIS")]) == [B]


def test_rotular_spans_colados_cada_um_com_o_seu_b():
    assert bio.rotular([(0, 3), (3, 6), (6, 9)], [(0, 3, "LEI"), (3, 9, "LEI")]) == [BL, BL, IL]


# ------------------------------------------------------------------ decodificar

TEXTO = "REsp 123 e Lei 9.099."


def test_decodificar_b_i_e_troca_de_tipo():
    offsets = [(0, 4), (5, 8), (9, 10), (11, 14), (15, 20), (20, 21)]
    assert bio.decodificar(offsets, [B, I, FORA, IL, IL, IL], TEXTO) == [(0, 8, "JURIS"), (11, 20, "LEI")]


def test_decodificar_i_de_outro_tipo_abre_span_novo():
    assert bio.decodificar([(0, 4), (5, 8)], [B, IL], TEXTO) == [(0, 4, "JURIS"), (5, 8, "LEI")]


def test_decodificar_b_seguido_de_b_sao_dois_spans():
    assert bio.decodificar([(0, 4), (5, 8)], [B, B], TEXTO) == [(0, 4, "JURIS"), (5, 8, "JURIS")]


def test_decodificar_ignora_especiais_e_ignorar():
    assert bio.decodificar([(0, 0), (0, 4), (5, 8)], [B, bio.IGNORAR, B], TEXTO) == [(5, 8, "JURIS")]


def test_decodificar_offset_repetido_continua_o_span():
    assert bio.decodificar([(0, 1), (0, 1), (1, 4)], [B, B, I], "⚖abc") == [(0, 4, "JURIS")]


def test_decodificar_estende_ate_a_borda_da_palavra():
    assert bio.decodificar([(2, 5)], [B], "Terna 725") == [(0, 5, "JURIS")]
    assert bio.decodificar([(0, 3)], [B], "Terna 725") == [(0, 5, "JURIS")]


def test_decodificar_nao_atravessa_o_vizinho():
    texto = "Lei 1Lei 2"
    assert bio.decodificar([(0, 3), (4, 5), (5, 8), (9, 10)], [BL, IL, BL, IL], texto) == [
        (0, 5, "LEI"), (5, 10, "LEI")]


def test_decodificar_apara_espaco_e_pontuacao_final():
    texto = " REsp 1,  "
    assert bio.decodificar([(0, 10)], [B], texto) == [(1, 7, "JURIS")]


def test_decodificar_nunca_devolve_spans_que_se_cruzam():
    assert bio.decodificar([(0, 4), (2, 6)], [B, B], "abcdefgh") == [(0, 4, "JURIS")]


def test_decodificar_span_so_de_pontuacao_some():
    assert bio.decodificar([(0, 2)], [B], ".; x") == []


def test_ida_e_volta():
    texto = "Ver REsp 1.234/SP, e art. 5º da CF; julgado de 2020, Rel. Min. X."
    gold = [(4, 17, "JURIS"), (21, 34, "LEI"), (36, 64, "VAGA")]
    offsets, i = [], 0
    for palavra in texto.split(" "):
        offsets.append((i, i + len(palavra)))
        i += len(palavra) + 1
    assert bio.decodificar(offsets, bio.rotular(offsets, gold), texto) == gold


# ------------------------------------------------------------------ janelas

@pytest.mark.parametrize("n,max_len,passo,esperado", [
    (5, 10, 5, [(0, 5)]),
    (10, 10, 5, [(0, 10)]),
    (0, 4, 2, [(0, 0)]),
    (10, 4, 2, [(0, 4), (2, 6), (4, 8), (6, 10)]),
    (11, 4, 4, [(0, 4), (4, 8), (8, 11)]),
    (9, 4, 3, [(0, 4), (3, 7), (6, 9)]),
])
def test_janelas(n, max_len, passo, esperado):
    assert bio.janelas(n, max_len, passo) == esperado


@pytest.mark.parametrize("max_len,passo", [(0, 1), (4, 0), (4, 5), (-1, -1)])
def test_janela_invalida(max_len, passo):
    with pytest.raises(ValueError):
        bio.janelas(10, max_len, passo)


@pytest.mark.parametrize("i,dona", [(0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 2), (6, 2), (9, 3)])
def test_janela_dona_e_a_mais_centrada(i, dona):
    assert bio.janela_dona(i, [(0, 4), (2, 6), (4, 8), (6, 10)]) == dona


def test_janela_dona_fora_de_toda_janela_e_a_primeira():
    assert bio.janela_dona(50, [(0, 4), (2, 6)]) == 0
