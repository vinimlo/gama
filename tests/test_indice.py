# -*- coding: utf-8 -*-
"""Índice do acervo sobre um SQLite minúsculo, montado no teste — sem ./dados.

Cada ficha imita o cabeçalho real do seu tribunal (cabecalho.py) e da sua natureza
(súmula, dispositivo). O acervo inteiro é coberto por test_cobertura.py (marca dados).
"""
import sqlite3

import pytest

from gama.indice import Indice, _chave_lei, construir

FICHAS = [
    # id, natureza, tribunal, ano, relator, texto
    ("stj-agrg", "acordao", "STJ", 2011, "Min. A",
     "AgRg no RECURSO ESPECIAL Nº 1.205.500 - SC (2010⁄0146585-7)\nRELATOR : MINISTRO A"),
    ("stj-principal", "acordao", "STJ", 2010, "Min. A",
     "RECURSO ESPECIAL Nº 1.205.500 - SC (2010⁄0146585-7)\nRELATOR : MINISTRO A"),
    ("stf-ms", "acordao", "STF", 2015, "Min. B",
     "Supremo Tribunal Federal\nPLENÁRIO EMB.DECL. NO AG.REG. EM MANDADO DE SEGURANÇA 32.714 "
     "MATO GROSSO RELATORA : MIN. B"),
    ("tst-rr", "acordao", "TST", None, "Min. C",
     "A C Ó R D Ã O\nVistos, relatados e discutidos estes autos de Recurso de Revista "
     "n° TST-RR-1835-06.2010.5.15.0042, em que é Recorrente X."),
    ("sum-211", "sumula", "STJ", 1998, None,
     "Súmula n. 211 do STJ\nInadmissível recurso especial quanto à questão que..."),
    ("sum-v10", "sumula", "STF", 2008, None,
     "Súmula Vinculante n. 10 do STF\nViola a cláusula de reserva de plenário..."),
    ("art-276-ce", "dispositivo", None, None, None,
     "Artigo 276 da Lei nº 4.737, de 15 de julho de 1965 Art. 276. As decisões..."),
    ("art-93-cf", "dispositivo", None, None, None,
     "Artigo 93 da Constituição Federal de 1988 Art. 93. Lei complementar..."),
    ("art-1-lc64", "dispositivo", None, None, None,
     "Artigo 1º da Lei Complementar nº 64, de 18 de maio de 1990 Art. 1º São inelegíveis..."),
]


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    caminho = tmp_path_factory.mktemp("acervo") / "acervo.db"
    con = sqlite3.connect(caminho)
    con.execute("CREATE TABLE documentos (id TEXT, natureza TEXT, tribunal TEXT, ano INTEGER, "
                "relator TEXT, texto TEXT)")
    con.executemany("INSERT INTO documentos VALUES (?, ?, ?, ?, ?, ?)", FICHAS)
    con.commit()
    con.close()
    return caminho


@pytest.fixture(scope="module")
def idx(db):
    return construir(db)


def test_meta_de_toda_ficha(idx):
    assert set(idx.meta) == {f[0] for f in FICHAS}
    assert idx.meta["stf-ms"] == ("STF", 2015, "Min. B")


def test_acordao_indexado_pelo_numero_proprio(idx):
    assert idx.candidatos_processo("32.714") == ["stf-ms"]
    assert idx.candidatos_processo("RR-1835-06.2010.5.15.0042") == ["tst-rr"]


def test_fichas_do_mesmo_processo_dividem_a_chave(idx):
    """Agravo e principal do mesmo número: os dois são candidatos; a cadeia os separa."""
    assert idx.candidatos_processo("1205500") == ["stj-agrg", "stj-principal"]
    assert idx.cadeia["stj-agrg"] == ("AgR", "REsp")
    assert idx.cadeia["stj-principal"] == ("REsp",)
    assert idx.cadeia["stf-ms"] == ("ED", "AgR", "MS")


def test_chave_ignora_zeros_e_pontuacao(idx):
    assert idx.candidatos_processo("01.205.500") == ["stj-agrg", "stj-principal"]


def test_numero_ausente_sem_candidato(idx):
    assert idx.candidatos_processo("9999999") == []


def test_sumulas_pelo_cabecalho(idx):
    assert idx.sumulas == {("211", "STJ", False): "sum-211", ("10", "STF", True): "sum-v10"}


def test_dispositivos_pelo_cabecalho(idx):
    assert idx.dispositivos == {("276", "4737"): "art-276-ce", ("93", "CF"): "art-93-cf",
                                ("1", "LC64"): "art-1-lc64"}


def test_acervo_so_leitura(db):
    """O índice abre o banco em modo leitura: não cria nada ao lado do acervo."""
    antes = sorted(p.name for p in db.parent.iterdir())
    construir(db)
    assert sorted(p.name for p in db.parent.iterdir()) == antes


@pytest.mark.parametrize("descricao,chave", [
    ("Lei nº 13.105, de 16 de março de 2015", "13105"),
    ("Lei nº 4.737, de 15 de julho de 1965", "4737"),
    ("Constituição Federal de 1988", "CF"),
    ("Constituição da República", "CF"),
    ("Lei Complementar nº 64, de 18 de maio de 1990", "LC64"),
    ("Decreto-Lei nº 01.001, de 1969", "1001"),
])
def test_chave_da_lei_no_cabecalho(descricao, chave):
    assert _chave_lei(descricao) == chave


# ------------------------------------------------------------------ consultas

def test_sumula_pelo_numero_tribunal_e_vinculo():
    idx = Indice(sumulas={("211", "STJ", False): "s211", ("10", "STF", True): "v10"})
    assert idx.resolve_sumula("211", "STJ", False) == ["s211"]
    assert idx.resolve_sumula("0211", "STJ", False) == ["s211"]
    assert idx.resolve_sumula("10", "STF", True) == ["v10"]
    assert idx.resolve_sumula("10", "STF", False) == []          # a comum 10 não é a vinculante


def test_sumula_de_outro_tribunal_nao_resolve_pelo_numero():
    """"Súmula 211 do TSE" não é a 211 do STJ: resolver pelo número faria de inventada real (τ)."""
    idx = Indice(sumulas={("211", "STJ", False): "s211"})
    assert idx.resolve_sumula("211", "TSE", False) == []


def test_sumula_sem_tribunal_pelo_numero_unico():
    idx = Indice(sumulas={("211", "STJ", False): "s211", ("10", "STF", True): "v10"})
    assert idx.resolve_sumula("211", None, False) == ["s211"]
    assert idx.resolve_sumula("10", None, True) == ["v10"]
    assert idx.resolve_sumula("10", None, False) == []


def test_sumula_sem_tribunal_com_numero_repetido_devolve_todas():
    """Empate: vão os dois candidatos, e o classificar entrega `real` com o primeiro (D-002)."""
    idx = Indice(sumulas={("7", "STJ", False): "s7stj", ("7", "STF", False): "s7stf"})
    assert sorted(idx.resolve_sumula("7", None, False)) == ["s7stf", "s7stj"]


def test_dispositivo_pelo_artigo_e_lei():
    idx = Indice(dispositivos={("5", "CF"): "a5cf"})
    assert idx.resolve_dispositivo("5", "CF") == ["a5cf"]
    assert idx.resolve_dispositivo("005", "CF") == ["a5cf"]
    assert idx.resolve_dispositivo("5", "13105") == []
    assert idx.resolve_dispositivo("6", "CF") == []
