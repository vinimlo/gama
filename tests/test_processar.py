# -*- coding: utf-8 -*-
"""Pipeline sem ./dados: sobreposição, processamento de um documento e o entrypoint.

O JSON de referência (golden/regua_doc1.json) é a saída da régua sobre um documento e um
acervo montados aqui. Ele congela o comportamento do conjunto (extração, resolução,
classificação, offsets, formato): mudou a saída, o teste diz onde.
"""
import json
import pathlib
import sqlite3

import pytest

from gama.classificar import Citacao
from gama.indice import Indice
from gama.pipeline import Aplicacao, Documento, Pipeline, SaidaJSON
from gama.span import Span
from tests.test_indice import FICHAS

GOLDEN = pathlib.Path(__file__).parent / "golden" / "regua_doc1.json"
TEXTO = ("PETIÇÃO\r\nConforme o AgRg no REsp 1.205.500/SC e a Súmula 211 do STJ, bem como o art. 93 da "
         "Constituição Federal e o art. 276 do Código Eleitoral. Também o REsp 9.999.999/SP; "
         "e ainda o MS 32.714/MT.\r\nInvoca-se precedente do STF de 2020, da relatoria de CÁRMEN LÚCIA, no ponto.")


def _c(inicio, fim):
    return Citacao(inicio, fim, "x" * (fim - inicio), "jurisprudencia", "inventada", None, 0.8)


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


# ------------------------------------------------------------------ sobreposição

@pytest.mark.parametrize("a,b,iou", [
    ((0, 10), (0, 10), 1.0), ((0, 10), (5, 15), 5 / 15), ((0, 10), (10, 20), 0.0),
    ((0, 10), (2, 7), 0.5), ((0, 10), (20, 30), 0.0),
])
def test_iou(a, b, iou):
    assert _c(*a).iou(_c(*b)) == _c(*b).iou(_c(*a)) == pytest.approx(iou)


def test_sobreposicao_de_meio_ou_mais_fica_so_a_primeira():
    assert Pipeline.sem_sobreposicao([_c(0, 10), _c(2, 7)]) == [_c(0, 10)]            # IoU 0,5 exato sai
    assert Pipeline.sem_sobreposicao([_c(0, 10), _c(3, 7)]) == [_c(0, 10), _c(3, 7)]  # IoU 0,4 fica


def test_no_mesmo_inicio_fica_a_mais_longa():
    assert Pipeline.sem_sobreposicao([_c(0, 8), _c(0, 10)]) == [_c(0, 10)]


def test_intersecao_pequena_fica_e_sai_ordenado():
    assert Pipeline.sem_sobreposicao([_c(20, 30), _c(8, 21), _c(0, 10)]) == [_c(0, 10), _c(8, 21), _c(20, 30)]


def test_sem_citacoes():
    assert Pipeline.sem_sobreposicao([]) == []


# ------------------------------------------------------------------ um documento

def test_ler_texto_preserva_crlf(tmp_path):
    arq = tmp_path / "a.txt"
    arq.write_bytes("linha 1\r\nlinha 2\n".encode())
    assert Documento.ler(arq) == Documento("a", "linha 1\r\nlinha 2\n")


class _Fixo:
    nome = "fixo"

    def __init__(self, spans):
        self.spans = spans

    def extrair(self, texto):
        return list(self.spans)


def test_processar_apara_resolve_classifica_e_tira_sobreposicao(db):
    texto = "Ver o REsp 1.205.500/SC; e REsp 1.205.500."
    a = texto.index("REsp")
    spans = [Span(a - 1, a + 18, texto[a - 1:a + 18], "jurisprudencia", "processo"),     # " REsp …/SC;"
             Span(a, a + 14, texto[a:a + 14], "jurisprudencia", "processo"),             # IoU 14/17 com o aparado
             Span(0, 3, "Ver", "jurisprudencia", "vaga"),
             Span(3, 4, " ", "jurisprudencia", "processo")]                              # vazio depois de aparar
    cits = Pipeline(Indice.do_banco(db), _Fixo(spans)).processar(texto)
    assert [(c.trecho, c.classificacao, c.id_canonico) for c in cits] == [
        ("Ver", "incompleta", None),
        ("REsp 1.205.500/SC", "real", "stj-principal")]


def test_processar_sem_extrator_usa_a_regua(db):
    cits = Pipeline(Indice.do_banco(db)).processar("Conforme o MS 32.714/MT, no ponto.")
    assert [(c.trecho, c.id_canonico) for c in cits] == [("MS 32.714/MT", "stf-ms")]


# ------------------------------------------------------------------ entrypoint

def _entrada(tmp_path, textos):
    ent = tmp_path / "in"
    ent.mkdir()
    for nome, t in textos.items():
        (ent / nome).write_bytes(t.encode())
    return ent


def test_main_gera_o_json_de_referencia(tmp_path, db):
    ent = _entrada(tmp_path, {"doc1.txt": TEXTO, "ignorado.md": "REsp 1"})
    assert Aplicacao().executar(["--input", str(ent), "--output", str(tmp_path / "out"), "--db", str(db)]) == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["doc1.json"]
    saida = (tmp_path / "out" / "doc1.json").read_text(encoding="utf-8")
    assert json.loads(saida) == json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert saida == GOLDEN.read_text(encoding="utf-8").rstrip("\n")               # indent 2, sem escapar acento


def test_offsets_do_json_batem_com_o_texto():
    doc = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert [c["trecho"] for c in doc["citacoes"]] == [TEXTO[c["inicio"]:c["fim"]] for c in doc["citacoes"]]


def test_main_sem_txt_devolve_1(tmp_path, db, capsys):
    ent = _entrada(tmp_path, {})
    assert Aplicacao().executar(["--input", str(ent), "--output", str(tmp_path / "out"), "--db", str(db)]) == 1
    assert "nenhum .txt" in capsys.readouterr().err


def test_main_sem_pesos_cai_na_regua(tmp_path, db, capsys):
    ent = _entrada(tmp_path, {"doc1.txt": TEXTO})
    assert Aplicacao().executar(["--input", str(ent), "--output", str(tmp_path / "out"), "--db", str(db),
                                 "--extrator", "neural", "--modelos", str(tmp_path / "sem_pesos")]) == 0
    err = capsys.readouterr().err
    assert "AVISO: sem pesos" in err and "extrator: regua" in err
    assert json.loads((tmp_path / "out" / "doc1.json").read_text()) == json.loads(GOLDEN.read_text())


def test_main_le_o_ambiente(tmp_path, db, monkeypatch, capsys):
    ent = _entrada(tmp_path, {"doc1.txt": TEXTO})
    monkeypatch.setenv("GAMA_DB", str(db))
    monkeypatch.setenv("GAMA_EXTRATOR", "neural")
    monkeypatch.setenv("GAMA_MODELOS", str(tmp_path / "sem_pesos"))
    assert Aplicacao().executar(["--input", str(ent), "--output", str(tmp_path / "out")]) == 0
    assert f"sem pesos em {tmp_path / 'sem_pesos'}" in capsys.readouterr().err


def test_pipeline_com_classificador_injetado(db):
    from gama.classificar import Classificador, TabelaDeConfianca
    p = Pipeline(Indice.do_banco(db), classificador=Classificador(TabelaDeConfianca(padrao=0.1, priores={})))
    assert [c.confianca for c in p.processar("Conforme o MS 32.714/MT, no ponto.")] == [0.1]


def test_saida_json(tmp_path):
    saida = SaidaJSON(tmp_path / "nova" / "pasta")
    cit = Citacao(0, 4, "Ação", "jurisprudencia", "inventada", None, 0.8)
    arq = saida.escrever(Documento("d1", "Ação"), [cit, cit])
    assert arq == tmp_path / "nova" / "pasta" / "d1.json"
    doc = json.loads(arq.read_text(encoding="utf-8"))
    ids = [c["id"] for c in doc["citacoes"]]
    assert (doc["schema_version"], doc["documento_id"], ids) == ("1.2", "d1", ["c1", "c2"])
    assert "Ação" in arq.read_text(encoding="utf-8")


def test_extrator_escolhido(tmp_path, capsys):
    (tmp_path / "config.json").write_text("{}")
    assert type(Aplicacao.extrator("regua", str(tmp_path))).__name__ == "ExtratorRegua"
    assert "AVISO" not in capsys.readouterr().err
    assert type(Aplicacao.extrator("neural", str(tmp_path / "nada"))).__name__ == "ExtratorRegua"
    assert "AVISO" in capsys.readouterr().err


def test_escolhas_do_extrator_vem_do_catalogo():
    from gama.extratores import CatalogoDeExtratores
    acoes = {a.dest: a for a in Aplicacao.argumentos()._actions}
    assert tuple(acoes["extrator"].choices) == CatalogoDeExtratores.NOMES
