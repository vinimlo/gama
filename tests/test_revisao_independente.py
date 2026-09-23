# -*- coding: utf-8 -*-
"""Achados da revisão independente do caminho crítico, cada um virando teste antes do fix.

Revisão estática (três rodadas); cada caso abaixo reproduz um achado.
"""
import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace as P

import pytest

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "src"))

from gama.extratores import bio  # noqa: E402


def test_1_tokens_com_o_mesmo_offset_nao_duplicam_span():
    """Byte-fallback: dois tokens com offsets idênticos não podem virar dois spans —
    duas predições com IoU 1 invalidam a submissão inteira."""
    spans = bio.decodificar([(0, 1), (0, 1)], [bio.ID["B-JURIS"], bio.ID["B-JURIS"]], "⚖")
    assert spans == [(0, 1, "JURIS")]


def test_1_pipeline_nunca_emite_spans_sobrepostos():
    from gama.pipeline import sem_sobreposicao
    a = P(inicio=0, fim=10, trecho="x" * 10)
    b = P(inicio=0, fim=10, trecho="x" * 10)
    c = P(inicio=5, fim=12, trecho="x" * 7)
    d = P(inicio=20, fim=25, trecho="x" * 5)
    # b duplica a (IoU 1) e sai; c cruza a com IoU 0,42 — permitido pela regra da
    # métrica (só IoU >= 0,5 invalida), então fica (revisão independente, rodada 2, achado 4).
    assert sem_sobreposicao([a, b, c, d]) == [a, c, d]


def test_2_crlf_preserva_offsets(tmp_path):
    """A leitura não pode trocar \\r\\n por \\n: o offset é sobre o texto como veio."""
    from gama.pipeline import ler_texto
    arq = tmp_path / "d.txt"
    arq.write_bytes("x\r\nLei 1".encode("utf-8"))
    texto = ler_texto(arq)
    assert texto.index("Lei 1") == 3


def test_3_casamento_da_calibracao_igual_ao_da_metrica():
    from avaliacao.calibrar import casar
    ps = [P(inicio=0, fim=5), P(inicio=5, fim=10)]
    gs = [(0, 10, "inventada", None)]
    assert casar(ps, gs)[0][0] is ps[0]


def test_4_id_normalizado_como_na_metrica():
    from avaliacao.calibrar import mesmo_id
    assert mesmo_id("123", "00123")
    assert mesmo_id(" 123 ", "123")
    assert not mesmo_id("123", "124")


def test_5_apara_qualquer_espaco_unicode():
    texto = " Lei 1 "
    assert bio.decodificar([(0, len(texto))], [bio.ID["B-LEI"]], texto) == [(1, 6, "LEI")]


def test_6_janela_sem_avanco_e_rejeitada():
    codigo = ("import sys; sys.path.insert(0, %r); from gama.extratores import bio\n"
              "try:\n    bio.janelas(2, 1, 0)\nexcept ValueError:\n    print('ok')") % str(RAIZ / "src")
    r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, timeout=5)
    assert r.stdout.strip() == "ok"


def test_span_que_abre_no_meio_da_palavra_estende_ate_a_borda():
    """Sonda m->rn: 'Terna' vira os tokens 'Ter'+'na' e o modelo abria o span em 'na'."""
    texto = "invoca-se o Terna 725 da repercussão geral, no ponto"
    ini = texto.index("na 725")
    fim = texto.index(", no ponto")
    assert bio.decodificar([(ini, fim)], [bio.ID["B-JURIS"]], texto) == \
        [(texto.index("Terna"), fim, "JURIS")]


# ---------------------------------------------------------------- segunda rodada
# Rodada 2 (22/09): 7 achados. Cada um vira teste antes do fix.

def test_r2_1_regua_nunca_emite_borda_com_espaco():
    from gama.pipeline import processar
    for texto in ("art. 1\n", "art. 1\r\n", "art. 1 "):
        for c in processar(texto, None):
            assert not c.trecho[-1].isspace() and texto[c.inicio:c.fim] == c.trecho, repr(c.trecho)


def test_r2_2_extensao_nao_atravessa_citacao_vizinha():
    texto = "Lei 1Lei 2"
    offsets = [(0, 3), (4, 5), (5, 8), (9, 10)]
    gold = [(0, 5, "LEI"), (5, 10, "LEI")]
    assert bio.decodificar(offsets, bio.rotular(offsets, gold), texto) == gold


def test_r2_3_treino_le_crlf_como_a_inferencia(tmp_path):
    sys.path.insert(0, str(RAIZ / "treino"))
    from treino.treinar import carregar
    (tmp_path / "txt").mkdir()
    (tmp_path / "txt" / "d.txt").write_bytes(b"x\r\nLei 1")
    (tmp_path / "goldenset_offsets.csv").write_text(
        "documento_id,inicio,fim,tipo,classificacao\nd,3,8,lei,inventada\n", encoding="utf-8")
    docs, _ = carregar(tmp_path)
    assert docs["d"][0][3:8] == "Lei 1"


def test_r2_4_sobreposicao_so_invalida_com_iou_meio():
    from gama.pipeline import sem_sobreposicao
    a = P(inicio=0, fim=11, trecho="x" * 11)
    b = P(inicio=9, fim=20, trecho="x" * 11)
    assert sem_sobreposicao([a, b]) == [a, b]


def test_r2_5_confianca_so_dos_tokens_da_entidade():
    from gama.classificar import faixa
    from gama.extratores.neural import ExtratorNeural
    n = ExtratorNeural.__new__(ExtratorNeural)
    n.tok = lambda texto, **kw: {"input_ids": [10, 11, 12, 13],
                                 "offset_mapping": [(0, 1), (1, 2), (2, 5), (6, 9)]}
    n._logits = lambda ids: ([0, 0, bio.ID["B-JURIS"], bio.ID["I-JURIS"]], [1.0, 1.0, .97, .97])
    s, = n.extrair("Terna 725")
    assert faixa(s.confianca) == "baixa"


def test_r2_6_id_aceita_lista_como_a_metrica():
    from avaliacao.calibrar import mesmo_id
    assert mesmo_id("456", "123 456")
    assert mesmo_id("456", "123:456")
    assert not mesmo_id("789", "123 456")


def test_r2_7_passo_maior_que_janela_e_rejeitado():
    with pytest.raises(ValueError):
        bio.janelas(10, 3, 4)


# ---------------------------------------------------------------- terceira rodada
# Rodada 3 (23/09), sobre o que mudou depois da rodada 2.

def test_r3_1_vinculante_colado_no_numero():
    from gama.resolver import _SUM_NUM
    assert _SUM_NUM.search("Súmula Vinculante10 do STF").group(2) == "10"
    assert _SUM_NUM.search("Súmula VincuIante 10 do STF").group(2) == "10"


def test_r3_2_lei_municipal_nao_vira_federal():
    """Palavra pulada entre 'Lei' e o número só vale se for 'Complementar' (ruidoso ou não):
    'Lei Municipal nº 8.078' virar o CDC federal seria falso real — τ."""
    from gama.resolver import _chave_lei_da_citacao
    assert _chave_lei_da_citacao("art. 14 da Lei Municipal nº 8.078/1990") is None
    assert _chave_lei_da_citacao("art. 14 da Lei Estadual nº 8.078/1990") is None
    assert _chave_lei_da_citacao("art 1º da Lei Cornplernentar nº 64/1990") == "LC64"


def test_r3_3_uf_e_relator_nao_viram_classe():
    from gama.cabecalho import cadeia_de_classe
    assert cadeia_de_classe("EDcl no REsp 123 - AL") == ("ED", "REsp")
    assert cadeia_de_classe("REsp 123/SP, Rel. Min. X") == ("REsp",)
    assert cadeia_de_classe("Ernb. Decl. no AgR no REsp") == ("ED", "AgR", "REsp")


def test_r3_4_adjudicacao_preserva_tipo_lei():
    from reais.adjudicar import criterio
    t = "arts. 543-A e 543-B do CPC"
    assert criterio(t, (0, len(t), "LEI"))[2] == "LEI"


def test_r3_5_corte_do_relator_com_virgula():
    from reais.adjudicar import criterio
    t = "REsp 123/SP, Rel. Min. X"
    assert criterio(t, (0, len(t), "JURIS")) == (0, 11, "JURIS")


def test_r3_6_divergencia_adjudicada_nao_esconde_falso_positivo():
    from reais.adjudicar import metricas
    item = {"prata": [(0, 10, "JURIS"), (20, 30, "LEI")],
            "gama": [(0, 10, "processo"), (20, 30, "processo")],
            "divergente_llm": []}
    assert metricas([item], "gama", {})["JURIS"]["precisao"] == 0.5
