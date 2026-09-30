# -*- coding: utf-8 -*-
"""ExtratorNeural sem pesos: torch e transformers falsos (torch_falso.py).

Mede o que é do extrator — carga, janela deslizante com os tokens especiais, dono de
cada token, decodificação, confiança e forma — e congela a saída sobre um texto. Os
testes com os pesos de verdade estão em test_neural.py (marcas modelos e dados).
"""
import pytest

from tests import torch_falso

TEXTO = "Ver o AgInt no REsp 1.234.567/SP e a Lei 9.099, julgado de 2020 no ponto, e o Rcl 88 aqui."
ESPERADO = [  # (inicio, fim, trecho, tipo, forma, confiança)
    (6, 11, "AgInt", "jurisprudencia", "processo", 0.9601369270062765),
    (15, 29, "REsp 1.234.567", "jurisprudencia", "processo", 0.6438696817463931),
    (30, 32, "SP", "jurisprudencia", "processo", 0.4072528143627722),
    (37, 40, "Lei", "lei", "artigo", 0.958091569823126),
    (41, 42, "9", "jurisprudencia", "processo", 0.7079319127523583),
    (43, 46, "099", "jurisprudencia", "processo", 0.7038401516950618),
    (48, 55, "julgado", "jurisprudencia", "vaga", 0.8958151557638161),
    (59, 63, "2020", "jurisprudencia", "processo", 0.7098455618860682),
    (78, 84, "Rcl 88", "jurisprudencia", "processo", 0.8341936740657718),
]


def _neural(monkeypatch, max_pos=16, cuda=False, **kw):
    torch, tok, modelo = torch_falso.instalar(monkeypatch, max_pos=max_pos, cuda=cuda)
    from gama.extratores.neural import ExtratorNeural
    return ExtratorNeural("/pesos", **kw), torch, tok, modelo


def test_carga(monkeypatch):
    n, torch, tok, modelo = _neural(monkeypatch, max_pos=512)
    assert torch.pastas == ["/pesos", "/pesos"]
    assert ("manual_seed", 0) in torch.chamadas and ("deterministico", True, True) in torch.chamadas
    assert (n.disp, modelo.no, n.max_len) == ("cpu", "cpu", 512)
    assert (n.pre, n.suf) == ([torch_falso.CLS], [torch_falso.SEP])
    assert n.tok is tok and n.modelo is modelo


@pytest.mark.parametrize("max_pos,max_len,esperado", [(512, None, 512), (4096, None, 2048), (4096, 100, 100),
                                                      (None, None, 512), (64, 100, 64)])
def test_teto_da_janela(monkeypatch, max_pos, max_len, esperado):
    n, *_ = _neural(monkeypatch, max_pos=max_pos, max_len=max_len)
    assert n.max_len == esperado


def test_cuda(monkeypatch):
    n, torch, _, modelo = _neural(monkeypatch, cuda=True)
    n.extrair("REsp 1")
    assert (n.disp, modelo.no) == ("cuda", "cuda")
    assert ("device", "cuda") in torch.chamadas


def test_extrair_congelado(monkeypatch):
    n, *_ = _neural(monkeypatch)
    spans = n.extrair(TEXTO)
    assert [(s.inicio, s.fim, s.trecho, s.tipo, s.forma) for s in spans] == [e[:5] for e in ESPERADO]
    assert [s.confianca for s in spans] == pytest.approx([e[5] for e in ESPERADO], abs=1e-12)
    assert all(TEXTO[s.inicio:s.fim] == s.trecho and s.digitos == "" for s in spans)


def test_janelas_levam_os_especiais_e_cobrem_o_documento(monkeypatch):
    n, _, tok, modelo = _neural(monkeypatch)
    n.extrair(TEXTO)
    ids = tok(TEXTO, add_special_tokens=False)["input_ids"]
    assert (len(ids), n.max_len) == (31, 16)                   # corpo de 14 tokens, passo de 7
    assert all(j[0] == torch_falso.CLS and j[-1] == torch_falso.SEP for j in modelo.janelas)
    assert [j[1:-1] for j in modelo.janelas] == [ids[a:b] for a, b in [(0, 14), (7, 21), (14, 28), (21, 31)]]


def test_dono_do_token_muda_a_confianca(monkeypatch):
    """A mesma palavra, em janela única ou no meio do documento, sai com confianças diferentes:
    o modelo falso pesa a posição na janela, e a dona é a janela em que o token está mais ao centro."""
    n, *_ = _neural(monkeypatch)
    sozinho, = [s for s in n.extrair("Rcl 88 aqui.") if s.trecho == "Rcl 88"]
    no_texto, = [s for s in n.extrair(TEXTO) if s.trecho == "Rcl 88"]
    assert sozinho.confianca != pytest.approx(no_texto.confianca)


def test_texto_sem_tokens(monkeypatch):
    n, _, _, modelo = _neural(monkeypatch)
    assert n.extrair("") == [] and n.extrair("   ") == [] and modelo.janelas == []


def test_deterministico(monkeypatch):
    n, *_ = _neural(monkeypatch)
    assert n.extrair(TEXTO) == n.extrair(TEXTO)


def test_esquema_e_formas_injetados(monkeypatch):
    from gama.extratores.bio import EsquemaBIO
    from gama.formas import DetectorDeForma

    class SoLei(EsquemaBIO):
        def decodificar(self, offsets, rotulos, texto):
            return [(a, b, "LEI") for a, b, _ in super().decodificar(offsets, rotulos, texto)]

    class Marca(DetectorDeForma):
        def span(self, texto, inicio, fim, rotulo, confianca=None):
            s = super().span(texto, inicio, fim, rotulo, confianca)
            return s.__class__(s.inicio, s.fim, s.trecho, s.tipo, "marcada", s.digitos, s.confianca)

    n, *_ = _neural(monkeypatch, esquema=SoLei(), formas=Marca())
    spans = n.extrair("o REsp 1 e")
    assert [(s.trecho, s.tipo, s.forma) for s in spans] == [("REsp 1", "lei", "marcada")]
