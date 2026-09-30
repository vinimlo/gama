# -*- coding: utf-8 -*-
"""Extrator neural: classificação de tokens BIO fine-tunada (JURIS, LEI, VAGA).

Pesos por volume (`--modelos /models`), nunca dentro da imagem. A janela e a
decodificação são as de `bio.py` — o MESMO módulo que rotulou o treino, então o
que o treino mede é o que a inferência entrega.

Determinismo: eval(), sem amostragem, algoritmos determinísticos do torch; na CPU
os JSONs saem byte-idênticos entre execuções.
"""
from __future__ import annotations

import pathlib

from ..formas import DetectorDeForma
from ..span import Span
from .base import Extrator
from .bio import EsquemaBIO, Janelas


class ExtratorNeural(Extrator):
    """O modelo (pesos em `pasta`) em janelas de até `max_len` tokens, com os tokens
    especiais do tokenizador em cada janela; a dona de cada token é a janela em que ele
    está mais ao centro."""
    nome = "neural-cru"
    MAX_LEN = 2048
    esquema = EsquemaBIO()
    formas = DetectorDeForma()

    def __init__(self, pasta: str | pathlib.Path, max_len: int | None = None,
                 esquema: EsquemaBIO | None = None, formas: DetectorDeForma | None = None):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(pasta))
        self.modelo = AutoModelForTokenClassification.from_pretrained(str(pasta)).eval()
        self.disp = "cuda" if torch.cuda.is_available() else "cpu"
        self.modelo.to(self.disp)
        teto = getattr(self.modelo.config, "max_position_embeddings", 512) or 512
        self.max_len = min(max_len or self.MAX_LEN, teto)
        self.pre, self.suf = self._especiais(self.tok)
        if esquema is not None:
            self.esquema = esquema
        if formas is not None:
            self.formas = formas

    @staticmethod
    def _especiais(tok) -> tuple[list, list]:
        """Os ids que o tokenizador põe antes e depois do texto ([CLS] ... [SEP])."""
        com = tok("a")["input_ids"]
        sem = tok("a", add_special_tokens=False)["input_ids"]
        k = next(i for i in range(len(com)) if com[i:i + len(sem)] == sem)
        return com[:k], com[k + len(sem):]

    def _probabilidades(self, janela: list):
        """Probabilidades por token de uma janela (sem as posições dos especiais)."""
        t = self.torch.tensor([self.pre + janela + self.suf], device=self.disp)
        lg = self.modelo(input_ids=t, attention_mask=self.torch.ones_like(t)).logits[0]
        p = self.torch.softmax(lg.float(), dim=-1)
        return p[len(self.pre):len(self.pre) + len(janela)].cpu()

    def _logits(self, ids: list) -> tuple[list, list]:
        """Rótulo e probabilidade dele, por token do documento inteiro, janela a janela."""
        corpo = self.max_len - len(self.pre) - len(self.suf)
        cob = Janelas(corpo, corpo // 2).cobrir(len(ids))
        with self.torch.no_grad():
            por_janela = [self._probabilidades(ids[a:b]) for a, b in cob]
        rot, conf = [], []
        for i in range(len(ids)):
            k = Janelas.dona(i, cob)
            p = por_janela[k][i - cob[k][0]]
            j = int(p.argmax())
            rot.append(j)
            conf.append(float(p[j]))
        return rot, conf

    def extrair(self, texto: str) -> list[Span]:
        enc = self.tok(texto, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"]
        offs = [tuple(o) for o in enc["offset_mapping"]]
        if not ids:
            return []
        rot, conf = self._logits(ids)
        o = self.esquema.ID["O"]
        spans = []
        for a, b, tipo in self.esquema.decodificar(offs, rot, texto):
            # confiança do span = média sobre os tokens ROTULADOS como entidade: token
            # "O" recuperado pela extensão até a borda da palavra não conta a favor
            # (revisão independente, rodada 2, achado 5)
            cs = [c for (s, e), c, r in zip(offs, conf, rot) if s < b and a < e and r != o]
            spans.append(self.formas.span(texto, a, b, tipo, sum(cs) / len(cs) if cs else None))
        return spans
