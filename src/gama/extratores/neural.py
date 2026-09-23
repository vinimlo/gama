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

from ..formas import span_de
from ..span import Span
from . import bio


class ExtratorNeural:
    def __init__(self, pasta: str | pathlib.Path, max_len: int | None = None):
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
        self.max_len = min(max_len or 2048, teto)
        com = self.tok("a")["input_ids"]
        sem = self.tok("a", add_special_tokens=False)["input_ids"]
        k = next(i for i in range(len(com)) if com[i:i + len(sem)] == sem)
        self.pre, self.suf = com[:k], com[k + len(sem):]

    def _logits(self, ids: list) -> tuple[list, list]:
        """Probabilidades por token do documento inteiro, janela a janela."""
        corpo = self.max_len - len(self.pre) - len(self.suf)
        cob = bio.janelas(len(ids), corpo, corpo // 2)
        por_janela = []
        with self.torch.no_grad():
            for a, b in cob:
                t = self.torch.tensor([self.pre + ids[a:b] + self.suf], device=self.disp)
                lg = self.modelo(input_ids=t, attention_mask=self.torch.ones_like(t)).logits[0]
                p = self.torch.softmax(lg.float(), dim=-1)
                por_janela.append(p[len(self.pre):len(self.pre) + (b - a)].cpu())
        rot, conf = [], []
        for i in range(len(ids)):
            k = bio.janela_dona(i, cob)
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
        spans = []
        o = bio.ID["O"]
        for a, b, tipo in bio.decodificar(offs, rot, texto):
            # confiança do span = média sobre os tokens ROTULADOS como entidade: token
            # "O" recuperado pela extensão até a borda da palavra não conta a favor
            # (revisão independente, rodada 2, achado 5)
            cs = [c for (s, e), c, r in zip(offs, conf, rot) if s < b and a < e and r != o]
            spans.append(span_de(texto, a, b, tipo, sum(cs) / len(cs) if cs else None))
        return spans
