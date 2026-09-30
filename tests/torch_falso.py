# -*- coding: utf-8 -*-
"""torch e transformers de mentira (numpy), para exercitar o ExtratorNeural sem pesos.

O tokenizador corta palavra e pontuação; o "modelo" dá logits por token a partir da
palavra e da posição na janela — a posição faz a confiança depender de qual janela é a
dona do token, que é o que a janela deslizante decide.
"""
import re
import types

import numpy as np

CLS, SEP = 1, 2
ROTULOS = ["O", "B-JURIS", "I-JURIS", "B-LEI", "I-LEI", "B-VAGA", "I-VAGA"]


class Tensor(np.ndarray):
    def float(self):
        return self

    def cpu(self):
        return self


def _t(x):
    return np.asarray(x, dtype=np.float64).view(Tensor)


class _SemGrad:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def torch_falso(cuda=False):
    chamadas = []
    t = types.SimpleNamespace(chamadas=chamadas)
    t.manual_seed = lambda s: chamadas.append(("manual_seed", s))
    t.use_deterministic_algorithms = lambda v, warn_only=False: chamadas.append(("deterministico", v, warn_only))
    t.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    t.no_grad = _SemGrad

    def tensor(x, device=None):
        chamadas.append(("device", device))
        return np.asarray(x, dtype=np.int64).view(Tensor)

    t.tensor = tensor
    t.ones_like = lambda x: np.ones_like(x).view(Tensor)

    def softmax(x, dim=-1):
        e = np.exp(x - x.max(axis=dim, keepdims=True))
        return (e / e.sum(axis=dim, keepdims=True)).view(Tensor)

    t.softmax = softmax
    return t


class Tokenizador:
    def __init__(self):
        self.vocab = {}
        self.palavra = {}

    def _id(self, w):
        if w not in self.vocab:
            self.vocab[w] = len(self.vocab) + 10
            self.palavra[self.vocab[w]] = w
        return self.vocab[w]

    def __call__(self, texto, return_offsets_mapping=False, add_special_tokens=True):
        ms = list(re.finditer(r"\w+|[^\w\s]", texto))
        ids = [self._id(m.group(0)) for m in ms]
        offs = [(m.start(), m.end()) for m in ms]
        if add_special_tokens:
            ids, offs = [CLS] + ids + [SEP], [(0, 0)] + offs + [(0, 0)]
        saida = {"input_ids": ids}
        if return_offsets_mapping:
            saida["offset_mapping"] = offs
        return saida


def _logit(palavra):
    z = np.zeros(len(ROTULOS))
    if palavra in ("REsp", "Rcl", "AgInt"):
        z[1] = 5
    elif palavra is not None and palavra.isdigit():
        z[2], z[1] = 3, 1
    elif palavra in (".", "/", "-", "SP"):
        z[2], z[0] = 2, 1.5
    elif palavra in ("Lei", "art"):
        z[3] = 5
    elif palavra == "julgado":
        z[5] = 4
    elif palavra == "de":
        z[6], z[0] = 2, 1.9
    else:
        z[0] = 4
    return z


class Modelo:
    def __init__(self, tok, max_pos=512):
        self.tok = tok
        self.config = types.SimpleNamespace(max_position_embeddings=max_pos)
        self.no = None
        self.janelas = []

    def eval(self):
        return self

    def to(self, disp):
        self.no = disp
        return self

    def __call__(self, input_ids, attention_mask):
        ids = [int(i) for i in input_ids[0]]
        self.janelas.append(ids)
        assert attention_mask.shape == input_ids.shape
        lg = [_logit(self.tok.palavra.get(i)) + 0.05 * k * (np.arange(len(ROTULOS)) == 0) for k, i in enumerate(ids)]
        return types.SimpleNamespace(logits=_t([lg]))


def instalar(monkeypatch, max_pos=512, cuda=False):
    """Põe os falsos em sys.modules; devolve (torch, tokenizador, modelo)."""
    torch = torch_falso(cuda)
    tok = Tokenizador()
    modelo = Modelo(tok, max_pos)
    pastas = []
    transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda p: pastas.append(p) or tok),
        AutoModelForTokenClassification=types.SimpleNamespace(from_pretrained=lambda p: pastas.append(p) or modelo))
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    monkeypatch.setitem(__import__("sys").modules, "transformers", transformers)
    torch.pastas = pastas
    return torch, tok, modelo
