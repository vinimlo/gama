# -*- coding: utf-8 -*-
# Trechos copiados do Laya 0.3.22, Copyright Convai Innovations, sob a Apache License 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). As mudanças estão descritas abaixo.
"""Cópia mínima do forward do Laya 0.3.22 (laya/common.py e laya/agent.py, Apache-2.0, Convai
Innovations) para pontuar o laya_ft em CPU no container sem instalar a biblioteca: montagem da
sequência (`build_sequence` para uma pergunta `choice`), `DecisionModel` com a atenção
`_DynamicMultiheadAttention`, `build_model` sem pesos pré-treinados, o conserto do
tokenizer_config e a temperatura do runtime (`clamp_temperature`, `temp_bucket`). O código
copiado é o da versão 0.3.22 (sdist do PyPI), sem mudar a conta; o que a pontuação local não usa
(noul com rótulos próprios, score, estatísticas de truncamento, meta device, TileLang, hooks) ficou
de fora. Única adaptação: a tabela de embedding carrega só as linhas dos tokens usados, em fp16 lido
como fp32 (a mesma conta, com menos memória: a VM do Docker é compartilhada). A paridade com a
biblioteca de verdade (HF Jobs) é conferida pelo CLI (`paridade`).
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

QTYPES = {"choice": 0, "score": 1, "noul": 2}
TEMP_MIN, TEMP_MAX = 0.5, 5.0


def render_criterion(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q: dict) -> list:
    if q["t"] != "choice":
        raise ValueError("laya_min só monta perguntas choice")
    return [str(k) if v is None or v == "" else "%s: %s" % (k, render_criterion(v)) for k, v in q["crit"].items()]


def build_sequence(tok, state: str, q: dict, max_len: int, head_max_len: int) -> tuple[list, list, bool]:
    """[CLS] <tipo> instruções [SEP] [MASK] opção0 [MASK] opção1 ... [SEP] estado [SEP] (laya.common)."""
    mask_tok = tok.mask_token
    opts = render_options(q)
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    opt_ids = []
    for o in opts:
        opt_tokens = tok(" " + o.replace(mask_tok, " "), add_special_tokens=False, truncation=True,
                         max_length=48)["input_ids"]
        opt_ids.append([tok.mask_token_id] + opt_tokens)
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    state_ids = tok(state.replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
    st = state_ids[:room]
    ids = ids + st + [tok.sep_token_id]
    ids, markers = ids[:max_len], [m for m in markers if m < max_len]
    return ids, markers, len(st) < len(state_ids)


class _DynamicMultiheadAttention(nn.MultiheadAttention):
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if (need_weights or self.in_proj_weight is None or self.bias_k is not None
                or self.bias_v is not None
                or (attn_mask is not None and attn_mask.dtype != torch.bool)):
            return super().forward(query, key, value, key_padding_mask=key_padding_mask,
                                   need_weights=need_weights, attn_mask=attn_mask,
                                   average_attn_weights=average_attn_weights, is_causal=is_causal)
        if self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        if query is key is value:
            q, k, v = (part.unflatten(-1, (self.num_heads, self.head_dim)).permute(1, 2, 0, 3)
                       for part in F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1))
        else:
            embed_dim = query.shape[-1]
            wq, wk, wv = self.in_proj_weight.split(embed_dim, dim=0)
            bq, bk, bv = ((None, None, None) if self.in_proj_bias is None
                          else self.in_proj_bias.split(embed_dim, dim=0))
            q, k, v = (
                F.linear(t, w, b).unflatten(-1, (self.num_heads, self.head_dim)).permute(1, 2, 0, 3)
                for t, w, b in ((query, wq, bq), (key, wk, bk), (value, wv, bv))
            )
        mask = None
        if attn_mask is not None:
            mask = ~attn_mask
        if key_padding_mask is not None:
            keep = key_padding_mask[:, None, None, :] == 0
            mask = keep if mask is None else mask & keep
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal and mask is None,
                                              dropout_p=self.dropout if self.training else 0.0)
        attn = attn.permute(2, 0, 1, 3).flatten(-2)
        if self.batch_first:
            attn = attn.transpose(0, 1)
        return self.out_proj(attn), None


class DecisionModel(nn.Module):
    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        layer.self_attn = _DynamicMultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, -1e4)


def _apply_rope_config(ecfg) -> None:
    rope = getattr(ecfg, "rope_parameters", None)
    if not isinstance(rope, dict):
        return
    flat = rope.get("rope_theta")
    for layer_type, attr in (("full_attention", "global_rope_theta"), ("sliding_attention", "local_rope_theta")):
        params = rope.get(layer_type)
        theta = params.get("rope_theta") if isinstance(params, dict) else flat
        if theta is not None and hasattr(ecfg, attr):
            setattr(ecfg, attr, float(theta))


def fix_tokenizer_config(model_dir: str) -> None:
    arq = os.path.join(model_dir, "tokenizer", "tokenizer_config.json")
    if not os.path.exists(arq):
        return
    with open(arq) as fh:
        tcfg = json.load(fh)
    mudou = False
    if tcfg.get("tokenizer_class") in (None, "TokenizersBackend"):
        tcfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        tcfg.pop("backend", None)
        tcfg.pop("is_local", None)
        mudou = True
    extra = tcfg.get("extra_special_tokens")
    if isinstance(extra, list):
        tcfg["extra_special_tokens"] = {"extra_%d" % i: t for i, t in enumerate(extra)}
        mudou = True
    if mudou:
        with open(arq, "w") as fh:
            json.dump(tcfg, fh, indent=2)


class _EmbeddingParcial(nn.Module):
    """nn.Embedding só com as linhas dos tokens que as sequências usam, guardadas em fp16 (como no
    arquivo) e lidas em fp32: o mesmo valor que o GPU usa (fp32 carregado do fp16). A conta do forward
    não muda (embedding é seleção de linha); a VM do Docker tem pouca memória livre e a tabela inteira
    (256 mil x 768) em fp32 são 786 MB. Buffer, não parâmetro, para o `dtype` do encoder seguir fp32."""

    def __init__(self, vocab: int, dim: int, linhas: list):
        super().__init__()
        mapa = torch.full((vocab,), -1, dtype=torch.long)
        mapa[torch.tensor(linhas, dtype=torch.long)] = torch.arange(len(linhas))
        self.register_buffer("mapa", mapa, persistent=False)
        self.register_buffer("weight", torch.empty(len(linhas), dim, dtype=torch.float16))
        self.linhas = linhas

    def forward(self, ids):
        j = self.mapa[ids]
        if bool((j < 0).any()):
            raise ValueError("token fora das linhas carregadas")
        return F.embedding(j, self.weight).float()


def clamp_temperature(t) -> float:
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if t != t or t in (float("inf"), float("-inf")):
        return 1.0
    return min(TEMP_MAX, max(TEMP_MIN, t))


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % ({v: n for n, v in QTYPES.items()}[int(qtype)], size)


EMB = "encoder.embeddings.tok_embeddings.weight"


class Pontuador:
    """P(A) da pergunta `choice` de duas opções, em fp32, com a temperatura do checkpoint (a conta de
    `Agent._decode_answers`). Uso único: `__call__` tokeniza tudo, solta o tokenizador e só então monta
    o modelo com as linhas de embedding usadas, carregando os pesos tensor a tensor com a conferência
    de `load_state_dict(strict=True)` (mesmas chaves, mesmas formas)."""

    def __init__(self, model_dir: str, pergunta: dict, threads: int = 4):
        from transformers import AutoTokenizer

        torch.set_num_threads(threads)
        self.dir = model_dir
        fix_tokenizer_config(model_dir)
        with open(os.path.join(model_dir, "rl_agent_config.json")) as fh:
            self.cfg = json.load(fh)
        self.tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
        crit = pergunta["criteria"]
        self.q = {"t": pergunta["type"], "ins": pergunta["instructions"], "crit": crit}
        tbo = {k: clamp_temperature(v) for k, v in self.cfg.get("temperature_by_options", {}).items()}
        temps = [clamp_temperature(t) for t in self.cfg.get("temperature", [1.0, 1.0, 1.0])]
        self.t = tbo.get(temp_bucket(QTYPES["choice"], len(crit)), temps[QTYPES["choice"]])
        self.max_len, self.head_max_len = self.cfg.get("max_len", 512), self.cfg.get("head_max_len", 192)

    def _modelo(self, linhas: list) -> DecisionModel:
        from safetensors import safe_open
        from transformers import AutoConfig, AutoModel
        from transformers.initialization import no_init_weights

        ecfg = AutoConfig.from_pretrained(os.path.join(self.dir, "encoder"))
        _apply_rope_config(ecfg)
        with no_init_weights():
            enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
        vocab, dim = enc.embeddings.tok_embeddings.num_embeddings, enc.embeddings.tok_embeddings.embedding_dim
        enc.embeddings.tok_embeddings = _EmbeddingParcial(vocab, dim, linhas)
        model = DecisionModel(enc, self.cfg.get("head_layers", 2), len(self.cfg.get("act_costs", {})) + 1)
        alvo = {**dict(model.named_parameters()), **dict(model.named_buffers())}
        esperadas = set(model.state_dict())
        with safe_open(os.path.join(self.dir, "model.safetensors"), framework="pt") as fh, torch.no_grad():
            chaves = set(fh.keys())
            if chaves != esperadas:
                raise ValueError(f"chaves diferentes: faltam {sorted(esperadas - chaves)[:3]}, sobram "
                                 f"{sorted(chaves - esperadas)[:3]}")
            for k in chaves:
                if k == EMB:
                    sl = fh.get_slice(k)
                    if tuple(sl.get_shape()) != (vocab, dim):
                        raise ValueError(f"{k}: forma {sl.get_shape()} != {(vocab, dim)}")
                    lin = torch.tensor(linhas, dtype=torch.long)
                    for a in range(0, vocab, 16384):
                        sel = ((lin >= a) & (lin < a + 16384)).nonzero().flatten()
                        if len(sel):
                            bloco = sl[a:a + 16384]
                            alvo[k][sel] = bloco[lin[sel] - a].to(torch.float16)
                    continue
                t = fh.get_tensor(k)
                if tuple(t.shape) != tuple(alvo[k].shape):
                    raise ValueError(f"{k}: forma {tuple(t.shape)} != {tuple(alvo[k].shape)}")
                alvo[k].copy_(t)
        model.eval()
        model.encoder.config.reference_compile = False
        return model

    @torch.no_grad()
    def __call__(self, estados: list, lote: int = 8) -> tuple[list, int]:
        import gc
        seqs = [build_sequence(self.tok, e, self.q, self.max_len, self.head_max_len) for e in estados]
        pad = self.tok.pad_token_id
        del self.tok
        gc.collect()
        trunc = sum(s[2] for s in seqs)
        model = self._modelo(sorted({i for s in seqs for i in s[0]} | {pad}))
        ordem = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
        out = [None] * len(seqs)
        for k in range(0, len(ordem), lote):
            idx = ordem[k:k + lote]
            L = max(len(seqs[i][0]) for i in idx)
            ids = torch.full((len(idx), L), pad, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            mpos = torch.zeros((len(idx), 2), dtype=torch.long)
            for j, i in enumerate(idx):
                s, m, _ = seqs[i]
                ids[j, :len(s)] = torch.tensor(s)
                att[j, :len(s)] = 1
                mpos[j] = torch.tensor(m)
            lg = model(ids, att, mpos, torch.ones((len(idx), 2), dtype=torch.bool),
                       torch.full((len(idx),), QTYPES["choice"], dtype=torch.long)).double()
            p = torch.softmax(lg / self.t, -1)[:, 0]
            for j, i in enumerate(idx):
                out[i] = float(p[j])
        return out, trunc
