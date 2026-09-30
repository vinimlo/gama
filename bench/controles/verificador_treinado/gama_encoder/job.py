# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.14.0",
#   "transformers==5.17.0",
#   "huggingface_hub",
#   "numpy",
# ]
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cu126" }
# ///
# -*- coding: utf-8 -*-
"""Verificador de candidatos "gama_encoder": o encoder do Gama v1.3 com uma cabeça de classificação
de sequência, fine-tunado inteiro para julgar se um candidato casa com o ouro. Roda no HF Jobs.

MODELO. `vinimlo/gama@5f924ca` (ModernBERT de 12 camadas, `ModernBertForTokenClassification`)
carregado como `ModernBertForSequenceClassification` com 2 rótulos (0 = não, 1 = sim). O encoder
(`model.*`) e o `head` (dense 768x768 + GELU + LayerNorm, que o checkpoint já tem com o mesmo nome)
vêm do Gama; o `classifier` (Linear 768 -> 2) nasce aleatório (semente 13), porque o do Gama tem 7
saídas (BIO). Tudo treina. O score de um candidato é softmax(logits)[1] em fp32.

ENTRADA. O campo `estado` de cada candidato, tal como está nos dados: 300 caracteres de cada lado com
o candidato entre [[ e ]] (`bench.controles.verificador.nucleo.estado`). Tokenizador do Gama, com
<bos>/<eos>, max_len 512 (o maior estado dos dados tem 367 tokens: nada é truncado; o job conta).

DADOS. `vinimlo/gama-exp-verificador-dados@24eede0` (privado, temporário): `treino.jsonl` (20.062
candidatos) para treinar e `validacao.jsonl` (2.108; documentos que o treino não vê) para escolher
hiperparâmetros e época. As 305, as 172, o estresse e o dev não entram em nada daqui. O sha256 de
cada arquivo é conferido contra o valor gravado no `meta.json` da montagem.

RECEITA (fixada antes de rodar). AdamW (betas 0,9/0,999, eps 1e-8, decaimento 0,01 fora de viés e
normas), lote 32 agrupado por comprimento (blocos de 50 lotes ordenados, ordem dos lotes sorteada),
aquecimento linear em 6% dos passos e decaimento linear até zero no fim de `--epocas` (4), norma do
gradiente 1,0, bf16 (autocast; pesos em fp32), semente 13 para tudo (a mesma de todos os modelos do
Gama). Avaliação na validação interna ao fim de cada época, em fp32.
    grade    lr em --lrs (2e-5, 5e-5) x pooling em --poolings (mean = o do config do Gama,
             cls), nessa ordem.
    parada   paciência 1: a configuração para na primeira época que não melhora a validação.
    critério maior AUROC na validação interna inteira (2.108 candidatos); empate em 4 casas ->
             menor log-loss; depois a ordem da grade. O mesmo critério escolhe a época dentro de
             uma configuração e a configuração vencedora.
O vencedor é salvo com `save_pretrained` (pesos + tokenizador), recarregado do disco e reavaliado
(a AUROC tem que bater), e sobe para o repositório PRIVADO `vinimlo/gama-exp-verif-gama_encoder`
com `treino.json` (histórico de todas as configurações) e um cartão.

PONTUAÇÃO. `pontuar --modelo <repo>@<rev>` baixa o modelo nessa revisão, pontua a validação interna
(de novo, a partir dos pesos gravados) e os 13.664 candidatos de
`vinimlo/gama-goldenset@de07c93:bench/saida/controles/verificador/candidatos.jsonl` (sha256
conferido), mede o tempo por candidato (`torch.cuda.synchronize()` antes e depois) e sobe os scores
para `bench/saida/controles/verificador_treinado/gama_encoder.jsonl` no mesmo dataset (1ª linha
{"meta": ...}; depois {"cid", "score"}), o formato que `python -m bench.controles.verificador rodar`
lê. Os scores da validação interna vão para o repositório do modelo, em `avaliacao/`.

    # fumaça: poucos exemplos e passos, pontua 200 candidatos, não sobe nada
    hf jobs uv run --flavor l4x1 --timeout 15m --secrets HF_TOKEN \\
        bench/controles/verificador_treinado/gama_encoder/job.py treinar --limite 512 --passos 12 --sem-upload
    # treino com a grade e publicação
    hf jobs uv run --flavor l4x1 --timeout 60m --secrets HF_TOKEN \\
        bench/controles/verificador_treinado/gama_encoder/job.py treinar
    # pontuação dos candidatos
    hf jobs uv run --flavor l4x1 --timeout 20m --secrets HF_TOKEN \\
        bench/controles/verificador_treinado/gama_encoder/job.py pontuar \\
        --modelo vinimlo/gama-exp-verif-gama_encoder@<rev>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import time

REPO_DADOS = "vinimlo/gama-exp-verificador-dados"
REV_DADOS = "24eede0c83a8530b2d6531e5000d7a0abf66f7f6"
SHA_DADOS = {"treino.jsonl": "563627727641ce8af2ccea4e88114e571c4e367a469602be041f9f8812ce4982",
             "validacao.jsonl": "3fa89d9c824fd1b3103cf325935991880a3df2ab70607d0ad44f8fac847e305d"}
BASE = "vinimlo/gama"
REV_BASE = "5f924ca2fa77c2aae6afe6d770c78ca4438f52f3"
REPO_MODELO = "vinimlo/gama-exp-verif-gama_encoder"
GOLDEN = "vinimlo/gama-goldenset"
REV_CAND = "de07c938a68cc43779f7226e5194c49807b497f2"
SHA_CAND = "c3745f168dd45df4f8914092df779bdbe00112995c14a69bbe9683d33c48ed6e"
ARQ_CAND = "bench/saida/controles/verificador/candidatos.jsonl"
DESTINO = "bench/saida/controles/verificador_treinado/gama_encoder.jsonl"
SEMENTE = 13
MAX_LEN = 512
ROTULOS = {0: "nao", 1: "sim"}
BLOCO = 50                         # lotes por bloco no agrupamento por comprimento

SUBCONJUNTOS = {
    "todos": lambda r: True,
    "destilacao": lambda r: r["conjunto"] == "destilacao",
    "destilacao_gama": lambda r: r["conjunto"] == "destilacao" and r["origem"] == "gama",
    "destilacao_regua": lambda r: r["conjunto"] == "destilacao" and r["origem"] == "regua",
    "elegivel_A": lambda r: r["elegivel"]["A"],
    "elegivel_B": lambda r: r["elegivel"]["B"],
    "final_v3": lambda r: r["conjunto"] == "final_v3",
    "final_v3_regua_e_borda": lambda r: r["conjunto"] == "final_v3" and r["origem"] != "gama",
}


# ---------------------------------------------------------------- utilidades

def _sha(caminho) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def auroc(pares: list) -> float | None:
    """AUROC de (score, rótulo 0/1) por postos médios; cópia de `verificador.nucleo.auroc`."""
    pos = sum(y for _, y in pares)
    neg = len(pares) - pos
    if not pos or not neg:
        return None
    ordem = sorted(pares, key=lambda p: p[0])
    postos, i = [0.0] * len(ordem), 0
    while i < len(ordem):
        j = i
        while j + 1 < len(ordem) and ordem[j + 1][0] == ordem[i][0]:
            j += 1
        for k in range(i, j + 1):
            postos[k] = (i + j) / 2 + 1
        i = j + 1
    soma = sum(r for r, (_, y) in zip(postos, ordem) if y)
    return round((soma - pos * (pos + 1) / 2) / (pos * neg), 4)


def _auroc_cheia(pares: list) -> float | None:
    """A mesma AUROC sem arredondar (desempate e conferência)."""
    pos = sum(y for _, y in pares)
    neg = len(pares) - pos
    if not pos or not neg:
        return None
    ordem = sorted(pares, key=lambda p: p[0])
    postos, i = [0.0] * len(ordem), 0
    while i < len(ordem):
        j = i
        while j + 1 < len(ordem) and ordem[j + 1][0] == ordem[i][0]:
            j += 1
        for k in range(i, j + 1):
            postos[k] = (i + j) / 2 + 1
        i = j + 1
    soma = sum(r for r, (_, y) in zip(postos, ordem) if y)
    return (soma - pos * (pos + 1) / 2) / (pos * neg)


def metricas(scores: list, regs: list) -> dict:
    out = {}
    for nome, f in SUBCONJUNTOS.items():
        pares = [(s, r["rotulo"]) for s, r in zip(scores, regs) if f(r)]
        if not pares:
            continue
        eps = 1e-7
        ll = -sum(math.log(max(eps, s)) if y else math.log(max(eps, 1 - s)) for s, y in pares) / len(pares)
        out[nome] = {"n": len(pares), "positivos": sum(y for _, y in pares), "auroc": auroc(pares),
                     "log_loss": round(ll, 5),
                     "acuracia_05": round(sum((s >= 0.5) == bool(y) for s, y in pares) / len(pares), 4)}
    out["todos"]["auroc_cheia"] = _auroc_cheia([(s, r["rotulo"]) for s, r in zip(scores, regs)])
    return out


def _chave(m: dict) -> tuple:
    """Critério: maior AUROC (4 casas) na validação inteira; empate -> menor log-loss."""
    a = m["todos"]["auroc"]
    return (-1.0 if a is None else a, -m["todos"]["log_loss"])


def _jsonl(caminho) -> list:
    return [json.loads(x) for x in pathlib.Path(caminho).read_text(encoding="utf-8").splitlines() if x.strip()]


def baixar_dados(nomes=("treino.jsonl", "validacao.jsonl")) -> dict:
    from huggingface_hub import hf_hub_download
    out = {}
    for n in nomes:
        arq = hf_hub_download(REPO_DADOS, n, repo_type="dataset", revision=REV_DADOS)
        sha = _sha(arq)
        if sha != SHA_DADOS[n]:
            raise SystemExit(f"{n}: sha256 {sha}, esperado {SHA_DADOS[n]}")
        out[n] = _jsonl(arq)
    return out


def tokenizar(tok, textos: list) -> tuple[list, int]:
    ids, trunc = [], 0
    for t in textos:
        x = tok(t, truncation=False)["input_ids"]
        if len(x) > MAX_LEN:
            trunc += 1
            x = tok(t, truncation=True, max_length=MAX_LEN)["input_ids"]
        ids.append(x)
    return ids, trunc


def colar(lista: list, pad: int, dev):
    import torch
    m = max(len(x) for x in lista)
    ids = torch.full((len(lista), m), pad, dtype=torch.long)
    mask = torch.zeros((len(lista), m), dtype=torch.long)
    for i, x in enumerate(lista):
        ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        mask[i, :len(x)] = 1
    return ids.to(dev), mask.to(dev)


def prever(model, ids: list, pad: int, dev, lote: int = 64) -> list:
    """P(sim) por exemplo, em fp32, com lotes ordenados por comprimento (a ordem de saída é a de entrada)."""
    import torch
    model.eval()
    ordem = sorted(range(len(ids)), key=lambda i: (len(ids[i]), i))
    out = [0.0] * len(ids)
    with torch.no_grad():
        for k in range(0, len(ordem), lote):
            idx = ordem[k:k + lote]
            x, m = colar([ids[i] for i in idx], pad, dev)
            p = torch.softmax(model(input_ids=x, attention_mask=m).logits.float(), dim=-1)[:, 1].tolist()
            for i, v in zip(idx, p):
                out[i] = float(v)
    return out


def lotes_por_comprimento(comp: list, lote: int, rng: random.Random) -> list:
    idx = list(range(len(comp)))
    rng.shuffle(idx)
    lotes = []
    for k in range(0, len(idx), lote * BLOCO):
        bloco = sorted(idx[k:k + lote * BLOCO], key=lambda i: -comp[i])
        lotes += [bloco[j:j + lote] for j in range(0, len(bloco), lote)]
    rng.shuffle(lotes)
    return lotes


def semear(s: int) -> None:
    import numpy as np
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def carregar_base(pasta: str, pooling: str):
    from transformers import AutoConfig, AutoModelForSequenceClassification
    cfg = AutoConfig.from_pretrained(pasta, num_labels=2, id2label=dict(ROTULOS),
                                     label2id={v: k for k, v in ROTULOS.items()},
                                     classifier_pooling=pooling, problem_type="single_label_classification")
    model, info = AutoModelForSequenceClassification.from_pretrained(
        pasta, config=cfg, ignore_mismatched_sizes=True, output_loading_info=True)
    carga = {k: sorted(str(x) for x in v) for k, v in info.items()}
    novos = sorted(str(x[0]) if isinstance(x, tuple) else str(x) for x in info["mismatched_keys"]) + \
        sorted(info["missing_keys"])
    if set(novos) != {"classifier.weight", "classifier.bias"} or info["unexpected_keys"]:
        raise SystemExit(f"carga inesperada: {carga}")
    return model, carga


def _dispositivo():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _versoes(dev) -> dict:
    import torch
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
            "dispositivo": torch.cuda.get_device_name(0) if dev.type == "cuda" else platform.processor()}


# ---------------------------------------------------------------- treino

def treinar_config(pasta: str, ids_tr: list, y_tr: list, ids_va: list, va: list, pad: int, lr: float,
                   pooling: str, a, dev) -> dict:
    import torch
    import torch.nn.functional as F
    semear(a.semente)
    model, carga = carregar_base(pasta, pooling)
    model.to(dev)
    decai, sem = [], []
    for n, p in model.named_parameters():
        (sem if p.ndim < 2 or "norm" in n or n.endswith(".bias") else decai).append(p)
    opt = torch.optim.AdamW([{"params": decai, "weight_decay": a.decaimento}, {"params": sem, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.999), eps=1e-8, fused=dev.type == "cuda")
    rng = random.Random(a.semente)
    comp = [len(x) for x in ids_tr]
    passos_epoca = math.ceil(len(ids_tr) / a.lote) if not a.passos else a.passos
    total = passos_epoca * a.epocas
    aquec = max(1, round(a.aquecimento * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / aquec if s < aquec else max(0.0, (total - s) / max(1, total - aquec)))
    usar_bf16 = bool(dev.type == "cuda" and a.bf16)
    hist, melhor, estado, sem_melhora = [], None, None, 0
    t_ini = time.perf_counter()
    for ep in range(1, a.epocas + 1):
        model.train()
        t0, soma, n = time.perf_counter(), 0.0, 0
        for k, idx in enumerate(lotes_por_comprimento(comp, a.lote, rng)):
            if a.passos and k >= a.passos:
                break
            x, m = colar([ids_tr[i] for i in idx], pad, dev)
            y = torch.tensor([y_tr[i] for i in idx], dtype=torch.long, device=dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=usar_bf16):
                logits = model(input_ids=x, attention_mask=m).logits
            perda = F.cross_entropy(logits.float(), y)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            soma += perda.item() * len(idx)
            n += len(idx)
            if k % 200 == 0:
                print(f"  lr {lr:g} {pooling} época {ep} passo {k}/{passos_epoca} perda {perda.item():.4f} "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        s_treino = time.perf_counter() - t0
        t1 = time.perf_counter()
        p = prever(model, ids_va, pad, dev, a.lote_inferencia)
        met = metricas(p, va)
        linha = {"epoca": ep, "perda_treino": round(soma / max(1, n), 5), "exemplos": n,
                 "segundos_treino": round(s_treino, 1), "segundos_validacao": round(time.perf_counter() - t1, 1),
                 "lr_final": sched.get_last_lr()[0], "validacao": met}
        melhorou = melhor is None or _chave(met) > _chave(melhor["validacao"])
        linha["melhorou"] = melhorou
        hist.append(linha)
        print(f"lr {lr:g} {pooling} época {ep}: perda {linha['perda_treino']} | validação AUROC "
              f"{met['todos']['auroc']} log-loss {met['todos']['log_loss']} | destilação {met.get('destilacao', {}).get('auroc')} "
              f"elegível A {met.get('elegivel_A', {}).get('auroc')} final_v3 {met.get('final_v3', {}).get('auroc')} | "
              f"{'melhorou' if melhorou else 'não melhorou'} ({s_treino:.0f}s)", flush=True)
        if melhorou:
            melhor, sem_melhora = linha, 0
            estado = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            melhor_p = p
        else:
            sem_melhora += 1
            if sem_melhora >= a.paciencia:
                break
    del model, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {"lr": lr, "pooling": pooling, "carga": carga, "historico": hist, "melhor_epoca": melhor["epoca"],
            "validacao": melhor["validacao"], "segundos": round(time.perf_counter() - t_ini, 1),
            "passos_por_epoca": passos_epoca, "passos_aquecimento": aquec, "passos_total_programados": total,
            "_estado": estado, "_p": melhor_p}


def cmd_treinar(a) -> int:
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = _dispositivo()
    t_job = time.perf_counter()
    dados = baixar_dados()
    tr, va = dados["treino.jsonl"], dados["validacao.jsonl"]
    if a.limite:
        tr, va = tr[:a.limite], va[:max(64, a.limite // 4)]
    pasta = snapshot_download(BASE, revision=REV_BASE,
                              allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"])
    tok = AutoTokenizer.from_pretrained(pasta)
    pad = tok.pad_token_id
    ids_tr, trunc_tr = tokenizar(tok, [r["estado"] for r in tr])
    ids_va, trunc_va = tokenizar(tok, [r["estado"] for r in va])
    y_tr = [int(r["rotulo"]) for r in tr]
    print(f"treino {len(tr)} ({sum(y_tr)} positivos), validação {len(va)}; truncados {trunc_tr}/{trunc_va}; "
          f"tokens médios {sum(map(len, ids_tr)) / len(ids_tr):.1f}; {_versoes(dev)}", flush=True)

    grade = [(float(lr), pool) for pool in a.poolings.split(",") for lr in a.lrs.split(",")]
    resultados, melhor = [], None
    for lr, pool in grade:
        r = treinar_config(pasta, ids_tr, y_tr, ids_va, va, pad, lr, pool, a, dev)
        estado, p = r.pop("_estado"), r.pop("_p")
        resultados.append(r)
        if melhor is None or _chave(r["validacao"]) > _chave(melhor[0]["validacao"]):
            melhor = (r, estado, p)
        else:
            del estado
        print(f"== lr {lr:g} {pool}: melhor época {r['melhor_epoca']} AUROC {r['validacao']['todos']['auroc']} "
              f"log-loss {r['validacao']['todos']['log_loss']} ({r['segundos']}s)", flush=True)
    r, estado, p_val = melhor
    print(f"escolhida: lr {r['lr']:g} pooling {r['pooling']} época {r['melhor_epoca']}", flush=True)

    # grava, recarrega do disco e confere
    destino = pathlib.Path("/tmp/verif_gama_encoder")
    destino.mkdir(parents=True, exist_ok=True)
    semear(a.semente)
    modelo, _ = carregar_base(pasta, r["pooling"])
    modelo.load_state_dict(estado)
    modelo.save_pretrained(destino)
    tok.save_pretrained(destino)
    del modelo
    recarregado = AutoModelForSequenceClassification.from_pretrained(destino).to(dev)
    p2 = prever(recarregado, ids_va, pad, dev, a.lote_inferencia)
    m2 = metricas(p2, va)
    conf = {"auroc_treino": r["validacao"]["todos"]["auroc_cheia"], "auroc_recarregado": m2["todos"]["auroc_cheia"],
            "max_dif_score": max(abs(x - y) for x, y in zip(p_val, p2))}
    conf["ok"] = abs(conf["auroc_treino"] - conf["auroc_recarregado"]) < 1e-4
    print("recarga:", conf, flush=True)
    if not conf["ok"]:
        raise SystemExit("o modelo recarregado não reproduz a validação")

    info = {"candidato": "gama_encoder", "base": f"{BASE}@{REV_BASE}", "dados": f"{REPO_DADOS}@{REV_DADOS}",
            "sha256_dados": SHA_DADOS, "treino": len(tr), "validacao": len(va), "truncados": [trunc_tr, trunc_va],
            "receita": {"lrs": a.lrs, "poolings": a.poolings, "epocas_max": a.epocas, "paciencia": a.paciencia,
                        "lote": a.lote, "aquecimento": a.aquecimento, "decaimento": a.decaimento, "bf16": a.bf16,
                        "semente": a.semente, "max_len": MAX_LEN, "otimizador": "AdamW(0.9, 0.999, eps 1e-8)",
                        "agrupamento": f"por comprimento, blocos de {BLOCO} lotes", "norma_gradiente": 1.0,
                        "limite": a.limite, "passos": a.passos},
            "criterio": "maior AUROC na validação interna inteira (4 casas); empate -> menor log-loss; depois a "
                        "ordem da grade; o mesmo critério escolhe a época (paciência 1)",
            "escolhida": {"lr": r["lr"], "pooling": r["pooling"], "epoca": r["melhor_epoca"],
                          "validacao": r["validacao"]},
            "configuracoes": resultados, "conferencia_recarga": conf, "versoes": _versoes(dev),
            "segundos_job": round(time.perf_counter() - t_job, 1)}
    (destino / "treino.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    (destino / "README.md").write_text(_cartao(info), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "configuracoes"}, ensure_ascii=False)[:3000], flush=True)
    if a.sem_upload:
        _fumaca_pontuar(recarregado, tok, dev, a)
        return 0
    api = HfApi()
    api.create_repo(a.saida, repo_type="model", private=True, exist_ok=True)
    c = api.upload_folder(folder_path=str(destino), repo_id=a.saida, repo_type="model",
                          commit_message=f"verificador gama_encoder: lr {r['lr']:g}, {r['pooling']}, "
                                         f"época {r['melhor_epoca']}")
    print("REVISAO_MODELO", c.oid, flush=True)
    return 0


def _fumaca_pontuar(model, tok, dev, a) -> None:
    from huggingface_hub import hf_hub_download
    arq = hf_hub_download(GOLDEN, ARQ_CAND, repo_type="dataset", revision=REV_CAND)
    if _sha(arq) != SHA_CAND:
        raise SystemExit("candidatos.jsonl com sha256 diferente")
    regs = [r for r in _jsonl(arq) if "meta" not in r][:200]
    ids, _ = tokenizar(tok, [r["estado"] for r in regs])
    p = prever(model, ids, tok.pad_token_id, dev, a.lote_inferencia)
    print("fumaça pontuar:", [(r["cid"], round(s, 4)) for r, s in zip(regs[:5], p[:5])], flush=True)


def _cartao(info: dict) -> str:
    e = info["escolhida"]
    return f"""---
license: other
tags: [experimento, privado, temporario]
---
# gama-exp-verif-gama_encoder (experimento privado e temporário)

Verificador de candidatos de citação: o encoder do Gama v1.3 (`{info['base']}`) com uma cabeça de
classificação de sequência (2 rótulos: nao, sim), fine-tunado inteiro. Entrada: 300 caracteres de
cada lado com o candidato entre [[ e ]]. Score = P(sim).

Não é modelo de produção e não faz parte de nenhuma submissão. Pode ser apagado.

- Dados: `{info['dados']}` (treino {info['treino']}, validação interna {info['validacao']})
- Escolhida na validação interna: lr {e['lr']:g}, pooling {e['pooling']}, época {e['epoca']};
  AUROC {e['validacao']['todos']['auroc']}, log-loss {e['validacao']['todos']['log_loss']}
- Critério: {info['criterio']}
- Histórico completo em `treino.json`.
"""


# ---------------------------------------------------------------- pontuação

def cmd_pontuar(a) -> int:
    import torch
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = _dispositivo()
    repo, rev = a.modelo.split("@")
    pasta = snapshot_download(repo, revision=rev)
    tok = AutoTokenizer.from_pretrained(pasta)
    model = AutoModelForSequenceClassification.from_pretrained(pasta).to(dev).eval()
    pad = tok.pad_token_id
    treino = json.loads(pathlib.Path(pasta, "treino.json").read_text(encoding="utf-8"))

    # validação interna, a partir dos pesos gravados
    va = baixar_dados(("validacao.jsonl",))["validacao.jsonl"]
    ids_va, _ = tokenizar(tok, [r["estado"] for r in va])
    p_va = prever(model, ids_va, pad, dev, a.lote)
    m_va = metricas(p_va, va)
    print("validação interna:", json.dumps(m_va, ensure_ascii=False), flush=True)

    arq = hf_hub_download(GOLDEN, ARQ_CAND, repo_type="dataset", revision=REV_CAND)
    sha = _sha(arq)
    if sha != SHA_CAND:
        raise SystemExit(f"candidatos.jsonl tem sha256 {sha}, esperado {SHA_CAND}")
    linhas = _jsonl(arq)
    meta_c = next((r["meta"] for r in linhas if "meta" in r), {})
    regs = [r for r in linhas if "meta" not in r][: a.limite or None]

    # aquecimento (fora da medida), depois tokenização e modelo medidos separados
    prever(model, [x for x in tokenizar(tok, [r["estado"] for r in regs[:64]])[0]], pad, dev, a.lote)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    ids, trunc = tokenizar(tok, [r["estado"] for r in regs])
    s_tok = time.perf_counter() - t0
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    p = prever(model, ids, pad, dev, a.lote)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    s_mod = time.perf_counter() - t1
    n = len(regs)
    tempo = {"candidatos": n, "lote": a.lote, "dtype": "float32 (tf32 desligado)",
             "segundos_tokenizacao": round(s_tok, 3), "segundos_modelo": round(s_mod, 3),
             "ms_por_candidato_modelo": round(1000 * s_mod / n, 4),
             "ms_por_candidato_total": round(1000 * (s_tok + s_mod) / n, 4), "truncados": trunc}
    print("tempo:", tempo, flush=True)

    meta = {"verificador": "gama_encoder", "modelo": f"{repo}@{rev}", "base": treino.get("base"),
            "escolhida": {k: v for k, v in treino.get("escolhida", {}).items() if k != "validacao"},
            "score": "P(sim) = softmax(logits)[1] em fp32: o candidato casa com o ouro (mesmo tipo, IoU >= 0,5)",
            "candidatos": {"arquivo": ARQ_CAND, "revisao": REV_CAND, "sha256": sha,
                           "meta": {k: v for k, v in meta_c.items() if k != "codigo"}},
            "validacao_interna": {k: m_va[k] for k in ("todos", "destilacao", "elegivel_A", "elegivel_B", "final_v3")
                                  if k in m_va},
            "tempo": tempo, "versoes": _versoes(dev)}
    destino = pathlib.Path("/tmp/gama_encoder.jsonl")
    with destino.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for r, s in zip(regs, p):
            fh.write(json.dumps({"cid": r["cid"], "score": s}) + "\n")
    val = pathlib.Path("/tmp/validacao_scores.jsonl")
    with val.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": {"modelo": f"{repo}@{rev}", "dados": f"{REPO_DADOS}@{REV_DADOS}",
                                      "metricas": m_va}}, ensure_ascii=False) + "\n")
        for r, s in zip(va, p_va):
            fh.write(json.dumps({"cid": r["cid"], "rotulo": r["rotulo"], "score": s}) + "\n")
    print(f"gravados {n} scores; sha256 {_sha(destino)}", flush=True)
    if a.sem_upload:
        return 0
    api = HfApi()
    c = api.upload_file(path_or_fileobj=str(destino), path_in_repo=DESTINO, repo_id=GOLDEN, repo_type="dataset",
                        commit_message="verificador treinado: scores do gama_encoder")
    print("REVISAO_SCORES", c.oid, flush=True)
    c2 = api.upload_file(path_or_fileobj=str(val), path_in_repo="avaliacao/validacao_scores.jsonl", repo_id=repo,
                         repo_type="model", commit_message="scores da validação interna (pesos em " + rev[:7] + ")")
    print("REVISAO_VALIDACAO", c2.oid, flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="acao", required=True)
    t = sub.add_parser("treinar")
    t.add_argument("--lrs", default="2e-5,5e-5")
    t.add_argument("--poolings", default="mean,cls")
    t.add_argument("--epocas", type=int, default=4)
    t.add_argument("--paciencia", type=int, default=1)
    t.add_argument("--lote", type=int, default=32)
    t.add_argument("--lote-inferencia", type=int, default=64)
    t.add_argument("--aquecimento", type=float, default=0.06)
    t.add_argument("--decaimento", type=float, default=0.01)
    t.add_argument("--bf16", type=int, default=1)
    t.add_argument("--semente", type=int, default=SEMENTE)
    t.add_argument("--limite", type=int, default=0, help="só os N primeiros exemplos de treino (fumaça)")
    t.add_argument("--passos", type=int, default=0, help="só N passos por época (fumaça)")
    t.add_argument("--saida", default=REPO_MODELO)
    t.add_argument("--sem-upload", action="store_true")
    p = sub.add_parser("pontuar")
    p.add_argument("--modelo", required=True, help="<repo>@<revisão>")
    p.add_argument("--lote", type=int, default=64)
    p.add_argument("--limite", type=int, default=0)
    p.add_argument("--sem-upload", action="store_true")
    a = ap.parse_args()
    return {"treinar": cmd_treinar, "pontuar": cmd_pontuar}[a.acao](a)


if __name__ == "__main__":
    raise SystemExit(main())
