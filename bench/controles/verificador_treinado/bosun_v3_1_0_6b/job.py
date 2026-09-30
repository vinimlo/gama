# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.14.0",
#   "transformers==5.17.0",
#   "peft==0.21.1",
#   "safetensors",
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
"""Verificador de candidatos treinado a partir do Bosun v3.1 0.6B (Qwen3-0.6B + adapter LoRA), no HF Jobs.

MODELO
    Qwen/Qwen3-0.6B@c1899de (a base pinada no serving.json do Bosun) carregado como
    Qwen3ForSequenceClassification (num_labels=1, cabeça `score` nova), mais o adapter LoRA do
    Bosun (Hanno-Labs/bosun-v3.1-0.6b@1d8b6f9, subpasta adapter/, r=16, alfa=32, q/k/v/o e
    gate/up/down) continuado no treino. O remote code BosunForDecision e as linhas de decisão
    (decision_embeddings.safetensors) não entram. No fim, merge_and_unload e pesos em bf16: o
    repositório guarda um Qwen3ForSequenceClassification comum, que o container local lê sem peft.

ENTRADA
    estado (janela de 300 caracteres de cada lado, candidato entre [[ e ]], campo `estado` dos
    dados e de candidatos.jsonl) + SUFIXO (a pergunta, no fim). Estado e sufixo são tokenizados em
    separado; se passar de MAX_LEN, o estado perde tokens das duas pontas (o sufixo fica inteiro).
    Leitura: o estado oculto final do último token não-pad (padding à direita, como o
    Qwen3ForSequenceClassification) vezes o peso de `score`, em fp32. Score = sigmoide do logit.

TREINO (só com os dados de treino; nada das 305, das 172, do estresse ou do dev)
    vinimlo/gama-exp-verificador-dados@24eede0 (treino.jsonl, validacao.jsonl, sha256 conferido).
    BCEWithLogits, AdamW, warmup linear (passos inteiros) e decaimento linear, lotes de 16 agrupados
    por comprimento, base congelada em bf16 e LoRA + score em fp32, semente fixa. Avaliação na validação interna a
    cada quarto de época; early stopping (paciência em avaliações) e escolha da taxa de aprendizado
    pela AUROC macro da validação interna (média das AUROC de destilacao e final_v3); empate -> menor
    perda de validação.

    hf jobs uv run --flavor l4x1 --timeout 20m --secrets HF_TOKEN job.py treinar --limite 256 --passos 30 --sem-upload
    hf jobs uv run --flavor a100-large --timeout 52m --secrets HF_TOKEN job.py treinar --lrs 1e-4,3e-4 --lote 32 \
        --minutos-por-lr 20

INFERÊNCIA (pesos do Hub numa revisão fixa)
    Score de cada um dos 13.664 candidatos de candidatos.jsonl (vinimlo/gama-goldenset@de07c93,
    sha256 conferido) -> bench/saida/controles/verificador_treinado/bosun_v3_1_0_6b.jsonl no
    gama-goldenset; e da validação interna -> scores/validacao.jsonl no repositório do modelo.

    hf jobs uv run --flavor l4x1 --timeout 30m --secrets HF_TOKEN job.py inferir --revisao-modelo <oid>
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import time

NOME = "bosun_v3_1_0_6b"
BASE, BASE_REV = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"
BOSUN, BOSUN_REV = "Hanno-Labs/bosun-v3.1-0.6b", "1d8b6f9611f9b64b514ce8b57cd86398fbc31a3b"
DADOS, DADOS_REV = "vinimlo/gama-exp-verificador-dados", "24eede0c83a8530b2d6531e5000d7a0abf66f7f6"
SHA_DADOS = {"treino.jsonl": "563627727641ce8af2ccea4e88114e571c4e367a469602be041f9f8812ce4982",
             "validacao.jsonl": "3fa89d9c824fd1b3103cf325935991880a3df2ab70607d0ad44f8fac847e305d"}
GOLDEN = "vinimlo/gama-goldenset"
CANDIDATOS, CANDIDATOS_REV = "bench/saida/controles/verificador/candidatos.jsonl", "de07c938a68cc43779f7226e5194c49807b497f2"
CANDIDATOS_SHA = "c3745f168dd45df4f8914092df779bdbe00112995c14a69bbe9683d33c48ed6e"
DESTINO_SCORES = f"bench/saida/controles/verificador_treinado/{NOME}.jsonl"
REPO_MODELO = f"vinimlo/gama-exp-verif-{NOME}"
SUFIXO = ("\n\nPergunta: o trecho entre [[ e ]] é uma citação completa de jurisprudência ou de lei, "
          "com as bordas certas?\nResposta:")
MAX_LEN = 384
SEMENTE = 13


# ---------------------------------------------------------------- comum (importável sem peft)

def sha256(caminho) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def codificar(tok, estado: str) -> tuple[list, bool]:
    """ids do estado + ids do sufixo; corta o estado nas duas pontas se passar de MAX_LEN."""
    e = tok(estado, add_special_tokens=False)["input_ids"]
    s = tok(SUFIXO, add_special_tokens=False)["input_ids"]
    orc = MAX_LEN - len(s)
    cortado = len(e) > orc
    if cortado:
        corte = len(e) - orc
        esq = corte // 2
        e = e[esq:len(e) - (corte - esq)]
    return e + s, cortado


def auroc(pares: list) -> float | None:
    """AUROC de (score, rótulo 0/1) por postos médios (o mesmo cálculo de verificador.nucleo.auroc)."""
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


def logits(qwen, seqs: list, pad_id: int, lote: int, dispositivo, autocast: bool) -> list:
    """Logit por sequência, na ordem de entrada. `qwen` é o Qwen3ForSequenceClassification (com ou sem
    LoRA injetado): estado oculto final do último token real vezes `score.weight`, em fp32."""
    import torch
    ordem = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = [0.0] * len(seqs)
    for k in range(0, len(ordem), lote):
        idx = ordem[k:k + lote]
        ids, mask = _lote([seqs[i] for i in idx], pad_id, dispositivo)
        z = _logits_lote(qwen, ids, mask, dispositivo, autocast)
        for i, v in zip(idx, z.tolist()):
            out[i] = v
    return out


def _lote(seqs: list, pad_id: int, dispositivo):
    import torch
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = 1
    return ids.to(dispositivo), mask.to(dispositivo)


def _logits_lote(qwen, ids, mask, dispositivo, autocast: bool):
    import torch
    tipo = "cuda" if str(dispositivo).startswith("cuda") else "cpu"
    with torch.autocast(tipo, dtype=torch.bfloat16, enabled=autocast):
        h = qwen.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    ult = mask.sum(1) - 1
    with torch.autocast(tipo, enabled=False):
        hl = h[torch.arange(h.shape[0], device=h.device), ult].float()
        return (hl @ qwen.score.weight.float().T).squeeze(-1)


def ler_jsonl(caminho) -> list:
    return [json.loads(x) for x in pathlib.Path(caminho).read_text(encoding="utf-8").splitlines() if x.strip()]


def metricas(regs: list, z: list) -> dict:
    """Perda, AUROC geral, por conjunto e macro, e nos elegíveis A/B das ementas."""
    import torch
    y = torch.tensor([r["rotulo"] for r in regs], dtype=torch.float32)
    perda = torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(z, dtype=torch.float32), y).item()
    pares = lambda f: [(v, r["rotulo"]) for r, v in zip(regs, z) if f(r)]  # noqa: E731
    out = {"perda": round(perda, 5), "auroc": auroc(pares(lambda r: True))}
    for c in ("destilacao", "final_v3"):
        out[f"auroc_{c}"] = auroc(pares(lambda r, c=c: r["conjunto"] == c))
    partes = [out[f"auroc_{c}"] for c in ("destilacao", "final_v3") if out[f"auroc_{c}"] is not None]
    out["auroc_macro"] = round(sum(partes) / len(partes), 4) if partes else None
    for e in ("A", "B"):
        out[f"auroc_destilacao_elegivel_{e}"] = auroc(pares(lambda r, e=e: r["conjunto"] == "destilacao" and r["elegivel"][e]))
    return out


# ---------------------------------------------------------------- treino

def _baixar_dados() -> dict:
    from huggingface_hub import hf_hub_download
    arqs = {}
    for nome, esperado in SHA_DADOS.items():
        p = pathlib.Path(hf_hub_download(DADOS, nome, repo_type="dataset", revision=DADOS_REV))
        if sha256(p) != esperado:
            raise SystemExit(f"{nome}: sha256 {sha256(p)}, esperado {esperado}")
        arqs[nome] = p
    return arqs


def _modelo(dispositivo):
    """Base pinada (bf16, congelada) + adapter do Bosun treinável + cabeça `score` treinável (treináveis em fp32)."""
    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForSequenceClassification
    qwen = AutoModelForSequenceClassification.from_pretrained(BASE, revision=BASE_REV, num_labels=1,
                                                              dtype=torch.bfloat16)
    # a config do adapter diz CAUSAL_LM; sem task_type o PeftModel genérico só injeta o LoRA (o wrapper de
    # CausalLM pede prepare_inputs_for_generation, que a classe de classificação não tem)
    cfg = LoraConfig.from_pretrained(BOSUN, subfolder="adapter", revision=BOSUN_REV)
    cfg.task_type = None
    peft = PeftModel.from_pretrained(qwen, BOSUN, subfolder="adapter", revision=BOSUN_REV, is_trainable=True,
                                     config=cfg)
    qwen.score.float()
    qwen.score.weight.requires_grad_(True)
    for p in qwen.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    nulos = [n for n, p in qwen.named_parameters() if "lora_B" in n and float(p.detach().abs().sum()) == 0.0]
    n_lora = sum(1 for n, _ in qwen.named_parameters() if "lora_" in n)
    if nulos or n_lora != 392:
        raise SystemExit(f"adapter não carregou: {len(nulos)} lora_B nulos, {n_lora} tensores LoRA (esperado 392)")
    return peft.to(dispositivo), qwen


def _lotes(comprimentos: list, lote: int, rng: random.Random) -> list:
    """Embaralha, agrupa por comprimento em blocos de 50 lotes, embaralha os lotes."""
    idx = list(range(len(comprimentos)))
    rng.shuffle(idx)
    lotes = []
    bloco = 50 * lote
    for k in range(0, len(idx), bloco):
        b = sorted(idx[k:k + bloco], key=lambda i: comprimentos[i])
        lotes += [b[j:j + lote] for j in range(0, len(b), lote)]
    rng.shuffle(lotes)
    return lotes


def _treinar_um(lr: float, treino: list, seqs_t: list, valid: list, seqs_v: list, pad_id: int, a, dispositivo) -> dict:
    import numpy as np
    import torch
    random.seed(SEMENTE)
    np.random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)
    torch.cuda.manual_seed_all(SEMENTE)
    peft, qwen = _modelo(dispositivo)
    trein = [p for p in qwen.parameters() if p.requires_grad]
    n_trein = sum(p.numel() for p in trein)
    opt = torch.optim.AdamW(trein, lr=lr, weight_decay=0.01)
    rng = random.Random(SEMENTE)
    comp = [len(s) for s in seqs_t]
    por_epoca = math.ceil(len(seqs_t) / a.lote)
    total = a.passos or por_epoca * a.epocas
    aquec = int(round(a.aquecimento * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, aquec) if s < aquec else max(0.0, (total - s) / max(1, total - aquec)))
    cada = a.avaliar_cada or max(1, por_epoca // 4)
    y_t = torch.tensor([r["rotulo"] for r in treino], dtype=torch.float32)
    hist, melhor, estado_melhor, sem_melhora, passo = [], None, None, 0, 0
    t0 = time.perf_counter()
    perda_acum, n_acum = 0.0, 0
    parar = por_tempo = False
    limite_s = 60 * a.minutos_por_lr if a.minutos_por_lr else None
    while not parar:
        for idx in _lotes(comp, a.lote, rng):
            qwen.train()
            ids, mask = _lote([seqs_t[i] for i in idx], pad_id, dispositivo)
            z = _logits_lote(qwen, ids, mask, dispositivo, False)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y_t[idx].to(dispositivo))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trein, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            passo += 1
            perda_acum += loss.item()
            n_acum += 1
            por_tempo = bool(limite_s) and time.perf_counter() - t0 > limite_s
            if passo % cada == 0 or passo == total or por_tempo:
                qwen.eval()
                with torch.inference_mode():
                    zv = logits(qwen, seqs_v, pad_id, 64, dispositivo, False)
                m = metricas(valid, zv)
                m.update({"passo": passo, "epoca": round(passo / por_epoca, 3), "perda_treino": round(perda_acum / n_acum, 5),
                          "lr": sched.get_last_lr()[0], "segundos": round(time.perf_counter() - t0, 1)})
                perda_acum, n_acum = 0.0, 0
                hist.append(m)
                print(f"lr {lr:g} " + json.dumps(m, ensure_ascii=False), flush=True)
                chave = (m["auroc_macro"] or 0.0, -m["perda"])
                if melhor is None or chave > melhor[0]:
                    melhor = (chave, m)
                    estado_melhor = {n: p.detach().to("cpu", copy=True) for n, p in qwen.named_parameters() if p.requires_grad}
                    sem_melhora = 0
                    if por_tempo:
                        print(f"lr {lr:g}: teto de tempo no passo {passo}", flush=True)
                        parar = True
                        break
                else:
                    sem_melhora += 1
                    if por_tempo:
                        print(f"lr {lr:g}: teto de tempo no passo {passo}", flush=True)
                        parar = True
                        break
                    if sem_melhora >= a.paciencia:
                        print(f"lr {lr:g}: early stopping no passo {passo}", flush=True)
                        parar = True
                        break
            if passo >= total:
                parar = True
                break
    segundos = time.perf_counter() - t0
    del peft, qwen, opt
    torch.cuda.empty_cache()
    return {"lr": lr, "passos": passo, "passos_previstos": total, "passos_por_epoca": por_epoca,
            "aquecimento_passos": aquec, "avaliar_cada": cada, "parametros_treinaveis": n_trein,
            "segundos": round(segundos, 1), "segundos_por_passo": round(segundos / max(1, passo), 4),
            "parou_por_tempo": por_tempo, "parou_por_paciencia": sem_melhora >= a.paciencia,
            "melhor": melhor[1], "historico": hist, "_estado": estado_melhor}


def cmd_treinar(a) -> None:
    import torch
    import transformers
    import peft as peft_mod
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    arqs = _baixar_dados()
    treino, valid = ler_jsonl(arqs["treino.jsonl"]), ler_jsonl(arqs["validacao.jsonl"])
    if a.limite:                     # ensaio: amostra espaçada (os arquivos vêm agrupados por conjunto)
        nv = max(64, a.limite // 4)
        treino = treino[::max(1, len(treino) // a.limite)][:a.limite]
        valid = valid[::max(1, len(valid) // nv)][:nv]
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REV)
    pad_id = tok.pad_token_id
    cod_t = [codificar(tok, r["estado"]) for r in treino]
    cod_v = [codificar(tok, r["estado"]) for r in valid]
    seqs_t, seqs_v = [s for s, _ in cod_t], [s for s, _ in cod_v]
    comp = sorted(len(s) for s in seqs_t)
    info_tok = {"pad_token": tok.pad_token, "pad_id": pad_id, "sufixo_tokens": len(tok(SUFIXO, add_special_tokens=False)["input_ids"]),
                "treino_cortados": sum(c for _, c in cod_t), "validacao_cortados": sum(c for _, c in cod_v),
                "treino_tokens_media": round(sum(comp) / len(comp), 1), "treino_tokens_max": comp[-1],
                "treino_tokens_mediana": comp[len(comp) // 2]}
    print("tokens", info_tok, flush=True)
    lrs = [float(x) for x in a.lrs.split(",")]
    resultados = []
    for lr in lrs:
        resultados.append(_treinar_um(lr, treino, seqs_t, valid, seqs_v, pad_id, a, dispositivo))
    escolhido = max(resultados, key=lambda r: (r["melhor"]["auroc_macro"] or 0.0, -r["melhor"]["perda"]))
    print("escolhido lr", escolhido["lr"], json.dumps(escolhido["melhor"], ensure_ascii=False), flush=True)

    # modelo final: o melhor estado do lr escolhido, merge do LoRA, bf16
    torch.manual_seed(SEMENTE)
    peft, qwen = _modelo("cpu")
    faltam = set(escolhido["_estado"]) - {n for n, _ in qwen.named_parameters()}
    if faltam:
        raise SystemExit(f"estado com parâmetros desconhecidos: {sorted(faltam)[:5]}")
    with torch.no_grad():
        for n, p in qwen.named_parameters():
            if n in escolhido["_estado"]:
                p.copy_(escolhido["_estado"][n])
    fundido = peft.merge_and_unload()
    fundido.config.pad_token_id = pad_id
    fundido = fundido.to(torch.bfloat16)
    pasta = pathlib.Path("/tmp") / REPO_MODELO.split("/")[1]
    pasta.mkdir(parents=True, exist_ok=True)
    fundido.save_pretrained(pasta, safe_serialization=True)
    tok.save_pretrained(pasta)
    # conferência: os pesos gravados reproduzem a validação do melhor estado
    from transformers import AutoModelForSequenceClassification
    relido = AutoModelForSequenceClassification.from_pretrained(pasta, dtype=torch.bfloat16).to(dispositivo).eval()
    with torch.inference_mode():
        zv = logits(relido, seqs_v, pad_id, 64, dispositivo, False)
    m_relido = metricas(valid, zv)
    print("relido (bf16, fundido)", json.dumps(m_relido, ensure_ascii=False), flush=True)
    for arq in ("LICENSE", "NOTICE"):
        (pasta / arq).write_bytes(pathlib.Path(hf_hub_download(BOSUN, arq, revision=BOSUN_REV)).read_bytes())
    meta = {"candidato": NOME, "base": f"{BASE}@{BASE_REV}", "adapter": f"{BOSUN}@{BOSUN_REV} (adapter/)",
            "dados": f"{DADOS}@{DADOS_REV}", "sha256_dados": SHA_DADOS, "sufixo": SUFIXO, "max_len": MAX_LEN,
            "leitura": "estado oculto final do último token real x score.weight (fp32); score = sigmoide",
            "semente": SEMENTE, "lote": a.lote, "epocas_max": a.epocas, "aquecimento": a.aquecimento,
            "paciencia_avaliacoes": a.paciencia, "minutos_por_lr": a.minutos_por_lr, "otimizador": "AdamW (weight_decay 0,01), clip 1,0, warmup linear + decaimento linear",
            "precisao": "base congelada em bf16, LoRA e score em fp32 no treino; modelo gravado fundido em bf16",
            "lora": "r=16, alfa=32, dropout 0,05, q/k/v/o/gate/up/down (config do Bosun), continuado; score treinável",
            "criterio": "AUROC macro da validação interna (média destilacao e final_v3); empate -> menor perda; "
                        "early stopping pelo mesmo critério",
            "grade_lr": lrs, "lr_escolhido": escolhido["lr"], "tokens": info_tok,
            "validacao_relida_bf16": m_relido,
            "resultados": [{k: v for k, v in r.items() if k != "_estado"} for r in resultados],
            "limite": a.limite, "passos_forcados": a.passos,
            "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft_mod.__version__,
            "dispositivo": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()}
    (pasta / "treino.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (pasta / "README.md").write_text(_cartao(meta), encoding="utf-8")
    if a.sem_upload:
        print("sem upload; arquivos:", sorted(p.name for p in pasta.iterdir()), flush=True)
        return
    api = HfApi()
    api.create_repo(REPO_MODELO, private=True, exist_ok=True)
    info = api.upload_folder(folder_path=str(pasta), repo_id=REPO_MODELO,
                             commit_message=f"verificador {NOME}: lr {escolhido['lr']:g}, passo {escolhido['melhor']['passo']}")
    print("REVISAO_MODELO", info.oid, flush=True)


def _cartao(meta: dict) -> str:
    m = meta["validacao_relida_bf16"]
    return f"""---
license: apache-2.0
base_model: {BASE}
tags: [experimento, privado, temporario]
---
# Verificador de candidatos (experimento, privado, temporário): {NOME}

Qwen3ForSequenceClassification (1 logit) = `{meta['base']}` + adapter LoRA de `{meta['adapter']}`
continuado no treino e fundido (merge_and_unload), pesos em bf16. Treinado em `{meta['dados']}`
(treino.jsonl; validação interna para early stopping e escolha da taxa de aprendizado).

Entrada: `estado` (janela de 300 caracteres de cada lado, candidato entre [[ e ]]) + sufixo
{json.dumps(meta['sufixo'], ensure_ascii=False)}, até {meta['max_len']} tokens (o estado perde as pontas se passar).
Leitura: {meta['leitura']}.

Validação interna (pesos gravados, bf16): AUROC {m['auroc']} (destilacao {m['auroc_destilacao']},
final_v3 {m['auroc_final_v3']}), perda {m['perda']}. Detalhes em `treino.json`.

Licença: Apache-2.0 (base Qwen3-0.6B e adapter Bosun v3.1 0.6B, Copyright 2026 Clause Logic Inc.; ver
LICENSE e NOTICE).
"""


# ---------------------------------------------------------------- inferência

def cmd_inferir(a) -> None:
    import torch
    import transformers
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEMENTE)
    pasta = snapshot_download(REPO_MODELO, revision=a.revisao_modelo)
    tok = AutoTokenizer.from_pretrained(pasta)
    modelo = AutoModelForSequenceClassification.from_pretrained(pasta, dtype=torch.bfloat16).to(dispositivo).eval()
    pad_id = modelo.config.pad_token_id
    arq = pathlib.Path(hf_hub_download(GOLDEN, CANDIDATOS, repo_type="dataset", revision=CANDIDATOS_REV))
    if sha256(arq) != CANDIDATOS_SHA:
        raise SystemExit(f"candidatos.jsonl com sha256 {sha256(arq)}, esperado {CANDIDATOS_SHA}")
    cands = [r for r in ler_jsonl(arq) if "meta" not in r]
    arqs = _baixar_dados()
    valid = ler_jsonl(arqs["validacao.jsonl"])
    if a.limite:
        cands, valid = cands[:a.limite], valid[:a.limite]
    saida = {}
    for nome, regs in (("candidatos", cands), ("validacao", valid)):
        estados = list(dict.fromkeys(r["estado"] for r in regs))
        cod = [codificar(tok, e) for e in estados]
        seqs = [s for s, _ in cod]
        with torch.inference_mode():
            logits(modelo, seqs[:64], pad_id, 64, dispositivo, False)          # aquecimento
            if dispositivo == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            z = logits(modelo, seqs, pad_id, a.lote, dispositivo, False)
            if dispositivo == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            um = None
            if nome == "candidatos":
                amostra = seqs[:a.amostra_lote1]
                t1 = time.perf_counter()
                for s in amostra:
                    logits(modelo, [s], pad_id, 1, dispositivo, False)
                if dispositivo == "cuda":
                    torch.cuda.synchronize()
                um = round(1000 * (time.perf_counter() - t1) / max(1, len(amostra)), 3)
        por_estado = dict(zip(estados, z))
        saida[nome] = {"regs": regs, "z": por_estado, "tempo": {
            "candidatos": len(regs), "estados_distintos": len(estados), "segundos": round(dt, 3),
            "ms_por_candidato": round(1000 * dt / max(1, len(regs)), 4),
            "ms_por_estado": round(1000 * dt / max(1, len(estados)), 4), "lote": a.lote,
            "ms_por_estado_lote_1": um, "amostra_lote_1": a.amostra_lote1 if um is not None else None,
            "cortados": sum(c for _, c in cod)}}
        print(nome, saida[nome]["tempo"], flush=True)
    zv = [saida["validacao"]["z"][r["estado"]] for r in valid]
    m_valid = metricas(valid, zv)
    print("validacao", json.dumps(m_valid, ensure_ascii=False), flush=True)
    meta = {"verificador": NOME, "modelo": f"{REPO_MODELO}@{a.revisao_modelo}", "sufixo": SUFIXO, "max_len": MAX_LEN,
            "score": "sigmoide do logit (logit em fp32 a partir do estado oculto bf16)",
            "candidatos": {"arquivo": CANDIDATOS, "revisao": CANDIDATOS_REV, "sha256": CANDIDATOS_SHA},
            "tempo": saida["candidatos"]["tempo"], "torch": torch.__version__, "transformers": transformers.__version__,
            "dispositivo": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
            "dtype": "bfloat16", "semente": SEMENTE, "limite": a.limite}
    sig = lambda v: 1.0 / (1.0 + math.exp(-v))  # noqa: E731
    destino = pathlib.Path("/tmp") / f"{NOME}.jsonl"
    with destino.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for r in cands:
            v = saida["candidatos"]["z"][r["estado"]]
            fh.write(json.dumps({"cid": r["cid"], "score": sig(v), "logit": v}) + "\n")
    meta_v = {**meta, "candidatos": {"arquivo": "validacao.jsonl", "dataset": f"{DADOS}@{DADOS_REV}",
                                     "sha256": SHA_DADOS["validacao.jsonl"]},
              "tempo": saida["validacao"]["tempo"], "metricas": m_valid}
    destino_v = pathlib.Path("/tmp") / "validacao.jsonl"
    with destino_v.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta_v}, ensure_ascii=False) + "\n")
        for r in valid:
            v = saida["validacao"]["z"][r["estado"]]
            fh.write(json.dumps({"cid": r["cid"], "score": sig(v), "logit": v}) + "\n")
    print("sha256 scores", sha256(destino), "validacao", sha256(destino_v), flush=True)
    if a.sem_upload:
        print(destino.read_text(encoding="utf-8")[:1500])
        return
    api = HfApi()
    info = api.upload_file(path_or_fileobj=str(destino), path_in_repo=DESTINO_SCORES, repo_id=GOLDEN,
                           repo_type="dataset", commit_message=f"verificador treinado: scores {NOME}")
    print("REVISAO_SCORES", info.oid, flush=True)
    info = api.upload_file(path_or_fileobj=str(destino_v), path_in_repo="scores/validacao.jsonl", repo_id=REPO_MODELO,
                           commit_message="scores da validação interna (pesos desta revisão)")
    print("REVISAO_VALIDACAO", info.oid, flush=True)


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="acao", required=True)
    p = sub.add_parser("treinar")
    p.add_argument("--lrs", default="1e-4,3e-4")
    p.add_argument("--epocas", type=int, default=3)
    p.add_argument("--lote", type=int, default=16)
    p.add_argument("--aquecimento", type=float, default=0.06, help="fração do total, convertida em passos inteiros")
    p.add_argument("--paciencia", type=int, default=4, help="avaliações sem melhora antes de parar")
    p.add_argument("--avaliar-cada", type=int, default=0, help="passos entre avaliações (0 = um quarto de época)")
    p.add_argument("--passos", type=int, default=0, help="força o total de passos (ensaio)")
    p.add_argument("--minutos-por-lr", type=float, default=0, help="teto de tempo de treino por taxa (0 = sem teto)")
    p.add_argument("--limite", type=int, default=0, help="só os N primeiros exemplos de treino (ensaio)")
    p.add_argument("--sem-upload", action="store_true")
    p = sub.add_parser("inferir")
    p.add_argument("--revisao-modelo", required=True)
    p.add_argument("--lote", type=int, default=64)
    p.add_argument("--amostra-lote1", type=int, default=200)
    p.add_argument("--limite", type=int, default=0)
    p.add_argument("--sem-upload", action="store_true")
    a = ap.parse_args()
    {"treinar": cmd_treinar, "inferir": cmd_inferir}[a.acao](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
