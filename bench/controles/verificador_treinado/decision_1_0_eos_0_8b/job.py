# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.14.0",
#   "transformers==5.17.0",
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
"""Verificador de candidatos treinado a partir do Decision-1.0-Eos-0.8B, no HF Jobs.

MODELO: o backbone do Decision-1.0-Eos-0.8B (Qwen3.5 0,8B híbrido: 18 GatedDeltaNet + 6 de atenção
plena) como `Qwen3_5TextForSequenceClassification` com num_labels=1. A cabeça de decisão nativa não é
usada (não tem código de inferência no repo). A única chave nova é `score.weight`. Saída:
sigmoid(logit) = P(o candidato casa com o ouro: mesmo tipo e IoU >= 0,5).

ENTRADA de cada candidato: o `estado` (janela de 300 caracteres de cada lado com o candidato entre
[[ e ]], o mesmo campo de candidatos.jsonl e dos dados de treino) + um sufixo fixo com o trecho, o
tipo e a forma propostos e a pergunta. O modelo é causal e o score lê a última posição, que vê o texto
inteiro. Se passar de MAX_LEN tokens, corta-se o começo da janela (o sufixo fica inteiro).
A origem (gama/régua/borda) e a confiança do Gama NÃO entram (a borda só existe no treino).

TREINO (`treinar`): full fine-tune com pesos mestres em fp32 e autocast bf16, embeddings congelados,
AdamW, aquecimento linear em passos inteiros (nunca warmup_ratio) e decaimento linear, BCEWithLogits,
lotes agrupados por tamanho, semente fixa. Avalia na validação interna (validacao.jsonl, 10% dos
documentos, nunca as 305 nem as 172) a cada `--avaliar-cada` passos; critério = maior AUROC nos
candidatos das ementas da validação (empate: menor perda); parada antecipada com `--paciencia`.
O melhor estado vira bf16, é reavaliado assim (como na inferência) e sobe para
vinimlo/gama-exp-verif-decision_1_0_eos_0_8b/<subpasta> com treino.json e validacao_scores.jsonl.

INFERÊNCIA (`inferir`): score de cada um dos 13.664 candidatos (candidatos.jsonl do dataset
vinimlo/gama-goldenset, sha256 conferido) em bf16 -> {"meta"} + {cid, score} por linha, sobe para
vinimlo/gama-goldenset em bench/saida/controles/verificador_treinado/decision_1_0_eos_0_8b.jsonl.
Também repontua a validação interna e compara com o validacao_scores.jsonl gravado no treino.

FUMAÇA (`fumaca`): poucos passos em dois subprocessos, com e sem o kernel do flash-linear-attention
(se instalado via --with), para medir a vazão e conferir que os dois caminhos dão os mesmos logits.

    hf jobs uv run --flavor l4x1 --timeout 20m --secrets HF_TOKEN \\
        --with flash-linear-attention==0.5.2 job.py fumaca --passos 40
    hf jobs uv run --flavor l4x1 --timeout 2h --secrets HF_TOKEN job.py treinar --lr 1e-5 --subpasta lr_1e-05
    hf jobs uv run --flavor l4x1 --timeout 30m --secrets HF_TOKEN job.py inferir \\
        --revisao <oid do modelo> --subpasta lr_1e-05
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
import shutil
import subprocess
import sys
import time

BASE = "llm-semantic-router/Decision-1.0-Eos-0.8B"
BASE_REV = "363c4a5e56afc115b1c78c837633956d0bbb63ab"
DADOS = "vinimlo/gama-exp-verificador-dados"
DADOS_REV = "24eede0c83a8530b2d6531e5000d7a0abf66f7f6"
DADOS_SHA = {"treino.jsonl": "563627727641ce8af2ccea4e88114e571c4e367a469602be041f9f8812ce4982",
             "validacao.jsonl": "3fa89d9c824fd1b3103cf325935991880a3df2ab70607d0ad44f8fac847e305d"}
MODELO = "vinimlo/gama-exp-verif-decision_1_0_eos_0_8b"
GOLDEN = "vinimlo/gama-goldenset"
CAND = "bench/saida/controles/verificador/candidatos.jsonl"
CAND_REV = "de07c938a68cc43779f7226e5194c49807b497f2"
CAND_SHA = "c3745f168dd45df4f8914092df779bdbe00112995c14a69bbe9683d33c48ed6e"
DESTINO = "bench/saida/controles/verificador_treinado/decision_1_0_eos_0_8b.jsonl"
NOME = "decision_1_0_eos_0_8b"
JANELA = 300
MAX_LEN = 384
SEMENTE = 13
TIPOS = {"jurisprudencia": "jurisprudência", "lei": "lei"}
PERGUNTA = ("\n\nCandidato: [[{trecho}]] ({tipo}, {forma})\n"
            "O trecho entre [[ e ]] é uma citação desse tipo, completa e com as bordas certas? Resposta:")


# ---------------------------------------------------------------- entrada

def partes(r: dict) -> tuple[str, str]:
    """(estado, sufixo). O trecho sai do próprio estado: a janela esquerda tem min(300, início)."""
    a, n, est = min(JANELA, r["inicio"]), r["fim"] - r["inicio"], r["estado"]
    if est[a:a + 2] != "[[" or est[a + 2 + n:a + 4 + n] != "]]":
        raise ValueError(f"estado sem [[ ]] no lugar esperado: {r['cid']}")
    return est, PERGUNTA.format(trecho=est[a + 2:a + 2 + n], tipo=TIPOS.get(r["tipo"], r["tipo"]), forma=r["forma"])


def codificar(tok, registros: list, max_len: int = MAX_LEN) -> tuple[list, int]:
    """ids por registro (estado + sufixo; corta o começo do estado se passar de max_len) e quantos
    foram cortados."""
    chaves = [partes(r) for r in registros]
    unicas = list(dict.fromkeys(chaves))
    e_ids = tok([e for e, _ in unicas], add_special_tokens=False)["input_ids"]
    s_ids = tok([s for _, s in unicas], add_special_tokens=False)["input_ids"]
    feitos, cortados = {}, 0
    for k, e, s in zip(unicas, e_ids, s_ids):
        corte = len(e) + len(s) - max_len
        feitos[k] = (e[corte:] if corte > 0 else e) + s
    ids = [feitos[k] for k in chaves]
    cortados = sum(len(e) + len(s) > max_len for e, s in zip(e_ids, s_ids))
    return ids, cortados


def lotes_por_tamanho(tamanhos: list, lote: int, rng: random.Random) -> list:
    """Embaralha, ordena por tamanho dentro de blocos de 50 lotes e embaralha os lotes."""
    idx = list(range(len(tamanhos)))
    rng.shuffle(idx)
    out = []
    for i in range(0, len(idx), lote * 50):
        g = sorted(idx[i:i + lote * 50], key=lambda j: tamanhos[j])
        out += [g[k:k + lote] for k in range(0, len(g), lote)]
    rng.shuffle(out)
    return out


def tensores(seqs: list, pad: int, dispositivo, multiplo: int = 32):
    """Pad à direita até o múltiplo de 32 acima do maior (poucas formas distintas para os kernels; o
    score lê o último token não-pad, e o modelo é causal: pad à direita não muda nada antes dele)."""
    import torch
    L = -(-max(map(len, seqs)) // multiplo) * multiplo
    x = torch.full((len(seqs), L), pad, dtype=torch.long)
    m = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        m[i, :len(s)] = 1
    return x.to(dispositivo), m.to(dispositivo)


def logits_de(modelo, x, m):
    return modelo(input_ids=x, attention_mask=m, use_cache=False).logits.float().squeeze(-1)


def pontuar(modelo, ids: list, pad: int, lote: int = 64, autocast: bool = False) -> list:
    """sigmoid(logit) por sequência, em lotes ordenados por tamanho (sequências iguais: um passe só)."""
    import torch
    dispositivo = next(modelo.parameters()).device
    unicas = list(dict.fromkeys(map(tuple, ids)))
    ordem = sorted(range(len(unicas)), key=lambda i: len(unicas[i]))
    p = {}
    modelo.eval()
    with torch.no_grad(), torch.autocast(device_type=dispositivo.type, dtype=torch.bfloat16, enabled=autocast):
        for k in range(0, len(ordem), lote):
            bl = [unicas[i] for i in ordem[k:k + lote]]
            x, m = tensores(bl, pad, dispositivo)
            for s, v in zip(bl, torch.sigmoid(logits_de(modelo, x, m)).tolist()):
                p[s] = v
    return [p[tuple(s)] for s in ids]


def auroc(pares: list) -> float | None:
    """AUROC de (score, rótulo 0/1) por postos médios (Mann-Whitney), igual a verificador.nucleo.auroc."""
    pos = sum(y for _, y in pares)
    neg = len(pares) - pos
    if not pos or not neg:
        return None
    ordem = sorted(pares, key=lambda q: q[0])
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


SUBCONJUNTOS = {
    "todos": lambda r: True,
    "ementas": lambda r: r["conjunto"] == "destilacao",
    "ementas_gama": lambda r: r["conjunto"] == "destilacao" and r["origem"] == "gama",
    "ementas_regua": lambda r: r["conjunto"] == "destilacao" and r["origem"] == "regua",
    "ementas_elegivel_A": lambda r: r["conjunto"] == "destilacao" and r["elegivel"]["A"],
    "ementas_elegivel_B": lambda r: r["conjunto"] == "destilacao" and r["elegivel"]["B"],
    "final_v3": lambda r: r["conjunto"] == "final_v3",
}


def metricas(regs: list, p: list) -> dict:
    out = {}
    for nome, f in SUBCONJUNTOS.items():
        pares = [(s, r["rotulo"]) for r, s in zip(regs, p) if f(r)]
        out[nome] = {"n": len(pares), "positivos": sum(y for _, y in pares), "auroc": auroc(pares)}
    eps = 1e-7
    out["perda"] = round(-sum(math.log(max(eps, s)) if r["rotulo"] else math.log(max(eps, 1 - s))
                              for r, s in zip(regs, p)) / len(regs), 5)
    fortes = [s for r, s in zip(regs, p) if r["conjunto"] == "final_v3" and r["origem"] == "gama"]
    out["final_v3_gama_abaixo_de_0_5"] = sum(s < 0.5 for s in fortes)
    out["final_v3_gama"] = len(fortes)
    return out


# ---------------------------------------------------------------- utilidades

def sha256(caminho) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def jsonl(caminho) -> list:
    return [json.loads(x) for x in pathlib.Path(caminho).read_text(encoding="utf-8").splitlines() if x.strip()]


def semear(s: int) -> None:
    import numpy as np
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dados() -> dict:
    from huggingface_hub import hf_hub_download
    out = {}
    for nome in ("treino.jsonl", "validacao.jsonl", "meta.json"):
        out[nome] = pathlib.Path(hf_hub_download(DADOS, nome, repo_type="dataset", revision=DADOS_REV))
    meta = json.loads(out["meta.json"].read_text(encoding="utf-8"))
    for nome, esperado in DADOS_SHA.items():
        sha = sha256(out[nome])
        no_meta = [v for k, v in meta.get("sha256", {}).items() if k.endswith("/" + nome)]
        if sha != esperado or no_meta != [esperado]:
            raise SystemExit(f"{nome}: sha256 {sha}, esperado {esperado} (meta.json: {no_meta})")
    return out


def ambiente() -> dict:
    import torch
    import transformers
    try:
        import fla  # noqa: F401
        fla_v = getattr(sys.modules.get("fla"), "__version__", "?")
    except Exception:
        fla_v = None
    return {"torch": torch.__version__, "transformers": transformers.__version__, "fla": fla_v,
            "dispositivo": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
            "python": platform.python_version()}


def carregar_base(dtype, pad: int, base: str = BASE, revisao: str | None = BASE_REV, subpasta: str = "backbone"):
    from transformers import AutoModelForSequenceClassification
    kw = {"revision": revisao} if revisao else {}
    m = AutoModelForSequenceClassification.from_pretrained(base, subfolder=subpasta, num_labels=1, dtype=dtype,
                                                           pad_token_id=pad, **kw)
    m.config.pad_token_id = pad
    return m


def tokenizer(base: str = BASE, revisao: str | None = BASE_REV, subpasta: str | None = None):
    from transformers import AutoTokenizer
    kw = {"revision": revisao} if revisao else {}
    if subpasta:
        kw["subfolder"] = subpasta
    return AutoTokenizer.from_pretrained(base, **kw)


# ---------------------------------------------------------------- treino

def _passos_avaliacao(a, passos_epoca: int, total: int) -> set:
    marcas = set(range(a.avaliar_cada, total + 1, a.avaliar_cada))
    marcas |= {passos_epoca * e for e in range(1, a.epocas + 1) if passos_epoca * e <= total}
    return marcas | {total}


def cmd_treinar(a) -> None:
    import torch
    t_inicio = time.perf_counter()
    semear(a.semente)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arqs = dados()
    tr, va = jsonl(arqs["treino.jsonl"]), jsonl(arqs["validacao.jsonl"])
    if a.limite:
        rng0 = random.Random(a.semente)
        rng0.shuffle(tr)
        tr = tr[:a.limite]
    if a.limite_validacao:
        va = va[:a.limite_validacao]
    tok = tokenizer(a.base, None if a.base != BASE else BASE_REV)
    pad = tok.pad_token_id
    xtr, cort_tr = codificar(tok, tr)
    xva, cort_va = codificar(tok, va)
    ytr = [float(r["rotulo"]) for r in tr]
    tam = [len(s) for s in xtr]
    print(f"treino {len(tr)} ({sum(ytr):.0f} positivos) validação {len(va)}; tokens média {sum(tam) / len(tam):.1f} "
          f"máx {max(tam)}; cortados {cort_tr}/{cort_va}; pad {pad} {tok.pad_token!r}", flush=True)

    torch.manual_seed(a.semente)
    modelo = carregar_base(torch.float32, pad, a.base, None if a.base != BASE else BASE_REV).to(dev)
    if a.congelar_embeddings:
        modelo.get_input_embeddings().weight.requires_grad_(False)
    if a.checkpointing:
        modelo.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    nomes = {n for n, p in modelo.named_parameters() if p.requires_grad}
    com_decay = [p for n, p in modelo.named_parameters() if p.requires_grad and p.ndim >= 2]
    sem_decay = [p for n, p in modelo.named_parameters() if p.requires_grad and p.ndim < 2]
    opt = torch.optim.AdamW([{"params": com_decay, "weight_decay": a.decaimento},
                             {"params": sem_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.999), eps=1e-8,
                            fused=dev.type == "cuda")
    passos_epoca = math.ceil(len(tr) / a.lote)
    total = min(a.passos, passos_epoca * a.epocas) if a.passos else passos_epoca * a.epocas
    aquec = max(1, int(round(a.aquecimento * total)))           # passos inteiros

    def fator(s: int) -> float:
        return (s + 1) / aquec if s < aquec else max(0.0, (total - s) / max(1, total - aquec))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, fator)
    marcas = _passos_avaliacao(a, passos_epoca, total)
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"parâmetros {sum(p.numel() for p in modelo.parameters()):,} treináveis {treinaveis:,}; passos {total} "
          f"({passos_epoca}/época) aquecimento {aquec}; avaliações {sorted(marcas)}", flush=True)

    rng = random.Random(a.semente)
    perda_fn = torch.nn.BCEWithLogitsLoss()
    curva, melhor, melhor_estado, paciencia, passo, parou, motivo = [], None, None, 0, 0, None, None
    tempos, tokens_vistos, t_treino = [], 0, time.perf_counter()
    acumulada, n_acum = 0.0, 0
    for epoca in range(a.epocas):
        for bl in lotes_por_tamanho(tam, a.lote, rng):
            if passo >= total:
                break
            modelo.train()
            t0 = time.perf_counter()
            x, m = tensores([xtr[i] for i in bl], pad, dev)
            y = torch.tensor([ytr[i] for i in bl], device=dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                lg = logits_de(modelo, x, m)
            perda = perda_fn(lg, y)
            perda.backward()
            torch.nn.utils.clip_grad_norm_([p for p in modelo.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            passo += 1
            acumulada += perda.item()
            n_acum += 1
            if dev.type == "cuda":
                torch.cuda.synchronize()
            tempos.append(time.perf_counter() - t0)
            tokens_vistos += int(m.sum())
            if passo % 50 == 0:
                print(f"passo {passo}/{total} época {epoca} perda {acumulada / n_acum:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{sum(tempos[-50:]) / len(tempos[-50:]):.3f} s/passo", flush=True)
            estourou = bool(a.tempo_max_min) and (time.perf_counter() - t_treino) / 60 > a.tempo_max_min
            if passo in marcas or estourou:
                t1 = time.perf_counter()
                p = pontuar(modelo, xva, pad, lote=a.lote_avaliacao, autocast=True)
                met = metricas(va, p)
                chave = (met["ementas"]["auroc"] or 0.0, -met["perda"])
                ponto = {"passo": passo, "epoca": round(passo / passos_epoca, 3), "perda_treino": round(acumulada / n_acum, 5),
                         "segundos_avaliacao": round(time.perf_counter() - t1, 1), **met}
                acumulada, n_acum = 0.0, 0
                if melhor is None or chave > melhor["chave"]:
                    melhor = {"chave": chave, "passo": passo}
                    melhor_estado = {k: v.detach().to("cpu", torch.bfloat16, copy=True) for k, v in modelo.state_dict().items()}
                    paciencia = 0
                    ponto["melhor"] = True
                else:
                    paciencia += 1
                curva.append(ponto)
                print("AVALIACAO", json.dumps(ponto, ensure_ascii=False), flush=True)
                if a.paciencia and paciencia >= a.paciencia:
                    parou, motivo = passo, "paciencia"
                    break
                if estourou:
                    parou, motivo = passo, "tempo"
                    break
        if parou or passo >= total:
            break
    segundos_treino = time.perf_counter() - t_treino
    memoria = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else None

    # melhor estado em bf16, reavaliado sem autocast (como a inferência roda)
    modelo.load_state_dict({k: v.to(torch.float32) for k, v in melhor_estado.items()})
    modelo = modelo.to(torch.bfloat16)
    if a.checkpointing:
        modelo.gradient_checkpointing_disable()
    t1 = time.perf_counter()
    p_final = pontuar(modelo, xva, pad, lote=a.lote_avaliacao)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    seg_val = time.perf_counter() - t1
    final = metricas(va, p_final)
    print("FINAL", json.dumps(final, ensure_ascii=False), flush=True)

    destino = pathlib.Path("/tmp") / "modelo"
    if destino.exists():
        shutil.rmtree(destino)
    modelo.save_pretrained(destino)
    tok.save_pretrained(destino)
    shutil.copy(__file__, destino / "job.py")
    with (destino / "validacao_scores.jsonl").open("w", encoding="utf-8") as fh:
        for r, s in zip(va, p_final):
            fh.write(json.dumps({"cid": r["cid"], "score": s, "rotulo": r["rotulo"], "conjunto": r["conjunto"],
                                 "origem": r["origem"], "elegivel": r["elegivel"], "forte": r["forte"]}) + "\n")
    info = {"modelo_base": f"{BASE}@{BASE_REV}" if a.base == BASE else a.base, "subpasta_base": "backbone",
            "classe": type(modelo).__name__, "parametros": sum(p.numel() for p in modelo.parameters()),
            "treinaveis": treinaveis, "dados": {"repo": DADOS, "revisao": DADOS_REV, "sha256": DADOS_SHA,
                                                  "treino": len(tr), "validacao": len(va)},
            "entrada": {"janela": JANELA, "max_len": MAX_LEN, "pergunta": PERGUNTA, "tipos": TIPOS,
                        "cortados_treino": cort_tr, "cortados_validacao": cort_va,
                        "tokens_media_treino": round(sum(tam) / len(tam), 1), "tokens_max_treino": max(tam)},
            "hiperparametros": {"lr": a.lr, "lote": a.lote, "epocas_max": a.epocas, "passos_total": total,
                                "aquecimento_passos": aquec, "agenda": "linear (aquecimento + decaimento até 0)",
                                "decaimento_peso": a.decaimento, "clip": 1.0, "semente": a.semente,
                                "embeddings_congelados": a.congelar_embeddings, "checkpointing": a.checkpointing,
                                "precisao": "pesos fp32 + autocast bf16; salvo em bf16",
                                "avaliar_cada": a.avaliar_cada, "paciencia": a.paciencia, "tempo_max_min": a.tempo_max_min,
                                "criterio": "maior AUROC nos candidatos das ementas da validação interna; empate: menor perda"},
            "melhor_passo": melhor["passo"], "parou_no_passo": parou, "motivo_parada": motivo, "passos_feitos": passo,
            "curva": curva, "validacao_final_bf16": final, "segundos_validacao_final": round(seg_val, 2),
            "segundos_treino": round(segundos_treino, 1), "s_por_passo_mediana": round(sorted(tempos)[len(tempos) // 2], 4),
            "tokens_por_s": round(tokens_vistos / max(1e-9, sum(tempos)), 1), "memoria_max_gb": memoria,
            "ambiente": ambiente(), "segundos_total": round(time.perf_counter() - t_inicio, 1)}
    (destino / "treino.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "curva"}, ensure_ascii=False)[:4000], flush=True)
    if a.sem_upload:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(MODELO, repo_type="model", private=True, exist_ok=True)
    c = api.upload_folder(folder_path=str(destino), path_in_repo=a.subpasta, repo_id=MODELO, repo_type="model",
                          commit_message=f"verificador {NOME}: {a.subpasta} (lr {a.lr}, melhor passo {melhor['passo']})")
    print("REVISAO", c.oid, flush=True)
    return c.oid


# ---------------------------------------------------------------- fumaça

def cmd_fumaca(a) -> None:
    """Dois subprocessos (com e sem fla) com a mesma semente: vazão e logits iniciais comparados."""
    res = {}
    for variante in ("fla", "torch"):
        env = dict(os.environ, GAMA_SEM_FLA="1" if variante == "torch" else "0")
        saida = f"/tmp/fumaca_{variante}.json"
        cmd = [sys.executable, __file__, "fumaca-um", "--passos", str(a.passos), "--lote", str(a.lote),
               "--saida", saida] + (["--checkpointing"] if a.checkpointing else [])
        t0 = time.perf_counter()
        r = subprocess.run(cmd, env=env)
        res[variante] = json.loads(pathlib.Path(saida).read_text()) if r.returncode == 0 else {"erro": r.returncode}
        res[variante]["segundos_processo"] = round(time.perf_counter() - t0, 1)
    if "logits_iniciais" in res.get("fla", {}) and "logits_iniciais" in res.get("torch", {}):
        d = [abs(x - y) for x, y in zip(res["fla"]["logits_iniciais"], res["torch"]["logits_iniciais"])]
        res["dif_max_logits_iniciais"] = max(d)
    print("FUMACA", json.dumps(res, ensure_ascii=False), flush=True)


def cmd_fumaca_um(a) -> None:
    import torch
    semear(SEMENTE)
    dev = torch.device("cuda")
    amb = ambiente()
    arqs = dados()
    tr, va = jsonl(arqs["treino.jsonl"]), jsonl(arqs["validacao.jsonl"])
    tok = tokenizer()
    pad = tok.pad_token_id
    t0 = time.perf_counter()
    xtr, cort_tr = codificar(tok, tr)
    xva, cort_va = codificar(tok, va)
    seg_tok = time.perf_counter() - t0
    tam = sorted(len(s) for s in xtr + xva)
    torch.manual_seed(SEMENTE)
    modelo = carregar_base(torch.float32, pad).to(dev)
    modelo.get_input_embeddings().weight.requires_grad_(False)
    if a.checkpointing:
        modelo.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # pad: o último token não-pad que a classe usa = último da máscara
    x, m = tensores(xva[:8], pad, dev)
    ult = (m.sum(1) - 1).tolist()
    nao_pad = (x != pad).int() * torch.arange(x.shape[1], device=dev)
    assert nao_pad.argmax(-1).tolist() == ult, "pad aparece dentro de alguma sequência"
    with torch.no_grad():
        sozinho = [logits_de(modelo, *tensores([s], pad, dev)).item() for s in xva[:8]]
        juntos = logits_de(modelo, x, m).tolist()
    dif_pad = max(abs(u - v) for u, v in zip(sozinho, juntos))
    ini = pontuar(modelo, xva[:64], pad, lote=32, autocast=True)
    lg_ini = [math.log(p / (1 - p)) if 0 < p < 1 else float("nan") for p in ini]
    opt = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=1e-5, fused=True)
    rng = random.Random(SEMENTE)
    tam_tr = [len(s) for s in xtr]

    def um_passo(bl) -> tuple[float, int]:
        x, m = tensores([xtr[i] for i in bl], pad, dev)
        y = torch.tensor([float(tr[i]["rotulo"]) for i in bl], device=dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lg = logits_de(modelo, x, m)
        perda = torch.nn.functional.binary_cross_entropy_with_logits(lg, y)
        perda.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return perda.item(), int(m.sum())

    configs = {}
    for nome, lote, ckpt in (("ckpt_16", 16, True), ("sem_ckpt_8", 8, False), ("sem_ckpt_16", 16, False)):
        if ckpt:
            modelo.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            modelo.gradient_checkpointing_disable()
        modelo.train()
        longas = sorted(range(len(xtr)), key=lambda i: -tam_tr[i])[:lote]
        bls = lotes_por_tamanho(tam_tr, lote, rng)[:a.passos] + [longas]     # o pior lote no fim
        tempos, toks, perdas, oom = [], 0, [], False
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for bl in bls:
            t0 = time.perf_counter()
            try:
                pe, nt = um_passo(bl)
            except torch.OutOfMemoryError:
                oom = True
                break
            torch.cuda.synchronize()
            tempos.append(time.perf_counter() - t0)
            toks += nt
            perdas.append(round(pe, 4))
        opt.zero_grad(set_to_none=True)
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        normais = tempos[3:-1] if not oom else tempos[3:]
        mediana = sorted(normais)[len(normais) // 2] if normais else None
        configs[nome] = {"lote": lote, "checkpointing": ckpt, "oom": oom, "passos": len(tempos),
                         "s_por_passo_mediana": round(mediana, 4) if mediana else None,
                         "s_pior_lote": round(tempos[-1], 3) if tempos and not oom else None,
                         "tokens_por_s": round(toks / sum(tempos), 1) if tempos else None,
                         "memoria_max_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                         "perdas": perdas,
                         "estimativa_epoca_min": round(mediana * math.ceil(len(tr) / lote) / 60, 1) if mediana else None}
        print(nome, json.dumps(configs[nome]), flush=True)
    modelo.gradient_checkpointing_disable()
    t0 = time.perf_counter()
    pontuar(modelo, xva[:512], pad, lote=64, autocast=True)
    torch.cuda.synchronize()
    seg_aval = time.perf_counter() - t0
    out = {"ambiente": amb, "tokenizacao_s": round(seg_tok, 1), "cortados": [cort_tr, cort_va],
           "tokens": {"media": round(sum(tam) / len(tam), 1), "p50": tam[len(tam) // 2], "p99": tam[int(len(tam) * 0.99)],
                      "max": tam[-1]},
           "dif_pad_sozinho_vs_lote": dif_pad, "logits_iniciais": lg_ini[:64], "configs": configs,
           "avaliacao_512_s": round(seg_aval, 2)}
    pathlib.Path(a.saida).write_text(json.dumps(out, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "logits_iniciais"}, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- inferência

def cmd_inferir(a) -> None:
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    semear(SEMENTE)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arq = pathlib.Path(hf_hub_download(GOLDEN, CAND, repo_type="dataset", revision=CAND_REV))
    sha = sha256(arq)
    if sha != CAND_SHA:
        raise SystemExit(f"candidatos.jsonl tem sha256 {sha}, esperado {CAND_SHA}")
    cands, meta_c = [], {}
    for r in jsonl(arq):
        if "meta" in r:
            meta_c = r["meta"]
        else:
            cands.append(r)
    if a.limite:
        cands = cands[:a.limite]
    tok = tokenizer(MODELO, a.revisao, a.subpasta)
    pad = tok.pad_token_id
    modelo = carregar_base(torch.bfloat16, pad, MODELO, a.revisao, a.subpasta).to(dev).eval()
    ids, cortados = codificar(tok, cands)
    unicas = len(set(map(tuple, ids)))
    print(f"candidatos {len(cands)} entradas distintas {unicas} cortados {cortados}", flush=True)
    pontuar(modelo, ids[:128], pad, lote=a.lote)                # aquecimento (fora da medida)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    p = pontuar(modelo, ids, pad, lote=a.lote)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    seg = time.perf_counter() - t0
    # reprodutibilidade: repontua a validação interna e compara com o gravado no treino
    arqs = dados()
    gravado = {r["cid"]: r["score"] for r in jsonl(hf_hub_download(MODELO, f"{a.subpasta}/validacao_scores.jsonl",
                                                                    revision=a.revisao))}
    va = [r for r in jsonl(arqs["validacao.jsonl"]) if r["cid"] in gravado]
    if len(va) != len(gravado):
        raise SystemExit(f"validacao_scores.jsonl tem {len(gravado)} cids, {len(va)} na validação")
    xva, _ = codificar(tok, va)
    pva = pontuar(modelo, xva, pad, lote=a.lote)
    dif_val = max(abs(gravado[r["cid"]] - s) for r, s in zip(va, pva))
    por_conj = {}
    for r in cands:
        por_conj[r["conjunto"]] = por_conj.get(r["conjunto"], 0) + 1
    meta = {"verificador": NOME, "modelo": f"{MODELO}@{a.revisao}", "subpasta": a.subpasta,
            "base": f"{BASE}@{BASE_REV}", "classe": type(modelo).__name__, "dtype": "bfloat16",
            "score": "sigmoid(logit) = P(o candidato casa com o ouro: mesmo tipo, IoU >= 0,5)",
            "entrada": {"janela": JANELA, "max_len": MAX_LEN, "pergunta": PERGUNTA, "cortados": cortados},
            "candidatos": {"repo": GOLDEN, "arquivo": CAND, "revisao": CAND_REV, "sha256": sha, "meta": meta_c,
                           "por_conjunto": por_conj},
            "entradas_distintas": unicas, "lote": a.lote, "segundos_inferencia": round(seg, 3),
            "ms_por_candidato": round(1000 * seg / len(cands), 4),
            "ms_por_entrada_distinta": round(1000 * seg / unicas, 4),
            "validacao_interna": {"auroc": metricas(va, pva), "dif_max_vs_treino": dif_val},
            "ambiente": ambiente(), "semente": SEMENTE}
    destino = pathlib.Path("/tmp") / f"{NOME}.jsonl"
    with destino.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for r, s in zip(cands, p):
            fh.write(json.dumps({"cid": r["cid"], "score": s}) + "\n")
    print(json.dumps(meta, ensure_ascii=False)[:3000], flush=True)
    print("SHA256", sha256(destino), flush=True)
    if a.sem_upload:
        return
    info = HfApi().upload_file(path_or_fileobj=str(destino), path_in_repo=DESTINO, repo_id=GOLDEN, repo_type="dataset",
                               commit_message=f"verificador treinado: {NOME} ({a.subpasta})")
    print("REVISAO", info.oid, flush=True)


def cmd_ensaio(a) -> None:
    """Ponta a ponta em miniatura: treino curto que sobe para <repo>/ensaio e inferência de 300
    candidatos lendo de lá (sem subir scores)."""
    t = argparse.Namespace(lr=1e-5, subpasta="ensaio", lote=8, lote_avaliacao=64, epocas=1, passos=40,
                           aquecimento=0.06, decaimento=0.01, avaliar_cada=20, tempo_max_min=None, paciencia=4,
                           semente=SEMENTE, congelar_embeddings=True, checkpointing=False, limite=400,
                           limite_validacao=300, base=BASE, sem_upload=False)
    oid = cmd_treinar(t)
    cmd_inferir(argparse.Namespace(revisao=oid, subpasta="ensaio", lote=64, limite=300, sem_upload=True))


def main() -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if os.environ.get("GAMA_SEM_FLA") == "1":
        sys.modules["fla"] = None                      # força o caminho torch do transformers
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="acao", required=True)
    p = sub.add_parser("treinar")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--subpasta", required=True)
    p.add_argument("--lote", type=int, default=8)
    p.add_argument("--lote-avaliacao", type=int, default=64)
    p.add_argument("--epocas", type=int, default=2)
    p.add_argument("--passos", type=int, help="teto de passos (ensaio)")
    p.add_argument("--aquecimento", type=float, default=0.06, help="fração do total, convertida em passos inteiros")
    p.add_argument("--decaimento", type=float, default=0.01)
    p.add_argument("--avaliar-cada", type=int, default=500)
    p.add_argument("--tempo-max-min", type=float, help="para o laço e salva o melhor até aqui")
    p.add_argument("--paciencia", type=int, default=4)
    p.add_argument("--semente", type=int, default=SEMENTE)
    p.add_argument("--congelar-embeddings", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--checkpointing", action="store_true")
    p.add_argument("--limite", type=int)
    p.add_argument("--limite-validacao", type=int)
    p.add_argument("--base", default=BASE)
    p.add_argument("--sem-upload", action="store_true")
    for nome in ("fumaca", "fumaca-um"):
        p = sub.add_parser(nome)
        p.add_argument("--passos", type=int, default=40)
        p.add_argument("--lote", type=int, default=16)
        p.add_argument("--checkpointing", action="store_true")
        p.add_argument("--saida", default="/tmp/fumaca.json")
    p = sub.add_parser("inferir")
    p.add_argument("--revisao", required=True)
    p.add_argument("--subpasta", required=True)
    p.add_argument("--lote", type=int, default=64)
    p.add_argument("--limite", type=int, help="só os N primeiros candidatos (ensaio)")
    p.add_argument("--sem-upload", action="store_true")
    p = sub.add_parser("ensaio")
    a = ap.parse_args()
    {"treinar": cmd_treinar, "fumaca": cmd_fumaca, "fumaca-um": cmd_fumaca_um, "inferir": cmd_inferir,
     "ensaio": cmd_ensaio}[a.acao](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
