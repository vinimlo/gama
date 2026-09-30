# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "laya==0.3.22",
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
"""Fine-tune do Laya multilíngue como verificador de candidatos, no HF Jobs (L4).

CAMINHO DE TREINO: o da biblioteca. O laya 0.3.22 não tem função de treino pronta; o caminho
documentado (README do modelo e do GitHub, seção Fine-Tuning) é o notebook
`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`, que monta os itens com
`laya.common.build_sequence`, carrega o modelo com `laya.common.build_model` + `load_state_dict`,
treina o encoder e a cabeça de decisão com RLCD (ruído gaussiano nos logits, recompensa
`laya.common.proper_reward`, vantagem de grupo) somado à entropia cruzada suave, ajusta uma
temperatura por tipo de pergunta num lote separado e grava o checkpoint no formato que
`laya.Agent` lê. Este script é esse laço, numa GPU em vez de duas (sem DDP), com bf16 (a
`amp_dtype` do checkpoint; o notebook usa fp16 + GradScaler porque a T4 não tem bf16).

PERGUNTA: o formato de treino do Laya exige pergunta (a sequência é pergunta + opções + estado e
a decisão sai dos marcadores das opções), então vale a f3 do experimento zero-shot
(`bench/controles/verificador/laya_job.py`, conferida igual pelo CLI local): escolha entre A (é
citação completa com as bordas certas) e B. Estado = o campo `estado` do candidato (300
caracteres de cada lado, candidato entre [[ e ]]). Alvo: [1, 0] se `rotulo` = 1, senão [0, 1].
Score do candidato = P(A) com a temperatura ajustada.

ESCOLHAS SÓ NA VALIDAÇÃO INTERNA (validacao.jsonl; as 305 e as 172 não entram aqui):
    lr do encoder numa grade {2,5e-5 (o do notebook), 1e-5}; o resto fixo no do notebook
    (lr da cabeça 1e-4, AdamW wd 0,01, cosseno até 1e-6, clip 1,0, lote efetivo 64, grupo 4,
    sigma 0,4 -> 0,1, pesos da recompensa 0,75/1,0, entropia cruzada com peso 1,0);
    early stopping: avaliação a cada meia época, Brier (T = 1) na validação, paciência 2, no
    máximo 3 épocas; vence a configuração com o menor Brier de validação. (A primeira versão usava
    a perda log; o job 6abc46fd foi cancelado na 1ª época porque ela subiu de 0,25 para 0,54
    enquanto Brier (0,065 -> 0,054) e AUROC (0,960 -> 0,973) melhoravam: poucos erros muito
    confiantes, que a temperatura corrige depois, dominavam a perda log. O Brier é regra de
    pontuação própria e limitada. A troca usa só a validação interna.)
    temperatura: `fit_one_temp` do notebook nos logits da validação (pesos já em fp16, como
    gravados), limitada ao intervalo que o runtime aplica ([0,5; 5]).

    hf jobs uv run --flavor l4x1 --timeout 90m --secrets HF_TOKEN \\
        bench/controles/verificador_treinado/laya_ft/treinar_job.py [--limite N --epocas 1 --sem-upload]
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

DADOS = "vinimlo/gama-exp-verificador-dados"
DADOS_REVISAO = "24eede0c83a8530b2d6531e5000d7a0abf66f7f6"
DADOS_SHA = "saidas/bench/controles/verificador_treinado/dados/"   # prefixo das chaves de meta.json["sha256"]
LAYA_REPO = "convaiinnovations/laya"
LAYA_REVISAO = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
SUBPASTA = "multilingual"
DESTINO = "vinimlo/gama-exp-verif-laya_ft"
SEMENTE = 0

# Pergunta f3 do zero-shot, copiada de bench/controles/verificador/laya_job.py (o CLI local confere).
INSTRUCOES = (
    "O trecho entre [[ e ]] é uma citação completa de jurisprudência (processo numerado, súmula, "
    "tema ou julgado identificado por tribunal, ano e relator) ou de dispositivo de lei com o "
    "diploma, com as bordas certas? Bordas certas: começa na classe processual (REsp, AgInt, Rcl, "
    "APL), em Súmula, Tema, art. ou no substantivo da referência (julgado, precedente, acórdão); "
    "artigo ou preposição antes fica fora; pontuação depois fica fora; número de processo vai até "
    "a UF; artigo de lei inclui o nome do diploma; referência a julgado termina no nome do relator."
)
PERGUNTA = {"type": "choice", "instructions": INSTRUCOES,
            "criteria": {"A": "sim, o trecho marcado é exatamente uma citação completa "
                              "com as bordas certas",
                         "B": "não, o trecho marcado não é uma citação completa ou tem "
                              "as bordas erradas"}}

# Hiperparâmetros do notebook (lote efetivo 64 = 8 x 2 GPUs x 4 lá; 16 x 4 aqui).
MICRO, ACUMULA = 16, 4
GRUPO, SIGMA_INI, SIGMA_FIM = 4, 0.4, 0.1
LR_CABECA, DECAIMENTO, CLIP = 1.0e-4, 0.01, 1.0
W_SPH, W_RPS, PESO_CE = 0.75, 1.0, 1.0
GRADE_LR_ENCODER = [2.5e-5, 1.0e-5]
EPOCAS_MAX, AVALIACOES_POR_EPOCA, PACIENCIA = 3, 2, 2


def _sha(caminho) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _auroc(pares: list) -> float | None:
    """AUROC por postos médios (a mesma conta de bench.controles.verificador.nucleo.auroc)."""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, help="ensaio: só N exemplos de treino e N/4 de validação")
    ap.add_argument("--epocas", type=int, default=EPOCAS_MAX)
    ap.add_argument("--grade", default=",".join(str(x) for x in GRADE_LR_ENCODER), help="lr do encoder")
    ap.add_argument("--sem-upload", action="store_true")
    a = ap.parse_args()

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import numpy as np
    import torch
    import transformers
    import laya
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from laya.agent import Agent, _fix_tokenizer_config, _load_tokenizer
    from laya.common import QTYPES, build_model, build_sequence, clamp_temperature, collate_items, proper_reward
    from safetensors.torch import load_file, save_file

    t_job = time.perf_counter()
    dev = torch.device("cuda")
    random.seed(SEMENTE)
    np.random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)

    # ------------------------------------------------------------ dados (revisão fixa, sha256 conferido)
    arqs = {n: pathlib.Path(hf_hub_download(DADOS, n, repo_type="dataset", revision=DADOS_REVISAO))
            for n in ("meta.json", "treino.jsonl", "validacao.jsonl")}
    meta_dados = json.loads(arqs["meta.json"].read_text(encoding="utf-8"))
    shas = {}
    for n in ("treino.jsonl", "validacao.jsonl"):
        shas[n] = _sha(arqs[n])
        esperado = meta_dados["sha256"][DADOS_SHA + n]
        if shas[n] != esperado:
            raise SystemExit(f"{n}: sha256 {shas[n]} != {esperado} (meta.json)")
    ler = lambda p: [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]  # noqa: E731
    treino, valid = ler(arqs["treino.jsonl"]), ler(arqs["validacao.jsonl"])
    if a.limite:
        treino = random.Random(SEMENTE).sample(treino, min(a.limite, len(treino)))
        valid = random.Random(SEMENTE).sample(valid, min(max(64, a.limite // 4), len(valid)))
    print(f"treino {len(treino)} validação {len(valid)} (sha256 conferido)", flush=True)

    # ------------------------------------------------------------ checkpoint base e tokenizador
    raiz = snapshot_download(LAYA_REPO, revision=LAYA_REVISAO, allow_patterns=[
        f"{SUBPASTA}/{n}" for n in ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")])
    base = os.path.join(raiz, SUBPASTA)
    _fix_tokenizer_config(base)
    cfg_base = json.loads(pathlib.Path(base, "rl_agent_config.json").read_text())
    tok = _load_tokenizer(os.path.join(base, "tokenizer"), cfg_base)
    max_len, head_max_len = cfg_base["max_len"], cfg_base["head_max_len"]
    q = Agent._to_internal(PERGUNTA)          # {"t": "choice", "ins": ..., "crit": {...}}: o que o runtime monta
    pesos_base = load_file(os.path.join(base, "model.safetensors"))

    def itens(rs: list) -> tuple[list, dict]:
        out, trunc, tam = [], 0, []
        for r in rs:
            ids, marc, st = build_sequence(tok, r["estado"], q, max_len, head_max_len, return_truncation_stats=True)
            if len(marc) != 2:
                raise SystemExit(f"{r['cid']}: {len(marc)} marcadores")
            trunc += st["truncated"]
            tam.append(len(ids))
            y = int(r["rotulo"])
            out.append({"ids": ids, "markers": marc, "qtype": QTYPES["choice"], "target": [float(y), 1.0 - float(y)],
                        "label": 0 if y else 1, "y": y})
        tam.sort()
        return out, {"sequencias": len(out), "truncadas": trunc, "tokens_mediana": tam[len(tam) // 2],
                     "tokens_max": tam[-1]}

    t0 = time.perf_counter()
    it_tr, info_tr = itens(treino)
    it_va, info_va = itens(valid)
    print("itens", info_tr, info_va, f"{time.perf_counter() - t0:.1f}s", flush=True)
    y_va = [it["y"] for it in it_va]

    def novo_modelo():
        m = build_model(cfg_base, encoder_dir=os.path.join(base, "encoder"))
        m.load_state_dict(pesos_base, strict=True)
        return m.to(dev)

    @torch.no_grad()
    def logits_de(model, its: list, amp: bool) -> np.ndarray:
        model.eval()
        ordem = sorted(range(len(its)), key=lambda i: len(its[i]["ids"]))
        out = np.zeros((len(its), 2), dtype=np.float64)
        for k in range(0, len(ordem), 64):
            idx = ordem[k:k + 64]
            b = collate_items([[its[i]] for i in idx], tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                lg, _ = model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev),
                              b["marker_mask"].to(dev), b["qtype"].to(dev))
            out[idx] = lg.float().cpu().numpy()[:, :2]
        return out

    def metricas(lg: np.ndarray, t: float = 1.0) -> dict:
        z = lg / t
        z = z - z.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        pa = np.exp(logp[:, 0])
        y = np.array(y_va)
        perda = float(-(y * logp[:, 0] + (1 - y) * logp[:, 1]).mean())
        return {"perda_log": round(perda, 5), "brier": round(float(((pa - y) ** 2).mean()), 5),
                "acuracia_0_5": round(float(((pa >= 0.5) == (y == 1)).mean()), 4),
                "auroc": _auroc(list(zip(pa.tolist(), y_va)))}

    # ------------------------------------------------------------ uma configuração (laço do notebook)
    def treinar(lr_enc: float) -> dict:
        torch.manual_seed(SEMENTE)
        random.seed(SEMENTE)
        model = novo_modelo()
        model.train()
        enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
        cab = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
        opt = torch.optim.AdamW([{"params": enc, "lr": lr_enc}, {"params": cab, "lr": LR_CABECA}],
                                weight_decay=DECAIMENTO)
        por_epoca = math.ceil(len(it_tr) / MICRO)
        atualizacoes = math.ceil(por_epoca / ACUMULA) * a.epocas
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, atualizacoes), eta_min=1e-6)
        marcos = {round(por_epoca * (j + 1) / AVALIACOES_POR_EPOCA) for j in range(AVALIACOES_POR_EPOCA)}
        hist, melhor, sem_melhora, passos = [], None, 0, 0
        t_ini = time.perf_counter()
        parar = False
        for ep in range(a.epocas):
            sigma = SIGMA_INI + (SIGMA_FIM - SIGMA_INI) * (ep / max(1, a.epocas - 1))
            ordem = list(range(len(it_tr)))
            random.Random(SEMENTE * 1000 + ep).shuffle(ordem)
            opt.zero_grad(set_to_none=True)
            soma, n_mb = 0.0, 0
            for mb in range(por_epoca):
                chunk = [it_tr[i] for i in ordem[mb * MICRO:(mb + 1) * MICRO]]
                b = collate_items([[x] for x in chunk], tok.pad_token_id)
                model.train()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, act = model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev),
                                        b["marker_mask"].to(dev), b["qtype"].to(dev))
                logits = logits.float()
                mask = b["marker_mask"].to(dev)
                k = mask.sum(-1, keepdim=True).float()
                target = b["target"].to(dev)
                eps = torch.randn((GRUPO,) + logits.shape, device=dev) * sigma * mask
                eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
                z = logits.detach().unsqueeze(0) + eps
                qd = torch.softmax(z.masked_fill(~mask, -1e4), -1)
                with torch.no_grad():
                    r = proper_reward(qd, target.unsqueeze(0), b["qtype"].to(dev), mask, w_sph=W_SPH, w_rps=W_RPS)
                    adv = r - r.mean(0, keepdim=True)
                    adv = adv / (adv.std() + 1e-6)
                logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
                perda_rl = -(adv * logp).mean()
                perda_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
                perda = (perda_rl + PESO_CE * perda_ce) / ACUMULA + 0.0 * act.sum()
                perda.backward()
                soma += perda.item() * ACUMULA
                n_mb += 1
                if (mb + 1) % ACUMULA == 0 or mb + 1 == por_epoca:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    passos += 1
                    if passos % 50 == 0:
                        dt = time.perf_counter() - t_ini
                        print(f"  lr_enc {lr_enc:g} ép {ep + 1} passo {passos}/{atualizacoes} perda {soma / n_mb:.4f} "
                              f"recompensa {r.mean().item():.3f} lr {sched.get_last_lr()[0]:.2e} "
                              f"{dt:.0f}s ({(mb + 1 + ep * por_epoca) * MICRO / dt:.1f} seq/s)", flush=True)
                if mb + 1 in marcos:
                    t_av = time.perf_counter()
                    m = metricas(logits_de(model, it_va, amp=True))
                    m.update({"epoca": round(ep + (mb + 1) / por_epoca, 3), "passos": passos,
                              "perda_treino_media": round(soma / max(1, n_mb), 5),
                              "segundos": round(time.perf_counter() - t_ini, 1),
                              "segundos_avaliacao": round(time.perf_counter() - t_av, 1)})
                    hist.append(m)
                    if melhor is None or m["brier"] < melhor["metricas"]["brier"]:
                        melhor = {"metricas": m, "estado": {k2: v.detach().to("cpu", copy=True)
                                                            for k2, v in model.state_dict().items()}}
                        sem_melhora = 0
                    else:
                        sem_melhora += 1
                    print(f"  AVALIAÇÃO lr_enc {lr_enc:g} {m} {'*' if sem_melhora == 0 else ''}", flush=True)
                    if sem_melhora >= PACIENCIA:
                        parar = True
                        break
            if parar:
                break
        out = {"lr_encoder": lr_enc, "historico": hist, "melhor": melhor["metricas"],
               "parada_antecipada": parar, "atualizacoes_previstas": atualizacoes, "atualizacoes_feitas": passos,
               "segundos": round(time.perf_counter() - t_ini, 1),
               "seq_por_segundo": round(passos * ACUMULA * MICRO / max(1e-9, time.perf_counter() - t_ini), 1)}
        estado = melhor["estado"]
        del model, opt, sched
        torch.cuda.empty_cache()
        return out, estado

    grade = [float(x) for x in a.grade.split(",")]
    runs, estados = [], {}
    for lr in grade:
        out, est = treinar(lr)
        runs.append(out)
        estados[lr] = est
        print("CONFIG", lr, out["melhor"], flush=True)
    vence = min(runs, key=lambda o: (o["melhor"]["brier"], -o["lr_encoder"]))
    lr_v = vence["lr_encoder"]
    print("ESCOLHIDA lr_encoder", lr_v, flush=True)

    # ------------------------------------------------------------ pesos em fp16 (como o notebook grava) e temperatura
    sd16 = {k: v.half().contiguous() for k, v in estados[lr_v].items()}
    del estados
    model = build_model(cfg_base, encoder_dir=os.path.join(base, "encoder"))
    model.load_state_dict(sd16, strict=True)
    model.to(dev).eval()
    lg = logits_de(model, it_va, amp=False)           # fp32, pesos já arredondados para fp16

    def fit_one_temp(sel):                             # do notebook, sem mudança
        if len(sel) < 10:
            return 1.0
        kmax = max(len(z) for z, _ in sel)
        Z = torch.full((len(sel), kmax), -1e4)
        T = torch.zeros((len(sel), kmax))
        for i, (z, t) in enumerate(sel):
            Z[i, :len(z)] = torch.tensor(z)
            T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
            loss.backward()
            return loss
        opt.step(closure)
        return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())

    t_bruta = fit_one_temp([(lg[i].tolist(), it["target"]) for i, it in enumerate(it_va)])
    t_aplicada = clamp_temperature(t_bruta)
    m_t1, m_t = metricas(lg, 1.0), metricas(lg, t_aplicada)
    print("temperatura", t_bruta, "aplicada", t_aplicada, "validação T=1", m_t1, "com T", m_t, flush=True)

    saida = pathlib.Path("/tmp/laya_ft")
    saida.mkdir(parents=True, exist_ok=True)
    save_file(sd16, str(saida / "model.safetensors"))
    model.encoder.config.save_pretrained(str(saida / "encoder"))
    tok.save_pretrained(str(saida / "tokenizer"))
    cfg = dict(cfg_base)
    cfg["fine_tuned"] = True
    cfg["model_name"] = "gama-exp-verif-laya_ft"
    cfg["temperature"] = [t_aplicada, cfg_base["temperature"][1], cfg_base["temperature"][2]]
    cfg.pop("temperature_by_options", None)
    cfg["training"] = {"base": f"{LAYA_REPO}/{SUBPASTA}@{LAYA_REVISAO}", "laya": laya.__version__,
                       "recipe": "laya_finetune_typed_decisions_2xT4_kaggle.ipynb (single GPU, bf16)",
                       "lr_encoder": lr_v, "data": f"{DADOS}@{DADOS_REVISAO}"}
    (saida / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    relatorio = {
        "candidato": "laya_ft", "base": {"repo": LAYA_REPO, "subpasta": SUBPASTA, "revisao": LAYA_REVISAO},
        "dados": {"repo": DADOS, "revisao": DADOS_REVISAO, "sha256": shas, "treino": info_tr, "validacao": info_va,
                  "limite": a.limite},
        "pergunta": PERGUNTA, "pergunta_interna": q, "score": "P(A) com a temperatura de rl_agent_config.json",
        "hiper": {"micro_lote": MICRO, "acumula": ACUMULA, "lote_efetivo": MICRO * ACUMULA, "grupo": GRUPO,
                  "sigma": [SIGMA_INI, SIGMA_FIM], "lr_cabeca": LR_CABECA, "decaimento": DECAIMENTO, "clip": CLIP,
                  "w_sph": W_SPH, "w_rps": W_RPS, "peso_ce": PESO_CE, "epocas_max": a.epocas,
                  "avaliacoes_por_epoca": AVALIACOES_POR_EPOCA, "paciencia": PACIENCIA, "grade_lr_encoder": grade,
                  "amp": "bf16 no treino e nas avaliações do early stopping; fp32 na temperatura",
                  "semente": SEMENTE, "max_len": max_len, "head_max_len": head_max_len},
        "criterio": "early stopping e escolha da configuração pelo menor Brier (T = 1) na validação "
                    "interna; temperatura ajustada nos logits da validação com os pesos em fp16",
        "criterio_anterior": {"regra": "menor perda log (T = 1)", "job_cancelado": "6abc46fd031314b696342bfb",
                              "motivo": "na 1ª época a perda log subiu de 0,2508 para 0,5438 enquanto Brier "
                                        "(0,0652 -> 0,0542) e AUROC (0,9595 -> 0,9728) melhoravam"},
        "configuracoes": runs, "escolhida": {"lr_encoder": lr_v},
        "temperatura": {"bruta": t_bruta, "aplicada": t_aplicada},
        "validacao_pesos_gravados": {"T1": m_t1, "T": m_t},
        "ambiente": {"laya": laya.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
                     "dispositivo": torch.cuda.get_device_name(0), "python": platform.python_version()},
        "segundos_job": round(time.perf_counter() - t_job, 1)}
    (saida / "treino.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=1), encoding="utf-8")
    (saida / "README.md").write_text(
        "---\nlicense: apache-2.0\nbase_model: convaiinnovations/laya\nlibrary_name: laya\n---\n\n"
        "# gama-exp-verif-laya_ft (temporário, privado)\n\n"
        f"Experimento: o Laya multilíngue ({LAYA_REPO}, subpasta {SUBPASTA}, revisão {LAYA_REVISAO}) fine-tunado "
        "como verificador de candidatos de citação jurídica (pergunta de escolha A/B; score = P(A)), pelo laço do "
        "notebook de fine-tune da biblioteca, numa GPU. Não é modelo de produção. Carregar com "
        "`laya.Agent(<pasta>)`. Detalhes em treino.json.\n", encoding="utf-8")
    print(json.dumps({k: relatorio[k] for k in ("escolhida", "temperatura", "validacao_pesos_gravados")},
                     ensure_ascii=False), flush=True)

    # Confere que o checkpoint gravado carrega pelo carregador público e responde igual.
    ag = Agent(str(saida), device="cuda")
    ag.amp_enabled = False
    r0 = ag.predict(valid[0]["estado"], {"f3": PERGUNTA})["answers"]["f3"]["probabilities"]["A"]
    z = lg[0] / t_aplicada
    p0 = float(np.exp(z[0] - z.max()) / np.exp(z - z.max()).sum())
    print(f"Agent público: P(A) {r0} x laço {p0:.6f}", flush=True)
    if abs(r0 - p0) > 2e-4:
        raise SystemExit("o checkpoint gravado não reproduz o score pelo laya.Agent")

    if a.sem_upload:
        print("sem upload", flush=True)
        return 0
    api = HfApi()
    api.create_repo(DESTINO, repo_type="model", private=True, exist_ok=True)
    info = api.upload_folder(folder_path=str(saida), repo_id=DESTINO, repo_type="model",
                             commit_message=f"laya_ft: lr_encoder {lr_v:g}, T {t_aplicada:.4f}")
    print("REVISAO", info.oid, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
