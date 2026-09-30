# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gliner2[local]==2.0.0",
#   "torch==2.14.0",
#   "transformers==4.57.6",
#   "sentencepiece",
#   "protobuf",
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
"""Verificador de candidatos "gliner_decide": GLiNER2.5-multi-Decide julgando cada candidato por
`classify_text`, sem treino (zs) e fine-tunado (ft). Roda no HF Jobs (L4).

MODELO. `fastino/GLiNER2.5-multi-Decide`@a35a0cd (287M, mDeBERTa-v3-base, arquitetura "boundary",
Apache-2.0), biblioteca `gliner2==2.0.0`, carregado por `AutoExtractor` a partir de um
`snapshot_download` da revisão (carregar da pasta fixa pesos e tokenizador juntos). Pesos em fp32,
TF32 desligado em toda inferência.

PERGUNTA. Uma tarefa de classificação só, `resposta`, com os rótulos curtos "sim" e "não" e a
pergunta no campo `prompt` (o padrão "pergunta sobre um trecho" do cartão do modelo). O texto é o
`estado` do candidato: 300 caracteres de cada lado, com o candidato entre [[ e ]]. O score é
P(sim): a biblioteca devolve o rótulo de maior probabilidade e a probabilidade dele (softmax sobre
os dois rótulos, temperatura 1 do checkpoint), então P(sim) = confiança se o rótulo é "sim", senão
1 - confiança. Três formulações, fixadas aqui antes de qualquer saída do modelo:
    f1  genérica: "O trecho entre [[ e ]] é uma citação de jurisprudência ou de lei?"
    f2  com o tipo do candidato: "O trecho entre [[ e ]] é {tipo}?"
    f3  com o tipo e as bordas: "O trecho entre [[ e ]] é {tipo}, completo e com as bordas certas,
        sem faltar nem sobrar texto?"
    {tipo} sai da forma do candidato (NOME_FORMA). Candidatos com a mesma pergunta vão na mesma
    chamada `batch_classify_text`; estados repetidos com a mesma pergunta são pontuados uma vez.
A escolha é SÓ nas 305: o job `zs-formulacoes` pontua apenas os 2.979 candidatos das 305 com as
três e sobe o arquivo bruto para o repositório do experimento; a escolha roda no container
(`python -m bench.controles.verificador_treinado.gliner_decide escolher`, que é o `cmd_escolher`
do CLI do verificador: maior AUROC sobre todos os candidatos das 305; empate -> a primeira). Só
depois `zs-pontuar --formulacao <f>` pontua os 13.664 candidatos e a validação interna.

FINE-TUNE (`treinar --formulacao <f>`, a formulação escolhida no zero-shot). Dados:
`vinimlo/gama-exp-verificador-dados@24eede0` (sha256 conferido): `treino.jsonl` (20.062) treina,
`validacao.jsonl` (2.108, documentos que o treino não vê) escolhe hiperparâmetro e época. As 305,
as 172, o estresse e o dev não entram em nada daqui. Cada candidato vira um registro de
classificação da gliner2 ({"input": estado, "output": {"classifications": [{"task": "resposta",
"labels": ["sim", "não"], "true_label": [...], "prompt": <pergunta>}]}}), e o treino é o
`ExtractorTrainer` da biblioteca (perda: entropia cruzada binária por rótulo, a da biblioteca).
Receita, fixada antes de rodar:
    tudo treina (encoder e cabeças); AdamW da biblioteca, decaimento 0,01, norma do gradiente 1,0,
    agendamento linear com aquecimento de 10% (padrões do TrainingConfig), lote 32, bf16 no
    autocast (pesos em fp32), TF32 desligado, semente 13.
    amostragem da biblioteca (SamplingConfig) com `synthetic_label_prob = 0`: o verificador vê
    sempre os mesmos dois rótulos, como na inferência. O resto no padrão (ordem dos rótulos
    embaralhada; com dois rótulos e sem exemplos nem descrições, o resto não age).
    grade  lr do encoder em --lrs (1e-5, 3e-5), lr das cabeças 5e-4 (padrão), nessa ordem.
    parada até --epocas (3), paciência 1: a configuração para na primeira época que não melhora.
    critério maior AUROC na validação interna inteira (4 casas); empate -> menor log-loss; depois a
           ordem da grade. O mesmo critério escolhe a época e a configuração.
A avaliação da validação roda ao fim de cada época, num gancho de `_save_checkpoint` (como no
fine-tune do GLiNER 2.5 dos controles), com a mesma função de pontuação da inferência. O vencedor
é recarregado do disco e reavaliado (a AUROC tem que bater), sobe para o repositório PRIVADO e
temporário `vinimlo/gama-exp-verif-glinerdecide` com `treino.json` e um cartão, e é baixado de
volta nessa revisão para pontuar (`pontuar`).

SAÍDAS. Scores dos 13.664 candidatos de
`vinimlo/gama-goldenset@de07c93:bench/saida/controles/verificador/candidatos.jsonl` (sha256
conferido) em `bench/saida/controles/verificador_treinado/gliner_decide_{zs,ft}.jsonl` no mesmo
dataset (1ª linha {"meta": ...}; depois {"cid", "score"}), o formato que o CLI do verificador lê.
O bruto das formulações e os scores da validação interna vão para o repositório do experimento
(`zs/` e `avaliacao/`). Nada vai para `vinimlo/gama`.

    # fumaça: API, poucos passos de treino, recarga; não sobe nada
    hf jobs uv run --flavor l4x1 --timeout 20m --secrets HF_TOKEN \\
        bench/controles/verificador_treinado/gliner_decide/job.py fumaca
    # zero-shot: as três formulações nas 305 (só elas)
    hf jobs uv run --flavor l4x1 --timeout 20m --secrets HF_TOKEN \\
        bench/controles/verificador_treinado/gliner_decide/job.py zs-formulacoes
    # zero-shot: a formulação escolhida nos 13.664 candidatos e na validação interna
    hf jobs uv run --flavor l4x1 --timeout 20m --secrets HF_TOKEN \\
        bench/controles/verificador_treinado/gliner_decide/job.py zs-pontuar --formulacao <f>
    # fine-tune com a grade, publicação e pontuação
    hf jobs uv run --flavor l4x1 --timeout <T> --secrets HF_TOKEN \\
        bench/controles/verificador_treinado/gliner_decide/job.py treinar --formulacao <f>
    # só a pontuação de um modelo publicado
    hf jobs uv run --flavor l4x1 --timeout 20m --secrets HF_TOKEN \\
        bench/controles/verificador_treinado/gliner_decide/job.py pontuar \\
        --modelo vinimlo/gama-exp-verif-glinerdecide@<rev>
"""
from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")          # sem barra de progresso no log do job

import argparse
import collections
import hashlib
import json
import math
import pathlib
import platform
import random
import shutil
import time

REPO_DADOS = "vinimlo/gama-exp-verificador-dados"
REV_DADOS = "24eede0c83a8530b2d6531e5000d7a0abf66f7f6"
SHA_DADOS = {"treino.jsonl": "563627727641ce8af2ccea4e88114e571c4e367a469602be041f9f8812ce4982",
             "validacao.jsonl": "3fa89d9c824fd1b3103cf325935991880a3df2ab70607d0ad44f8fac847e305d"}
MODELO = "fastino/GLiNER2.5-multi-Decide"
REVISAO = "a35a0cd3b7a0f00f2effc576f454cd48fa98aa5f"
REPO_EXP = "vinimlo/gama-exp-verif-glinerdecide"
GOLDEN = "vinimlo/gama-goldenset"
REV_CAND = "de07c938a68cc43779f7226e5194c49807b497f2"
SHA_CAND = "c3745f168dd45df4f8914092df779bdbe00112995c14a69bbe9683d33c48ed6e"
ARQ_CAND = "bench/saida/controles/verificador/candidatos.jsonl"
DESTINO = "bench/saida/controles/verificador_treinado/gliner_decide_{}.jsonl"
SEMENTE = 13

TAREFA = "resposta"
ROTULOS = ("sim", "não")
_PRECEDENTE = "um precedente judicial identificado por número de processo ou de recurso"
_SUMULA = "uma súmula ou um tema de tribunal"
NOME_FORMA = {"processo": _PRECEDENTE, "cnj": _PRECEDENTE, "sumula": _SUMULA, "tema": _SUMULA,
              "vaga": "uma referência a julgado sem número", "artigo": "um dispositivo de lei"}
NOME_TIPO = {"jurisprudencia": "uma citação de jurisprudência", "lei": "um dispositivo de lei"}
FORMULACOES = {
    "f1": "O trecho entre [[ e ]] é uma citação de jurisprudência ou de lei?",
    "f2": "O trecho entre [[ e ]] é {tipo}?",
    "f3": "O trecho entre [[ e ]] é {tipo}, completo e com as bordas certas, sem faltar nem sobrar texto?",
}

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


def _jsonl(caminho) -> list:
    return [json.loads(x) for x in pathlib.Path(caminho).read_text(encoding="utf-8").splitlines() if x.strip()]


def auroc(pares: list) -> float | None:
    """AUROC de (score, rótulo 0/1) por postos médios; cópia de `verificador.nucleo.auroc`."""
    v = _auroc_cheia(pares)
    return None if v is None else round(v, 4)


def _auroc_cheia(pares: list) -> float | None:
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


def semear(s: int) -> None:
    import numpy as np
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def sem_tf32() -> None:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _versoes() -> dict:
    import gliner2
    import torch
    import transformers
    return {"gliner2": gliner2.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(),
            "dispositivo": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()}


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


def baixar_candidatos() -> tuple[dict, list, str]:
    from huggingface_hub import hf_hub_download
    arq = hf_hub_download(GOLDEN, ARQ_CAND, repo_type="dataset", revision=REV_CAND)
    sha = _sha(arq)
    if sha != SHA_CAND:
        raise SystemExit(f"candidatos.jsonl tem sha256 {sha}, esperado {SHA_CAND}")
    linhas = _jsonl(arq)
    meta = next((r["meta"] for r in linhas if "meta" in r), {})
    return {k: v for k, v in meta.items() if k != "codigo"}, [r for r in linhas if "meta" not in r], sha


def pasta_base() -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(MODELO, revision=REVISAO)


def carregar(pasta: str, dispositivo: str = "cuda"):
    from gliner2 import AutoExtractor
    m = AutoExtractor.from_pretrained(pasta)
    return m.float().to(dispositivo).eval()


# ---------------------------------------------------------------- pergunta e pontuação

def nome_tipo(r: dict) -> str:
    return NOME_FORMA.get(r.get("forma")) or NOME_TIPO.get(r.get("tipo"), "uma citação jurídica")


def pergunta(form: str, r: dict) -> str:
    return FORMULACOES[form].format(tipo=nome_tipo(r))


def tarefa(q: str) -> dict:
    return {TAREFA: {"labels": list(ROTULOS), "prompt": q}}


def p_sim(res: dict) -> float:
    """P(sim) a partir do resultado de `classify_text` com `include_confidence=True`."""
    x = res[TAREFA]
    c = float(x["confidence"])
    if x["label"] == ROTULOS[0]:
        return c
    if x["label"] != ROTULOS[1]:
        raise SystemExit(f"rótulo inesperado: {x}")
    return 1.0 - c


def pontuar(modelo, regs: list, form: str, lote: int) -> tuple[list, dict]:
    """P(sim) por registro. Agrupa pela pergunta (uma chamada `batch_classify_text` por pergunta),
    pontua cada (pergunta, estado) distinto uma vez, com os estados ordenados por comprimento."""
    import torch
    por_q = collections.defaultdict(set)
    for r in regs:
        por_q[pergunta(form, r)].add(r["estado"])
    p = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for q in sorted(por_q):
        es = sorted(por_q[q], key=lambda e: (len(e), e))
        res = modelo.batch_classify_text(es, tarefa(q), batch_size=lote, include_confidence=True)
        if len(res) != len(es):
            raise SystemExit(f"{len(res)} resultados para {len(es)} textos")
        for e, x in zip(es, res):
            p[(q, e)] = p_sim(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seg = time.perf_counter() - t0
    info = {"registros": len(regs), "entradas_distintas": len(p), "perguntas": len(por_q),
            "segundos": round(seg, 3), "ms_por_entrada": round(1000 * seg / max(1, len(p)), 3),
            "ms_por_registro": round(1000 * seg / max(1, len(regs)), 3), "lote": lote}
    return [p[(pergunta(form, r), r["estado"])] for r in regs], info


def gravar_scores(destino: pathlib.Path, meta: dict, regs: list, scores: list, com_rotulo: bool = False) -> str:
    with destino.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for r, s in zip(regs, scores):
            linha = {"cid": r["cid"], "score": s}
            if com_rotulo:
                linha = {"cid": r["cid"], "rotulo": r["rotulo"], "score": s}
            fh.write(json.dumps(linha) + "\n")
    return _sha(destino)


def subir(api, arq: pathlib.Path, repo: str, caminho: str, tipo: str, mensagem: str, marca: str) -> str:
    c = api.upload_file(path_or_fileobj=str(arq), path_in_repo=caminho, repo_id=repo, repo_type=tipo,
                        commit_message=mensagem)
    print(marca, caminho, c.oid, "sha256", _sha(arq), flush=True)
    return c.oid


def _repo_exp(api) -> None:
    api.create_repo(REPO_EXP, repo_type="model", private=True, exist_ok=True)


# ---------------------------------------------------------------- zero-shot

def cmd_zs_formulacoes(a) -> int:
    """As três formulações SÓ nos candidatos das 305 (conjunto `reais`); bruto no repositório do
    experimento, no formato que `cmd_escolher` lê ({"cid", "p": {f1, f2, f3}})."""
    from huggingface_hub import HfApi
    sem_tf32()
    semear(0)
    meta_c, regs, sha = baixar_candidatos()
    regs = [r for r in regs if r["conjunto"] == "reais"][: a.limite or None]
    if any(not r["cid"].startswith("reais/") for r in regs):
        raise SystemExit("candidato fora das 305")
    modelo = carregar(pasta_base())
    p, tempos = {}, {}
    for f in FORMULACOES:
        p[f], tempos[f] = pontuar(modelo, regs, f, a.lote)
        print("formulação", f, "pontuada:", tempos[f], flush=True)
    meta = {"verificador": "gliner_decide_zs", "modelo": f"{MODELO}@{REVISAO}", "tarefa": TAREFA,
            "rotulos": list(ROTULOS), "formulacoes": FORMULACOES, "nome_forma": NOME_FORMA, "nome_tipo": NOME_TIPO,
            "score": "P(sim): confiança se o rótulo devolvido é sim, senão 1 - confiança (softmax dos dois rótulos)",
            "conjunto": "reais (305) apenas", "candidatos": {"arquivo": ARQ_CAND, "revisao": REV_CAND, "sha256": sha,
                                                               "meta": meta_c},
            "tempo": tempos, "versoes": _versoes()}
    destino = pathlib.Path("/tmp/zs_formulacoes_reais.jsonl")
    with destino.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for i, r in enumerate(regs):
            fh.write(json.dumps({"cid": r["cid"], "p": {f: p[f][i] for f in FORMULACOES}}) + "\n")
    print(f"bruto: {len(regs)} candidatos das 305; sha256 {_sha(destino)}", flush=True)
    if a.sem_upload:
        return 0
    api = HfApi()
    _repo_exp(api)
    subir(api, destino, REPO_EXP, "zs/formulacoes_reais.jsonl", "model",
          "zero-shot: três formulações nos candidatos das 305", "REVISAO_BRUTO")
    return 0


def _pontuar_tudo(modelo, variante: str, rotulo_modelo: str, form: str, extra: dict, a) -> int:
    """Validação interna + 13.664 candidatos; sobe os dois arquivos."""
    from huggingface_hub import HfApi
    va = baixar_dados(("validacao.jsonl",))["validacao.jsonl"]
    p_va, t_va = pontuar(modelo, va, form, a.lote)
    m_va = metricas(p_va, va)
    print("validação interna:", json.dumps(m_va["todos"]), t_va, flush=True)
    meta_c, regs, sha = baixar_candidatos()
    regs = regs[: a.limite or None]
    pontuar(modelo, regs[:64], form, a.lote)                         # aquecimento
    p, tempo = pontuar(modelo, regs, form, a.lote)
    print("tempo:", tempo, flush=True)
    meta = {"verificador": f"gliner_decide_{variante}", "modelo": rotulo_modelo, "base": f"{MODELO}@{REVISAO}",
            "formulacao": form, "pergunta": FORMULACOES[form], "tarefa": TAREFA, "rotulos": list(ROTULOS),
            "nome_forma": NOME_FORMA,
            "score": "P(sim) em fp32: confiança se o rótulo devolvido é sim, senão 1 - confiança; o candidato "
                     "casa com o ouro (mesmo tipo, IoU >= 0,5)",
            "candidatos": {"arquivo": ARQ_CAND, "revisao": REV_CAND, "sha256": sha, "meta": meta_c},
            "validacao_interna": {k: m_va[k] for k in ("todos", "destilacao", "elegivel_A", "elegivel_B", "final_v3")
                                  if k in m_va},
            "tempo": tempo, "tempo_validacao": t_va, "dtype": "float32 (tf32 desligado)", "versoes": _versoes(),
            **extra}
    destino = pathlib.Path(f"/tmp/gliner_decide_{variante}.jsonl")
    sha_s = gravar_scores(destino, meta, regs, p)
    val = pathlib.Path(f"/tmp/{variante}_validacao_scores.jsonl")
    gravar_scores(val, {"modelo": rotulo_modelo, "formulacao": form, "dados": f"{REPO_DADOS}@{REV_DADOS}",
                        "metricas": m_va}, va, p_va, com_rotulo=True)
    print(f"gravados {len(regs)} scores; sha256 {sha_s}", flush=True)
    if a.sem_upload:
        print("amostra:", [(r["cid"], round(s, 4)) for r, s in zip(regs[:5], p[:5])], flush=True)
        return 0
    api = HfApi()
    _repo_exp(api)
    subir(api, val, REPO_EXP, f"avaliacao/{variante}_validacao_scores.jsonl", "model",
          f"scores da validação interna ({variante})", "REVISAO_VALIDACAO")
    subir(api, destino, GOLDEN, DESTINO.format(variante), "dataset",
          f"verificador treinado: scores do gliner_decide_{variante}", "REVISAO_SCORES")
    return 0


def cmd_zs_pontuar(a) -> int:
    sem_tf32()
    semear(0)
    modelo = carregar(pasta_base())
    return _pontuar_tudo(modelo, "zs", f"{MODELO}@{REVISAO}", a.formulacao,
                         {"escolha_formulacao": "maior AUROC nas 305 (escolha.json local)"}, a)


# ---------------------------------------------------------------- fine-tune

class Parada(Exception):
    """Paciência esgotada: a configuração para aqui."""


def registros(regs: list, form: str) -> list:
    return [{"input": r["estado"],
             "output": {"classifications": [{"task": TAREFA, "labels": list(ROTULOS),
                                             "true_label": [ROTULOS[0] if r["rotulo"] else ROTULOS[1]],
                                             "prompt": pergunta(form, r)}]}} for r in regs]


def treinar_config(base: str, tr: list, va: list, form: str, lr: float, a) -> dict:
    import torch
    from gliner2 import AutoExtractor
    from gliner2.processor import SamplingConfig
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

    semear(a.semente)
    modelo = AutoExtractor.from_pretrained(base)
    modelo.float()
    modelo.processor.sampling_config = SamplingConfig(synthetic_label_prob=a.sintetico)
    saida = pathlib.Path(f"/tmp/treino_lr{lr:g}")
    shutil.rmtree(saida, ignore_errors=True)
    estado = {"hist": [], "melhor": None, "sem_melhora": 0, "t_epoca": time.perf_counter()}

    class Treinador(ExtractorTrainer):
        """O trainer da biblioteca, com a validação interna ao salvar o checkpoint de cada época."""

        def _save_checkpoint(self, name: str):
            if not name.startswith("checkpoint-epoch-"):
                return                                   # o "final" não é usado
            super()._save_checkpoint(name)
            ep = int(name.rsplit("-", 1)[1])
            s_treino = time.perf_counter() - estado["t_epoca"]
            p, t = pontuar(self.model, va, form, a.lote_inferencia)
            met = metricas(p, va)
            melhorou = estado["melhor"] is None or _chave(met) > _chave(estado["melhor"]["validacao"])
            linha = {"epoca": ep, "passo": self.global_step, "segundos_treino": round(s_treino, 1),
                     "segundos_validacao": t["segundos"], "validacao": met, "melhorou": melhorou,
                     "lr_atual": self.scheduler.get_last_lr() if self.scheduler else None}
            estado["hist"].append(linha)
            print(f"lr {lr:g} época {ep} (passo {self.global_step}, {s_treino:.0f}s): validação AUROC "
                  f"{met['todos']['auroc']} log-loss {met['todos']['log_loss']} | destilação "
                  f"{met.get('destilacao', {}).get('auroc')} elegível A {met.get('elegivel_A', {}).get('auroc')} "
                  f"final_v3 {met.get('final_v3', {}).get('auroc')} | {'melhorou' if melhorou else 'não melhorou'}",
                  flush=True)
            pasta = self.output_dir / name
            if melhorou:
                if estado["melhor"] is not None:
                    shutil.rmtree(estado["melhor"]["pasta"], ignore_errors=True)
                estado["melhor"] = {**linha, "pasta": str(pasta), "_p": p}
                estado["sem_melhora"] = 0
            else:
                shutil.rmtree(pasta, ignore_errors=True)
                estado["sem_melhora"] += 1
                if estado["sem_melhora"] >= a.paciencia:
                    raise Parada(ep)
            self.model.train()
            self.processor.change_mode(is_training=True)
            estado["t_epoca"] = time.perf_counter()

    config = TrainingConfig(
        output_dir=str(saida), experiment_name="gama-verif-glinerdecide", num_epochs=a.epocas,
        max_steps=a.passos or -1, batch_size=a.lote, eval_batch_size=a.lote_inferencia, encoder_lr=lr,
        task_lr=a.lr_cabecas, weight_decay=0.01, max_grad_norm=1.0, scheduler_type="linear", warmup_ratio=0.1,
        fp16=False, bf16=True, eval_strategy="epoch", save_total_limit=10, save_best=False, logging_steps=100,
        seed=a.semente, num_workers=a.workers, validate_data=True, report_to_wandb=False,
        allow_tf32=False, float32_matmul_precision="highest")
    trainer = Treinador(modelo, config)
    t0 = time.perf_counter()
    parou = None
    try:
        resumo = trainer.train(train_data=registros(tr, form))
    except Parada as e:
        parou, resumo = int(str(e)), {}
    if not estado["hist"]:                                # fumaça: max_steps antes do fim da época
        trainer._save_checkpoint("checkpoint-epoch-1")
    seg = time.perf_counter() - t0
    del trainer, modelo
    torch.cuda.empty_cache()
    m = estado["melhor"]
    return {"lr_encoder": lr, "lr_cabecas": a.lr_cabecas, "historico": estado["hist"], "melhor_epoca": m["epoca"],
            "melhor_passo": m["passo"], "validacao": m["validacao"], "parou_na_epoca": parou,
            "segundos": round(seg, 1), "resumo_trainer": {k: v for k, v in resumo.items() if not k.endswith("history")},
            "_pasta": m["pasta"], "_p": m["_p"]}


def cmd_treinar(a) -> int:
    import torch
    from huggingface_hub import HfApi, snapshot_download

    sem_tf32()
    t_job = time.perf_counter()
    dados = baixar_dados()
    tr, va = dados["treino.jsonl"], dados["validacao.jsonl"]
    if a.limite:
        tr, va = tr[: a.limite], va[: max(64, a.limite // 2)]
    print(f"treino {len(tr)} ({sum(r['rotulo'] for r in tr)} positivos), validação {len(va)}; formulação "
          f"{a.formulacao}: {FORMULACOES[a.formulacao]}; {_versoes()}", flush=True)
    base = pasta_base()
    resultados, melhor = [], None
    for lr in [float(x) for x in a.lrs.split(",")]:
        r = treinar_config(base, tr, va, a.formulacao, lr, a)
        if melhor is None or _chave(r["validacao"]) > _chave(melhor["validacao"]):
            if melhor is not None:
                shutil.rmtree(melhor["_pasta"], ignore_errors=True)
            melhor = r
        else:
            shutil.rmtree(r["_pasta"], ignore_errors=True)
        resultados.append({k: v for k, v in r.items() if not k.startswith("_")})
        print(f"== lr {lr:g}: melhor época {r['melhor_epoca']} AUROC {r['validacao']['todos']['auroc']} "
              f"log-loss {r['validacao']['todos']['log_loss']} ({r['segundos']}s)", flush=True)
    print(f"escolhida: lr {melhor['lr_encoder']:g} época {melhor['melhor_epoca']}", flush=True)

    # recarrega do disco e confere
    pasta = pathlib.Path(melhor["_pasta"])
    sem_tf32()
    recarregado = carregar(str(pasta))
    p2, _ = pontuar(recarregado, va, a.formulacao, a.lote_inferencia)
    m2 = metricas(p2, va)
    conf = {"auroc_treino": melhor["validacao"]["todos"]["auroc_cheia"], "auroc_recarregado": m2["todos"]["auroc_cheia"],
            "max_dif_score": max(abs(x - y) for x, y in zip(melhor["_p"], p2))}
    conf["ok"] = abs(conf["auroc_treino"] - conf["auroc_recarregado"]) < 1e-4
    print("recarga:", conf, flush=True)
    if not conf["ok"]:
        raise SystemExit("o modelo recarregado não reproduz a validação")
    info = {"candidato": "gliner_decide_ft", "base": f"{MODELO}@{REVISAO}", "dados": f"{REPO_DADOS}@{REV_DADOS}",
            "sha256_dados": SHA_DADOS, "treino": len(tr), "validacao": len(va), "formulacao": a.formulacao,
            "pergunta": FORMULACOES[a.formulacao], "nome_forma": NOME_FORMA, "tarefa": TAREFA, "rotulos": list(ROTULOS),
            "receita": {"lrs_encoder": a.lrs, "lr_cabecas": a.lr_cabecas, "epocas_max": a.epocas,
                        "paciencia": a.paciencia, "lote": a.lote, "agendamento": "linear, aquecimento 10%",
                        "decaimento": 0.01, "norma_gradiente": 1.0, "bf16": True, "tf32": False,
                        "semente": a.semente, "synthetic_label_prob": a.sintetico,
                        "demais": "padrões do TrainingConfig e do SamplingConfig da gliner2 2.0.0",
                        "limite": a.limite, "passos": a.passos},
            "criterio": "maior AUROC na validação interna inteira (4 casas); empate -> menor log-loss; depois a "
                        "ordem da grade; o mesmo critério escolhe a época (paciência 1)",
            "escolhida": {"lr_encoder": melhor["lr_encoder"], "epoca": melhor["melhor_epoca"],
                          "passo": melhor["melhor_passo"], "validacao": melhor["validacao"]},
            "configuracoes": resultados, "conferencia_recarga": conf, "versoes": _versoes(),
            "segundos_job_ate_aqui": round(time.perf_counter() - t_job, 1)}
    (pasta / "treino.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    (pasta / "README.md").write_text(_cartao(info), encoding="utf-8")
    print("TREINO", json.dumps({k: v for k, v in info.items() if k not in ("configuracoes", "nome_forma")},
                               ensure_ascii=False)[:2500], flush=True)
    if a.sem_upload:
        meta_c, regs, _ = baixar_candidatos()
        p, t = pontuar(recarregado, regs[:200], a.formulacao, a.lote_inferencia)
        print("fumaça pontuar:", t, [(r["cid"], round(s, 4)) for r, s in zip(regs[:5], p[:5])], flush=True)
        return 0
    del recarregado
    torch.cuda.empty_cache()
    api = HfApi()
    _repo_exp(api)
    c = api.upload_folder(folder_path=str(pasta), repo_id=REPO_EXP, repo_type="model",
                          commit_message=f"verificador gliner_decide_ft: lr {melhor['lr_encoder']:g}, "
                                         f"época {melhor['melhor_epoca']}, formulação {a.formulacao}")
    print("REVISAO_MODELO", c.oid, flush=True)
    if a.sem_pontuar:
        return 0
    a.modelo = f"{REPO_EXP}@{c.oid}"
    return cmd_pontuar(a)


def cmd_pontuar(a) -> int:
    from huggingface_hub import snapshot_download
    sem_tf32()
    semear(0)
    repo, rev = a.modelo.split("@")
    pasta = snapshot_download(repo, revision=rev)
    treino = json.loads(pathlib.Path(pasta, "treino.json").read_text(encoding="utf-8"))
    form = treino["formulacao"]
    modelo = carregar(pasta)
    extra = {"escolhida": {k: v for k, v in treino["escolhida"].items() if k != "validacao"},
             "receita": treino["receita"]}
    a.lote = getattr(a, "lote_inferencia", None) or a.lote
    return _pontuar_tudo(modelo, "ft", f"{repo}@{rev}", form, extra, a)


def _cartao(info: dict) -> str:
    e = info["escolhida"]
    return f"""---
license: apache-2.0
base_model: {MODELO}
tags: [experimento, privado, temporario]
---
# gama-exp-verif-glinerdecide (experimento privado e temporário)

Verificador de candidatos de citação: `{info['base']}` fine-tunado como classificador binário
(tarefa `{info['tarefa']}`, rótulos sim/não) pelo `ExtractorTrainer` da gliner2. Entrada: 300 caracteres
de cada lado com o candidato entre [[ e ]], e a pergunta `{info['pergunta']}`. Score = P(sim).

Não é modelo de produção e não faz parte de nenhuma submissão. Pode ser apagado.

- Dados: `{info['dados']}` (treino {info['treino']}, validação interna {info['validacao']})
- Escolhida na validação interna: lr do encoder {e['lr_encoder']:g}, época {e['epoca']};
  AUROC {e['validacao']['todos']['auroc']}, log-loss {e['validacao']['todos']['log_loss']}
- Critério: {info['criterio']}
- Histórico completo em `treino.json`.
"""


# ---------------------------------------------------------------- fumaça

def cmd_fumaca(a) -> int:
    """API e formato do classify_text (três exemplos de TREINO por formulação, sem métrica), tempo de
    inferência, e o caminho de treino inteiro em miniatura, sem subir nada."""
    import torch
    sem_tf32()
    semear(0)
    modelo = carregar(pasta_base())
    print("modelo:", type(modelo).__name__, getattr(modelo.config, "architecture", None),
          sum(p.numel() for p in modelo.parameters()), str(next(modelo.parameters()).dtype), _versoes(), flush=True)
    dados = baixar_dados()
    tr, va = dados["treino.jsonl"], dados["validacao.jsonl"]
    amostra = [next(r for r in tr if r["forma"] == f) for f in ("processo", "artigo", "vaga")]
    for f in FORMULACOES:
        for r in amostra:
            q = pergunta(f, r)
            res = modelo.classify_text(r["estado"], tarefa(q), include_confidence=True)
            print("FUMACA", f, r["forma"], "|", q, "|", json.dumps(res, ensure_ascii=False), "| p_sim", p_sim(res),
                  flush=True)
    _, t = pontuar(modelo, va[:512], "f2", a.lote_inferencia)
    print("tempo de inferência (512 da validação):", t, flush=True)
    tok = modelo.processor.tokenizer
    comp = sorted(len(tok(r["estado"], add_special_tokens=False)["input_ids"]) for r in tr[:2000])
    print("subtokens por estado:", {"mediana": comp[len(comp) // 2], "max": comp[-1]}, flush=True)
    del modelo
    torch.cuda.empty_cache()
    a.formulacao, a.limite, a.sem_upload = "f2", a.limite or 512, True
    t0 = time.perf_counter()
    cmd_treinar(a)
    print(f"treino da fumaça: {time.perf_counter() - t0:.0f}s para {a.limite} exemplos x {a.epocas} épocas x "
          f"{len(a.lrs.split(','))} lr", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="acao", required=True)

    def comum_treino(p):
        p.add_argument("--lrs", default="1e-5,3e-5")
        p.add_argument("--lr-cabecas", type=float, default=5e-4)
        p.add_argument("--epocas", type=int, default=3)
        p.add_argument("--paciencia", type=int, default=1)
        p.add_argument("--lote", type=int, default=32)
        p.add_argument("--lote-inferencia", type=int, default=64)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--sintetico", type=float, default=0.0, help="synthetic_label_prob do SamplingConfig")
        p.add_argument("--semente", type=int, default=SEMENTE)
        p.add_argument("--passos", type=int, default=0, help="max_steps (fumaça)")
        p.add_argument("--limite", type=int, default=0, help="só os N primeiros exemplos de treino (fumaça)")

    p = sub.add_parser("fumaca")
    comum_treino(p)
    p.set_defaults(lrs="3e-5", epocas=2, lote=32)
    p = sub.add_parser("zs-formulacoes")
    p.add_argument("--lote", type=int, default=64)
    p.add_argument("--limite", type=int, default=0)
    p.add_argument("--sem-upload", action="store_true")
    p = sub.add_parser("zs-pontuar")
    p.add_argument("--formulacao", required=True, choices=sorted(FORMULACOES))
    p.add_argument("--lote", type=int, default=64)
    p.add_argument("--limite", type=int, default=0)
    p.add_argument("--sem-upload", action="store_true")
    p = sub.add_parser("treinar")
    comum_treino(p)
    p.add_argument("--formulacao", required=True, choices=sorted(FORMULACOES))
    p.add_argument("--sem-upload", action="store_true")
    p.add_argument("--sem-pontuar", action="store_true")
    p = sub.add_parser("pontuar")
    p.add_argument("--modelo", required=True, help="<repo>@<revisão>")
    p.add_argument("--lote", type=int, default=64)
    p.add_argument("--limite", type=int, default=0)
    p.add_argument("--sem-upload", action="store_true")
    a = ap.parse_args()
    return {"fumaca": cmd_fumaca, "zs-formulacoes": cmd_zs_formulacoes, "zs-pontuar": cmd_zs_pontuar,
            "treinar": cmd_treinar, "pontuar": cmd_pontuar}[a.acao](a)


if __name__ == "__main__":
    raise SystemExit(main())
