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
"""Laya zero-shot (checkpoint multilíngue) julgando cada candidato do verificador, no HF Jobs (L4).

Entrada: `candidatos.jsonl` gravado por `python -m bench.controles.verificador candidatos` e subido
para o dataset privado em `bench/saida/controles/verificador/` (o sha256 é conferido antes de rodar).
Cada candidato já traz o `estado`: a janela de 300 caracteres de cada lado com o candidato entre
[[ e ]]. O job só roda o modelo: políticas e métricas rodam no container local.

Uma chamada `Router.predict(estado, perguntas, model="multilingual")` por estado distinto (um span
do Gama e um da régua com as mesmas bordas têm o mesmo estado e a mesma resposta). As perguntas
de `--formulacoes` vão juntas na chamada: o Laya monta uma sequência por pergunta (pergunta +
opções + estado) e as responde no mesmo passe, sem uma ver a outra.

    hf jobs uv run --flavor l4x1 --timeout 45m --secrets HF_TOKEN \\
        bench/controles/verificador/laya_job.py \\
        --revisao-dados <oid do candidatos.jsonl> --sha256 <sha256 do candidatos.jsonl> \\
        --conjuntos reais --formulacoes f1,f2,f3 --saida laya_formulacoes_reais.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import time

REPO = "vinimlo/gama-goldenset"
PASTA = "bench/saida/controles/verificador"
LAYA_REPO = "convaiinnovations/laya"
LAYA_REVISAO = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"   # = laya.PINNED_REVISIONS[LAYA_REPO]
MODELO = "multilingual"
SEMENTE = 0

# Instruções tiradas de wiki/conceitos/convencoes-de-borda.md (regras universais e por forma).
INSTRUCOES = (
    "O trecho entre [[ e ]] é uma citação completa de jurisprudência (processo numerado, súmula, "
    "tema ou julgado identificado por tribunal, ano e relator) ou de dispositivo de lei com o "
    "diploma, com as bordas certas? Bordas certas: começa na classe processual (REsp, AgInt, Rcl, "
    "APL), em Súmula, Tema, art. ou no substantivo da referência (julgado, precedente, acórdão); "
    "artigo ou preposição antes fica fora; pontuação depois fica fora; número de processo vai até "
    "a UF; artigo de lei inclui o nome do diploma; referência a julgado termina no nome do relator."
)
FORMULACOES = {
    # f1: booleana simples (noul), probabilidade de "sim"
    "f1": {"pergunta": {"type": "noul", "instructions": INSTRUCOES},
           "score": ["noul"]},
    # f2: booleana com as duas respostas descritas
    "f2": {"pergunta": {"type": "noul", "instructions": INSTRUCOES,
                        "criteria": {"true": "sim: o trecho marcado é exatamente uma citação completa, "
                                             "sem faltar nem sobrar texto",
                                     "false": "não: o trecho marcado não é citação, está incompleto, "
                                              "pega texto a mais ou junta duas citações"}},
           "score": ["noul"]},
    # f3: a mesma decisão como escolha de duas opções neutras (contorno do README para noul "preso")
    "f3": {"pergunta": {"type": "choice", "instructions": INSTRUCOES,
                        "criteria": {"A": "sim, o trecho marcado é exatamente uma citação completa "
                                          "com as bordas certas",
                                     "B": "não, o trecho marcado não é uma citação completa ou tem "
                                          "as bordas erradas"}},
           "score": ["probabilities", "A"]},
}


def _sha(caminho: pathlib.Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revisao-dados", required=True, help="revisão do dataset com o candidatos.jsonl")
    ap.add_argument("--sha256", required=True, help="sha256 esperado do candidatos.jsonl")
    ap.add_argument("--conjuntos", default="estresse,reais,novas")
    ap.add_argument("--formulacoes", default="f1,f2,f3")
    ap.add_argument("--saida", required=True, help="nome do arquivo em bench/saida/controles/verificador/")
    ap.add_argument("--limite", type=int, help="só os N primeiros estados (ensaio)")
    ap.add_argument("--sem-upload", action="store_true")
    a = ap.parse_args()

    os.environ.setdefault("USE_TF", "0")
    import numpy as np
    import torch
    import laya
    from huggingface_hub import HfApi, hf_hub_download
    from laya import Router

    torch.manual_seed(SEMENTE)
    np.random.seed(SEMENTE)
    arq = pathlib.Path(hf_hub_download(REPO, f"{PASTA}/candidatos.jsonl", repo_type="dataset",
                                       revision=a.revisao_dados))
    sha = _sha(arq)
    if sha != a.sha256:
        raise SystemExit(f"candidatos.jsonl tem sha256 {sha}, esperado {a.sha256}")
    conjuntos = set(a.conjuntos.split(","))
    forms = a.formulacoes.split(",")
    perguntas = {f: FORMULACOES[f]["pergunta"] for f in forms}

    cands, meta_cand = [], {}
    for linha in arq.read_text(encoding="utf-8").splitlines():
        r = json.loads(linha)
        if "meta" in r:
            meta_cand = r["meta"]
        elif r["conjunto"] in conjuntos:
            cands.append(r)
    estados = list(dict.fromkeys(r["estado"] for r in cands))[: a.limite]
    print(f"candidatos {len(cands)} estados distintos {len(estados)} formulações {forms}", flush=True)

    router = Router(device="cuda" if torch.cuda.is_available() else None, revision=LAYA_REVISAO, max_loaded=1)
    t0 = time.perf_counter()
    agente = router.load(MODELO)
    carga = time.perf_counter() - t0
    revisoes = router.loaded_revisions

    respostas, truncados = {}, 0
    t0 = time.perf_counter()
    for i, est in enumerate(estados):
        r = router.predict(est, perguntas, model=MODELO)
        if r.get("routing", {}).get("model") != MODELO:
            raise SystemExit(f"roteado para {r.get('routing')}, esperado {MODELO}")
        truncados += bool(r.get("usage", {}).get("truncated"))
        p = {}
        for f in forms:
            v = r["answers"][f]
            for k in FORMULACOES[f]["score"]:
                v = v[k]
            p[f] = float(v)
        respostas[est] = {"p": p, "r": {f: {k: v for k, v in r["answers"][f].items() if k != "action"} for f in forms}}
        if i % 500 == 0:
            print(f"{i}/{len(estados)} {time.perf_counter() - t0:.1f}s", flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total = time.perf_counter() - t0

    destino = pathlib.Path("/tmp") / a.saida
    with destino.open("w", encoding="utf-8") as fh:
        meta = {"verificador": "laya", "laya": laya.__version__, "repo": LAYA_REPO, "modelo": MODELO,
                "revisao_pedida": LAYA_REVISAO, "revisoes_carregadas": revisoes,
                "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                "dispositivo": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
                "dtype": str(getattr(agente, "dtype", None)), "semente": SEMENTE,
                "formulacoes": {f: FORMULACOES[f] for f in forms}, "conjuntos": sorted(conjuntos),
                "candidatos": {"arquivo": f"{PASTA}/candidatos.jsonl", "revisao": a.revisao_dados, "sha256": sha,
                               "meta": meta_cand},
                "estados": len(estados), "segundos_carga": round(carga, 2), "segundos_inferencia": round(total, 2),
                "ms_por_estado": round(1000 * total / max(1, len(estados)), 2),
                "revisao_agente": getattr(agente, "revision", None), "estados_truncados": truncados}
        fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        n = 0
        for r in cands:
            if r["estado"] not in respostas:
                continue
            x = respostas[r["estado"]]
            fh.write(json.dumps({"cid": r["cid"], "p": x["p"], "r": x["r"]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"gravados {n} candidatos em {total:.1f}s ({meta['ms_por_estado']} ms/estado); revisões {revisoes}",
          flush=True)
    if a.sem_upload:
        print(destino.read_text(encoding="utf-8")[:2000])
        return 0
    info = HfApi().upload_file(path_or_fileobj=str(destino), path_in_repo=f"{PASTA}/{a.saida}", repo_id=REPO,
                               repo_type="dataset", commit_message=f"verificador: {a.saida}")
    print("REVISAO", info.oid, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
