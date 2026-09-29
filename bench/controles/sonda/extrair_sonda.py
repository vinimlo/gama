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
"""Spans crus da sonda H2 e do controle de cabeça aleatória, em FP32 na L4, nos três conjuntos do
harness: estresse (600) + 305 ementas (`bench/entrada_remota.jsonl`) e as 200 ementas novas
(`bench/entrada_novas.jsonl`, o harness pontua as 172 com ouro). Grava
`bench/saida/controles/<experimento>/modelo_<nome>.jsonl` no dataset; guarda, resolver e métrica
rodam localmente em `bench.controles.avaliar`.

Cópia mínima de `bench/extrair_alunos.py`, pelo que ele não faz: (1) sobe para `bench/saida/`, fora
de `bench/saida/controles/`; (2) extrai também os 4.000 docs do `final_v1`, que os controles não
usam; (3) não monta o controle de cabeça aleatória. O laço de extração, a linha `meta`, o
aquecimento e a medição com `torch.cuda.synchronize()` são os dele.

`--aleatoria nome=repo@rev` monta o encoder original com uma cabeça de tokens sem treino:
`set_seed(semente)` e `AutoModelForTokenClassification.from_pretrained` com os rótulos de `bio.py`,
exatamente o ponto de partida da sonda (mesma semente, mesmo sha do classificador). O `head` vem do
checkpoint MLM; o `classifier` é aleatório. É ruído por construção: diagnóstico, não concorrente.

    hf jobs uv run --flavor l4x1 --timeout 30m --secrets HF_TOKEN bench/controles/sonda/extrair_sonda.py \\
        --modelos sonda_1e-3=vinimlo/gama-exp-sonda-1e-3@<sha> \\
        --aleatoria sonda_aleatoria=jhu-clsp/mmBERT-base@c5955035435e2bf121cde7f3c8863ef52ff35d82
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default="vinimlo/gama-goldenset")
    ap.add_argument("--revisao", default="2fff5f670e13f77f339937cfe3da52ed0af9572d")
    ap.add_argument("--experimento", default="sonda")
    ap.add_argument("--modelos", default="", help="nome=repo@revisao,...")
    ap.add_argument("--aleatoria", default="", help="nome=repo_base@revisao,...: base + cabeça sem treino")
    ap.add_argument("--semente", type=int, default=13)
    ap.add_argument("--limite", type=int, help="ensaio: só os N primeiros docs, sem upload")
    a = ap.parse_args()

    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModelForTokenClassification, AutoTokenizer, set_seed
    raiz = pathlib.Path(snapshot_download(a.dados, repo_type="dataset", revision=a.revisao, allow_patterns=[
        "bench/entrada_remota.jsonl", "bench/entrada_novas.jsonl", "bench/codigo/**"]))
    sys.path.insert(0, str(raiz / "bench" / "codigo"))
    from gama.extratores import bio
    from gama.extratores.neural import ExtratorNeural

    docs = []
    for arq in ("entrada_remota.jsonl", "entrada_novas.jsonl"):        # estresse + reais, novas
        docs += [json.loads(x) for x in (raiz / "bench" / arq).read_text(encoding="utf-8").splitlines()]
    docs = docs[: a.limite]

    alvos = []                                                           # (nome, pasta, rótulo do modelo)
    for item in filter(None, a.modelos.split(",")):
        nome, ref = item.split("=")
        repo, rev = ref.split("@")
        alvos.append((nome, lambda repo=repo, rev=rev: snapshot_download(repo, revision=rev), f"{repo}@{rev[:7]}"))
    for item in filter(None, a.aleatoria.split(",")):
        nome, ref = item.split("=")
        repo, rev = ref.split("@")

        def montar(nome=nome, repo=repo, rev=rev):
            set_seed(a.semente)
            m = AutoModelForTokenClassification.from_pretrained(
                repo, revision=rev, num_labels=len(bio.ROTULOS), id2label=dict(enumerate(bio.ROTULOS)), label2id=bio.ID)
            sha = hashlib.sha256(b"classifier.weight")          # o mesmo `_sha` de treinar_sonda.py
            sha.update(m.classifier.weight.detach().float().cpu().contiguous().numpy().tobytes())
            print(f"ALEATORIA {nome}: classifier.weight sha {sha.hexdigest()[:16]}", flush=True)
            pasta = pathlib.Path(f"/tmp/aleatoria_{nome}")
            m.save_pretrained(pasta)
            AutoTokenizer.from_pretrained(repo, revision=rev).save_pretrained(pasta)
            return str(pasta)
        alvos.append((nome, montar, f"{repo}@{rev[:7]}+cabeca-aleatoria-s{a.semente}"))

    api = HfApi()
    for nome, pasta, rotulo in alvos:
        torch.cuda.empty_cache()
        ext = ExtratorNeural(pasta())
        parametros = sum(p.numel() for p in ext.modelo.parameters())
        torch.cuda.reset_peak_memory_stats()
        ext.extrair("aquecimento: REsp nº 1.234.567/SP e art. 927 do Código Civil.")
        saida = pathlib.Path(f"/tmp/modelo_{nome}.jsonl")
        tempos: dict = {}
        with saida.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"meta": {"extrator": nome, "modelo": rotulo, "parametros": parametros,
                                          "torch": torch.__version__, "dispositivo": torch.cuda.get_device_name(0),
                                          "max_len": ext.max_len, "dados": f"{a.dados}@{a.revisao[:7]}"}}) + "\n")
            for d in docs:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                ss = ext.extrair(d["texto"])
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                tempos.setdefault(d["conjunto"], []).append(dt)
                fh.write(json.dumps({"conjunto": d["conjunto"], "id": d["id"], "segundos": round(dt, 4), "spans": [
                    [s.inicio, s.fim, s.tipo, s.forma, s.digitos, s.confianca] for s in ss]},
                    ensure_ascii=False) + "\n")
        vram = torch.cuda.max_memory_allocated() / 2**20
        print(f"FIM {nome} ({parametros / 1e6:.1f}M): " + " | ".join(
            f"{c} {len(v)} docs {sum(v) / len(v):.4f} s/doc" for c, v in tempos.items()) + f" | VRAM {vram:.0f} MiB", flush=True)
        del ext
        if not a.limite:
            destino = f"bench/saida/controles/{a.experimento}/modelo_{nome}.jsonl"
            assert destino.startswith("bench/saida/controles/") and ".." not in destino, destino
            info = api.upload_file(path_or_fileobj=str(saida), path_in_repo=destino, repo_id=a.dados,
                                   repo_type="dataset", commit_message=f"controles/{a.experimento}: spans de {nome} ({rotulo})")
            print("REVISAO", info.oid, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
