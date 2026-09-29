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
"""Spans crus de modelos BIO na L4, nos conjuntos do protocolo comum (estresse, 305, 172).

Cópia de `bench/extrair_alunos.py` com três mudanças, todas de destino ou de escopo, nenhuma
na extração (mesmo `ExtratorNeural`, mesmo aquecimento, mesma medida de tempo, mesmo formato):
  1. sobe para `bench/saida/controles/<experimento>/modelo_<nome>.jsonl` (o original grava em
     `bench/saida/`, fora da área permitida aos controles);
  2. baixa só as duas entradas e `bench/codigo/**` (o original lê também `final_v1`, 4.000
     documentos que o protocolo não pontua);
  3. a linha `meta` leva também a revisão inteira, o `max_len` efetivo do extrator, o pico de
     VRAM e o sha256 dos módulos do extrator no snapshot, para a proveniência.

    hf jobs uv run --flavor l4x1 --timeout 40m --secrets HF_TOKEN bench/controles/bertimbau/extrair.py \\
        --revisao 2fff5f670e13f77f339937cfe3da52ed0af9572d --experimento bertimbau \\
        --modelos bertimbau=vinimlo/gama-exp-bertimbau@<sha>,v12=vinimlo/gama@ad06ffd...,v13=vinimlo/gama@5f924ca...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

DADOS = "vinimlo/gama-goldenset"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revisao", required=True)
    ap.add_argument("--experimento", required=True)
    ap.add_argument("--modelos", required=True, help="nome=repo@revisao,...")
    ap.add_argument("--limite", type=int, help="teste: só os N primeiros documentos, sem subir nada")
    a = ap.parse_args()

    import torch
    from huggingface_hub import HfApi, snapshot_download
    raiz = pathlib.Path(snapshot_download(DADOS, repo_type="dataset", revision=a.revisao, allow_patterns=[
        "bench/entrada_remota.jsonl", "bench/entrada_novas.jsonl", "bench/codigo/**"]))
    sys.path.insert(0, str(raiz / "bench" / "codigo"))
    from gama.extratores.neural import ExtratorNeural
    codigo = {p: hashlib.sha256((raiz / "bench" / "codigo" / "gama" / p).read_bytes()).hexdigest()[:12]
              for p in ("extratores/neural.py", "extratores/bio.py", "formas.py", "span.py")}
    print("codigo", codigo, flush=True)

    docs = []
    for arq in ("entrada_remota.jsonl", "entrada_novas.jsonl"):        # estresse + reais, novas
        docs += [json.loads(x) for x in (raiz / "bench" / arq).read_text(encoding="utf-8").splitlines()]
    docs = docs[: a.limite]
    api = HfApi()
    for item in a.modelos.split(","):
        nome, ref = item.split("=")
        repo, rev = ref.split("@")
        torch.cuda.empty_cache()
        t_carga = time.perf_counter()
        ext = ExtratorNeural(snapshot_download(repo, revision=rev))
        t_carga = time.perf_counter() - t_carga
        parametros = sum(p.numel() for p in ext.modelo.parameters())
        torch.cuda.reset_peak_memory_stats()
        ext.extrair("aquecimento: REsp nº 1.234.567/SP e art. 927 do Código Civil.")
        saida = pathlib.Path(f"/tmp/modelo_{nome}.jsonl")
        tempos: dict = {}
        linhas = []
        for d in docs:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ss = ext.extrair(d["texto"])
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            tempos.setdefault(d["conjunto"], []).append(dt)
            linhas.append(json.dumps({"conjunto": d["conjunto"], "id": d["id"], "segundos": round(dt, 4), "spans": [
                [s.inicio, s.fim, s.tipo, s.forma, s.digitos, s.confianca] for s in ss]}, ensure_ascii=False))
        vram = torch.cuda.max_memory_allocated() / 2**20
        meta = {"extrator": nome, "modelo": f"{repo}@{rev[:7]}", "parametros": parametros,
                "torch": torch.__version__, "dispositivo": torch.cuda.get_device_name(0),
                "revisao": rev, "max_len": ext.max_len, "vram_pico_mib": round(vram), "carga_s": round(t_carga, 1),
                "dados": f"{DADOS}@{a.revisao}", "codigo": codigo,
                "s_por_doc": {c: round(sum(v) / len(v), 4) for c, v in tempos.items()}}
        saida.write_text("\n".join([json.dumps({"meta": meta})] + linhas) + "\n", encoding="utf-8")
        print(f"FIM {nome} ({parametros / 1e6:.1f}M, max_len {ext.max_len}): " + " | ".join(
            f"{c} {len(v)} docs {sum(v) / len(v):.4f} s/doc" for c, v in tempos.items()) + f" | VRAM {vram:.0f} MiB",
            flush=True)
        del ext
        if not a.limite:
            info = api.upload_file(path_or_fileobj=str(saida), repo_id=DADOS, repo_type="dataset",
                                   path_in_repo=f"bench/saida/controles/{a.experimento}/modelo_{nome}.jsonl",
                                   commit_message=f"controles/{a.experimento}: spans de {nome} ({repo}@{rev[:7]})")
            print("REVISAO", info.oid, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
