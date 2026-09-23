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
"""Professor e alunos destilados, em FP32 e na mesma L4, sobre os conjuntos da régua de zero
perda: estresse (600), final_v1 (4.000), as 305 ementas do ouro antigo e as 200 do teste
novo. Grava os spans de cada modelo em `bench/saida/modelo_<nome>.jsonl`; a comparação
(spans, guarda, JSON final, F1 no texto real) é local, em `bench.alunos`.

    hf jobs uv run --flavor l4x1 --secrets HF_TOKEN bench/extrair_alunos.py \\
        --dados vinimlo/gama-goldenset --revisao <sha> \\
        --modelos professor=vinimlo/gama@<sha>,a=vinimlo/gama-aluno-a@<sha>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", required=True)
    ap.add_argument("--revisao", required=True)
    ap.add_argument("--modelos", required=True, help="nome=repo@revisao,...")
    ap.add_argument("--limite", type=int)
    a = ap.parse_args()

    import torch
    from huggingface_hub import HfApi, snapshot_download
    raiz = pathlib.Path(snapshot_download(a.dados, repo_type="dataset", revision=a.revisao, allow_patterns=[
        "bench/*.jsonl", "bench/codigo/**", "final_v1/txt/*"]))
    sys.path.insert(0, str(raiz / "bench" / "codigo"))
    from gama.extratores.neural import ExtratorNeural

    docs = []
    for arq in ("entrada_remota.jsonl", "entrada_novas.jsonl"):        # estresse + reais, novas
        docs += [json.loads(x) for x in (raiz / "bench" / arq).read_text(encoding="utf-8").splitlines()]
    for arq in sorted((raiz / "final_v1" / "txt").glob("*.txt")):
        with open(arq, encoding="utf-8", newline="") as fh:
            docs.append({"conjunto": "v1", "id": arq.stem, "texto": fh.read()})
    docs = docs[: a.limite]
    api = HfApi()
    for item in a.modelos.split(","):
        nome, ref = item.split("=")
        repo, rev = ref.split("@")
        torch.cuda.empty_cache()
        ext = ExtratorNeural(snapshot_download(repo, revision=rev))
        parametros = sum(p.numel() for p in ext.modelo.parameters())
        torch.cuda.reset_peak_memory_stats()
        ext.extrair("aquecimento: REsp nº 1.234.567/SP e art. 927 do Código Civil.")
        saida = pathlib.Path(f"/tmp/modelo_{nome}.jsonl")
        tempos: dict = {}
        with saida.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"meta": {"extrator": nome, "modelo": f"{repo}@{rev[:7]}", "parametros": parametros,
                                          "torch": torch.__version__, "dispositivo": torch.cuda.get_device_name(0)}}) + "\n")
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
            info = api.upload_file(path_or_fileobj=str(saida), path_in_repo=f"bench/saida/modelo_{nome}.jsonl",
                                   repo_id=a.dados, repo_type="dataset", commit_message=f"bench: spans de {nome} ({repo}@{rev[:7]})")
            print("REVISAO", info.oid, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
