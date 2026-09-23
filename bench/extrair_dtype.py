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
"""Gama v1.2 em FP32, BF16 e FP16 na L4: a saída muda? quanto tempo e VRAM economiza?

Três conjuntos: estresse + reais (as entradas do benchmark) e o `final_v1` do dataset
(4.000 documentos sintéticos de outra semente, que o v1.2 não treinou), que também
serve para medir a cauda de confiança dos acertos no estilo da organização. Grava os
spans por precisão em `bench/saida/dtype_<precisao>.jsonl`; a comparação é local.

    hf jobs uv run --flavor l4x1 --secrets HF_TOKEN bench/extrair_dtype.py \\
        --dados vinimlo/gama-goldenset --revisao <sha> --pesos vinimlo/gama --pesos-rev <sha>
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
    ap.add_argument("--pesos", required=True)
    ap.add_argument("--pesos-rev", required=True)
    ap.add_argument("--precisoes", default="fp32,bf16,fp16")
    ap.add_argument("--limite", type=int)
    a = ap.parse_args()

    import torch
    from huggingface_hub import HfApi, snapshot_download
    raiz = pathlib.Path(snapshot_download(a.dados, repo_type="dataset", revision=a.revisao,
                                          allow_patterns=["bench/*", "bench/codigo/**", "final_v1/txt/*"]))
    sys.path.insert(0, str(raiz / "bench" / "codigo"))
    from gama.extratores.neural import ExtratorNeural

    docs = [json.loads(x) for x in (raiz / "bench" / "entrada_remota.jsonl").read_text(encoding="utf-8").splitlines()]
    for arq in sorted((raiz / "final_v1" / "txt").glob("*.txt")):
        with open(arq, encoding="utf-8", newline="") as fh:
            docs.append({"conjunto": "v1", "id": arq.stem, "texto": fh.read()})
    docs = docs[: a.limite]
    ext = ExtratorNeural(snapshot_download(a.pesos, revision=a.pesos_rev))
    fp32 = {k: v.clone() for k, v in ext.modelo.state_dict().items()}
    api = HfApi()
    tipos = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    for nome in a.precisoes.split(","):
        ext.modelo.load_state_dict(fp32)
        ext.modelo.to(tipos[nome])
        torch.cuda.reset_peak_memory_stats()
        ext.extrair("aquecimento: REsp nº 1.234.567/SP e art. 927 do Código Civil.")
        meta = {"extrator": "gama", "precisao": nome, "modelo": f"{a.pesos}@{a.pesos_rev[:7]}",
                "dados": f"{a.dados}@{a.revisao[:7]}", "torch": torch.__version__,
                "dispositivo": torch.cuda.get_device_name(0)}
        saida = pathlib.Path(f"/tmp/dtype_{nome}.jsonl")
        tempos: dict = {}
        with saida.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"meta": meta}) + "\n")
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
        print(f"FIM {nome}: " + " | ".join(f"{c} {len(v)} docs {sum(v) / len(v):.4f} s/doc"
                                          for c, v in tempos.items()) + f" | VRAM pico {vram:.0f} MiB", flush=True)
        if not a.limite:
            info = api.upload_file(path_or_fileobj=str(saida), path_in_repo=f"bench/saida/dtype_{nome}.jsonl",
                                   repo_id=a.dados, repo_type="dataset",
                                   commit_message=f"bench: spans do Gama em {nome} (estresse, reais, final_v1)")
            print("REVISAO", info.oid, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
