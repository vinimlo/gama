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
"""Ensaio do Gama no hardware-alvo (1 GPU 24 GB), com o torch da imagem (cu126).

Mede o que a organização vai medir e o que pode quebrar a reprodução:
  - tempo por documento na GPU (teto: média <= 60 s);
  - spans GPU x CPU no mesmo documento (reprodutibilidade: queda <= 5%);
  - duas execuções na GPU (determinismo).
Roda o ExtratorNeural de verdade (pacote gama do dataset), sem rede depois do download,
sobre os documentos sintéticos de estresse — os dados da organização não saem da máquina.

    hf jobs uv run --flavor l4x1 --secrets HF_TOKEN treino/ensaio.py \\
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
    ap.add_argument("--subpasta", default="estresse_glm")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-cpu", type=int, default=40)
    a = ap.parse_args()

    from huggingface_hub import snapshot_download
    import torch
    raiz = pathlib.Path(snapshot_download(a.dados, repo_type="dataset", revision=a.revisao))
    pesos = snapshot_download(a.pesos, revision=a.pesos_rev)
    sys.path.insert(0, str(raiz / "codigo"))
    from gama.extratores.neural import ExtratorNeural

    info = {"cuda": torch.cuda.is_available(), "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    print(json.dumps(info), flush=True)
    docs = sorted((raiz / a.subpasta / "txt").glob("*.txt"))[: a.n]
    textos = []
    for d in docs:
        with open(d, encoding="utf-8", newline="") as fh:
            textos.append(fh.read())

    gama = ExtratorNeural(pesos)
    t0 = time.perf_counter()
    gpu1 = [[(s.inicio, s.fim, s.tipo) for s in gama.extrair(t)] for t in textos]
    t_gpu = (time.perf_counter() - t0) / len(textos)
    gpu2 = [[(s.inicio, s.fim, s.tipo) for s in gama.extrair(t)] for t in textos]

    gama.modelo.to("cpu")
    gama.disp = "cpu"
    t0 = time.perf_counter()
    cpu = [[(s.inicio, s.fim, s.tipo) for s in gama.extrair(t)] for t in textos[: a.n_cpu]]
    t_cpu = (time.perf_counter() - t0) / max(1, len(cpu))

    iguais_gpu = sum(x == y for x, y in zip(gpu1, gpu2))
    iguais_cpu = sum(x == y for x, y in zip(gpu1[: a.n_cpu], cpu))
    spans = sum(len(x) for x in gpu1)
    res = {**info, "docs": len(textos), "spans": spans,
           "s_por_doc_gpu": round(t_gpu, 4), "s_por_doc_cpu": round(t_cpu, 3),
           "gpu_2_execucoes_identicas": f"{iguais_gpu}/{len(textos)}",
           "gpu_x_cpu_identicos": f"{iguais_cpu}/{len(cpu)}"}
    print("ENSAIO", json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
