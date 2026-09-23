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
"""Gama e régua sobre ementas REAIS (fora da distribuição), em GPU.

Lê `reais/amostra.jsonl` do dataset, extrai com os dois extratores e publica
`reais/predicoes.jsonl` de volta no dataset (uma linha por ementa, spans dos dois).
Só extração: a resolução no acervo não entra aqui (o acervo não sai da máquina).
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
    a = ap.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    raiz = pathlib.Path(snapshot_download(a.dados, repo_type="dataset", revision=a.revisao))
    sys.path.insert(0, str(raiz / "codigo"))
    from gama.extratores.neural import ExtratorNeural
    from gama.extratores.regua import ExtratorRegua

    gama = ExtratorNeural(snapshot_download(a.pesos, revision=a.pesos_rev))
    regua = ExtratorRegua()
    saida = pathlib.Path("/tmp/predicoes.jsonl")
    t0 = time.perf_counter()
    n = 0
    with saida.open("w", encoding="utf-8") as fh:
        for linha in (raiz / "reais" / "amostra.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(linha)
            t = r["texto"]
            fh.write(json.dumps({
                "id": r["id"],
                "gama": [(s.inicio, s.fim, s.forma, round(s.confianca or 0, 4)) for s in gama.extrair(t)],
                "regua": [(s.inicio, s.fim, s.forma) for s in regua.extrair(t)],
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} ementas em {time.perf_counter() - t0:.0f}s", flush=True)
    info = HfApi().upload_file(path_or_fileobj=str(saida), path_in_repo="reais/predicoes.jsonl",
                               repo_id=a.dados, repo_type="dataset",
                               commit_message=f"predicoes Gama@{a.pesos_rev[:7]} e regua em ementas reais")
    print("REVISAO", info.oid, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
