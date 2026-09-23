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
"""Gama v1.2 e régua sobre estresse + reais, na L4, com o torch da imagem avaliada (cu126).

Usa o pacote `gama` publicado junto com as entradas do benchmark (`bench/codigo/`) e os
pesos na revisão do MODELO.md. Grava o Span completo (tipo, forma, dígitos, confiança),
para a pontuação local seguir exatamente o caminho de produção.

    hf jobs uv run --flavor l4x1 --secrets HF_TOKEN bench/extrair_gama.py \\
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
    ap.add_argument("--limite", type=int)
    a = ap.parse_args()

    import torch
    from huggingface_hub import HfApi, snapshot_download
    raiz = pathlib.Path(snapshot_download(a.dados, repo_type="dataset", revision=a.revisao,
                                          allow_patterns=["bench/*", "bench/codigo/**"]))
    sys.path.insert(0, str(raiz / "bench" / "codigo"))
    from gama.extratores.neural import ExtratorNeural
    from gama.extratores.regua import ExtratorRegua

    docs = [json.loads(x) for x in (raiz / "bench" / "entrada_remota.jsonl")
            .read_text(encoding="utf-8").splitlines()][: a.limite]
    extratores = {"gama": ExtratorNeural(snapshot_download(a.pesos, revision=a.pesos_rev)),
                  "regua": ExtratorRegua()}
    dispositivo = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    api = HfApi()
    for nome, ext in extratores.items():
        meta = {"extrator": nome, "modelo": f"{a.pesos}@{a.pesos_rev[:7]}" if nome == "gama" else "regras",
                "torch": torch.__version__, "dispositivo": dispositivo}
        saida = pathlib.Path(f"/tmp/{nome}.jsonl")
        t_total = 0.0
        with saida.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"meta": meta}) + "\n")
            for d in docs:
                t0 = time.perf_counter()
                ss = ext.extrair(d["texto"])
                dt = time.perf_counter() - t0
                t_total += dt
                fh.write(json.dumps({"conjunto": d["conjunto"], "id": d["id"], "segundos": round(dt, 4), "spans": [
                    [s.inicio, s.fim, s.tipo, s.forma, s.digitos, s.confianca] for s in ss]},
                    ensure_ascii=False) + "\n")
        print(f"FIM {nome}: {len(docs)} documentos, {t_total / max(1, len(docs)):.4f} s/doc", flush=True)
        if not a.limite:
            info = api.upload_file(path_or_fileobj=str(saida), path_in_repo=f"bench/saida/{nome}.jsonl",
                                   repo_id=a.dados, repo_type="dataset",
                                   commit_message=f"bench: spans de {nome} (estresse + reais)")
            print("REVISAO", info.oid, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
