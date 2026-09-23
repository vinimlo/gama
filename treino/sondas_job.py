# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.0", "transformers==5.17.0", "safetensors", "huggingface_hub", "numpy"]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
# -*- coding: utf-8 -*-
"""Sondas de ruído documentado (tests/test_sondas_ruido.py) contra pesos no Hub, em CPU.

Só extração: a sonda passa se algum span tem IoU >= 0,5 com a citação. Evita carregar
um terceiro modelo na máquina local quando ela está sob pressão de memória.
"""
import argparse
import json
import pathlib
import sys

SONDAS = [
    ("m->rn Sumula", "Súrnula 211 do STJ"),
    ("m->rn Tema", "Terna 725 da repercussão geral"),
    ("m->rn Reclamacao", "Reclarnação nº 66.516/RO"),
    ("l->1 Especial", "AgInt no Recurso Especia1 nº 1.620.021/PR"),
    ("l->1 relatoria", "precedente do STF de 2024, da re1atoria de Cármen Lúcia"),
    ("l->1 Rel.", "Rcl de 2021, Re1. Min. Rosa Weber"),
]
MOLDURA = ("PODER JUDICIÁRIO\nTRIBUNAL REGIONAL\n\nProcesso nº 8133385-26.2020.5.05.4913\n"
           "Relator: Desembargador PAULO HENRIQUE\n\n\nACÓRDÃO\n\n"
           "Trata-se de recurso. Contrarrazões apresentadas.\n\n"
           "No mérito, invoca-se o {cit}, no ponto.\n\nConclusão.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", required=True)
    ap.add_argument("--revisao", required=True)
    ap.add_argument("--modelos", required=True)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    raiz = pathlib.Path(snapshot_download(a.dados, repo_type="dataset", revision=a.revisao,
                                          allow_patterns=["codigo/**"]))
    sys.path.insert(0, str(raiz / "codigo"))
    from gama.extratores.neural import ExtratorNeural
    for spec in a.modelos.split(","):
        repo, _, rev = spec.partition("@")
        g = ExtratorNeural(snapshot_download(repo, revision=rev or None))
        for nome, cit in SONDAS:
            t = MOLDURA.format(cit=cit)
            a0 = t.index(cit); b0 = a0 + len(cit)
            spans = g.extrair(t)
            ok = any(max(0, min(b0, s.fim) - max(a0, s.inicio)) / (max(b0, s.fim) - min(a0, s.inicio)) >= 0.5 for s in spans)
            print("SONDA", json.dumps({"modelo": spec, "sonda": nome, "passa": ok,
                                       "spans": [s.trecho for s in spans]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
