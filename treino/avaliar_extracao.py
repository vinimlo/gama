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
"""Compara extratores pela EXTRAÇÃO (span e tipo) num conjunto rotulado, em GPU.

A resolução no acervo é o mesmo código determinístico para todos os candidatos, e o
acervo é dado da organização (não sai da máquina). Então, para escolher entre
modelos, basta medir a extração: F1 por tipo com borda exata e com IoU >= 0,5.

    hf jobs uv run --flavor a10g-small --secrets HF_TOKEN treino/avaliar_extracao.py \\
        --dados vinimlo/gama-goldenset --revisao <sha> \\
        --modelos vinimlo/gama,vinimlo/gama-d0-llm,vinimlo/gama-bertimbau-d0-llm
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys

ROT = {"incompleta": "VAGA"}


def f1(gold: dict, pred: dict) -> dict:
    out = {}
    for crit in ("exato", "iou50"):
        tp, fp, fn = collections.Counter(), collections.Counter(), collections.Counter()
        for doc, gs in gold.items():
            ps = list(pred.get(doc, []))
            usados = set()
            for a, b, t in gs:
                achou = None
                for k, (pa, pb, pt) in enumerate(ps):
                    if k in usados or pt != t:
                        continue
                    inter = max(0, min(b, pb) - max(a, pa))
                    ok = (pa, pb) == (a, b) if crit == "exato" else inter / (max(b, pb) - min(a, pa)) >= 0.5
                    if ok:
                        achou = k
                        break
                if achou is None:
                    fn[t] += 1
                else:
                    usados.add(achou)
                    tp[t] += 1
            for k, (_, _, pt) in enumerate(ps):
                if k not in usados:
                    fp[pt] += 1
        out[crit] = {t: round(2 * tp[t] / max(1, 2 * tp[t] + fp[t] + fn[t]), 4)
                     for t in ("JURIS", "LEI", "VAGA")}
        out[crit]["erros"] = {t: {"fp": fp[t], "fn": fn[t]} for t in ("JURIS", "LEI", "VAGA")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", required=True)
    ap.add_argument("--revisao", required=True)
    ap.add_argument("--subpasta", default="estresse_glm")
    ap.add_argument("--modelos", required=True, help="repos separados por vírgula (repo[@rev])")
    ap.add_argument("--uniao", action="store_true", help="mede também neural + régua (ExtratorUniao)")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    raiz = pathlib.Path(snapshot_download(a.dados, repo_type="dataset", revision=a.revisao))
    sys.path.insert(0, str(raiz / "codigo"))
    from gama.extratores.neural import ExtratorNeural
    from gama.extratores.regua import ExtratorRegua
    from gama.pipeline import aparar

    pasta = raiz / a.subpasta
    gold = collections.defaultdict(list)
    for r in csv.DictReader(open(pasta / "goldenset_offsets.csv", encoding="utf-8-sig")):
        tipo = ROT.get(r["classificacao"]) or ("LEI" if r["tipo"] == "lei" else "JURIS")
        gold[r["documento_id"]].append((int(r["inicio"]), int(r["fim"]), tipo))
    textos = {}
    for d in gold:
        with open(pasta / "txt" / f"{d}.txt", encoding="utf-8", newline="") as fh:
            textos[d] = fh.read()
    tabela = {}
    for spec in a.modelos.split(","):
        repo, _, rev = spec.partition("@")
        ext = ExtratorNeural(snapshot_download(repo, revision=rev or None))
        regua = ExtratorRegua()

        def rot(s):
            return "LEI" if s.tipo == "lei" else ("VAGA" if s.forma == "vaga" else "JURIS")

        neural, uniao = {}, {}
        for d, t in textos.items():
            ns = [x for x in (aparar(s, t) for s in ext.extrair(t)) if x]
            neural[d] = [(s.inicio, s.fim, rot(s)) for s in ns]
            if a.uniao:
                extra = [x for x in (aparar(r, t) for r in regua.extrair(t)) if x
                         and not any(x.inicio < s.fim and s.inicio < x.fim for s in ns)]
                uniao[d] = neural[d] + [(s.inicio, s.fim, rot(s)) for s in extra]
        for nome, pred in (("neural", neural), ("uniao", uniao)):
            if not pred:
                continue
            niveis = {}
            for nv in ("_n1_", "_n2_"):
                g = {d: v for d, v in gold.items() if nv in d}
                niveis[nv.strip("_")] = f1(g, {d: pred[d] for d in g})
            tabela[f"{spec}:{nome}"] = niveis
            print("EXTRACAO", f"{spec}:{nome}", json.dumps(niveis), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
