# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gliner==0.2.29",
#   "transformers>=4.51,<5",
#   "sentencepiece",
#   "protobuf",
#   "torch",
#   "huggingface_hub",
# ]
# ///
# -*- coding: utf-8 -*-
"""Extrator cru 1: GLiNER multilíngue em zero-shot, sem nenhum treino nosso.

Recebe só os nomes dos rótulos em português. O GLiNER lê até 384 tokens, então o
documento é percorrido em janelas de palavras sobrepostas; na sobreposição fica o
span de maior score. Saída: uma linha JSON por documento com `spans` [(inicio, fim,
rótulo)] em codepoints, no mesmo formato de todos os extratores do benchmark.

    # estresse + reais, em GPU (os textos já estão no dataset privado)
    hf jobs uv run --flavor l4x1 --secrets HF_TOKEN bench/extrair_gliner.py \\
        --dados vinimlo/gama-goldenset --revisao <sha>
    # dev, em CPU e local (os documentos da organização não saem da máquina)
    uv run bench/extrair_gliner.py --entrada saidas/bench/entrada_dev.jsonl \\
        --saida saidas/bench/gliner_dev.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import time

MODELO = "urchade/gliner_multi-v2.1"
REVISAO = "443d26d654e0324125a96bebd8e796c14ff2efe6"
# Rótulo em linguagem natural -> tipo do benchmark. Escolhidos uma vez, sem ajuste nos
# conjuntos de avaliação.
ROTULOS = {
    "precedente judicial com número": "JURIS",
    "súmula ou tema de tribunal": "JURIS",
    "artigo de lei": "LEI",
    "referência a julgado sem número": "VAGA",
}
LIMIAR = 0.5
PALAVRAS, PASSO = 200, 150      # 200 palavras ficam abaixo dos 384 tokens do GLiNER


def janelas(texto: str) -> list[tuple[int, int]]:
    palavras = [m.span() for m in re.finditer(r"\S+", texto)]
    out, i = [], 0
    while palavras:
        bloco = palavras[i:i + PALAVRAS]
        out.append((bloco[0][0], bloco[-1][1]))
        if i + PALAVRAS >= len(palavras):
            break
        i += PASSO
    return out


def extrair(modelo, texto: str) -> list[list]:
    achados = []
    for a, b in janelas(texto):
        for e in modelo.predict_entities(texto[a:b], list(ROTULOS), threshold=LIMIAR):
            achados.append((a + e["start"], a + e["end"], ROTULOS[e["label"]], float(e["score"])))
    # Sobreposição entre janelas: fica o de maior score.
    ficam: list = []
    for s in sorted(achados, key=lambda x: -x[3]):
        if all(s[1] <= f[0] or f[1] <= s[0] for f in ficam):
            ficam.append(s)
    return [[a, b, r, round(sc, 4)] for a, b, r, sc in sorted(ficam)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", help="JSONL local {conjunto, id, texto}")
    ap.add_argument("--saida", help="JSONL local de saída")
    ap.add_argument("--dados", help="dataset do Hub com bench/entrada_remota.jsonl")
    ap.add_argument("--revisao")
    ap.add_argument("--limite", type=int)
    a = ap.parse_args()

    import gliner
    import torch
    from gliner import GLiNER
    from huggingface_hub import HfApi, hf_hub_download

    entrada = pathlib.Path(a.entrada) if a.entrada else pathlib.Path(hf_hub_download(
        a.dados, "bench/entrada_remota.jsonl", repo_type="dataset", revision=a.revisao))
    saida = pathlib.Path(a.saida or "/tmp/gliner.jsonl")
    docs = [json.loads(x) for x in entrada.read_text(encoding="utf-8").splitlines()][: a.limite]

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = GLiNER.from_pretrained(MODELO, revision=REVISAO).to(dispositivo).eval()
    meta = {"extrator": "gliner", "modelo": f"{MODELO}@{REVISAO[:7]}", "gliner": gliner.__version__,
            "torch": torch.__version__, "dispositivo": torch.cuda.get_device_name(0)
            if dispositivo == "cuda" else "cpu"}
    print(json.dumps(meta), flush=True)

    t_total = 0.0
    with saida.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for k, d in enumerate(docs):
            t0 = time.perf_counter()
            with torch.inference_mode():
                spans = extrair(modelo, d["texto"])
            dt = time.perf_counter() - t0
            t_total += dt
            fh.write(json.dumps({"conjunto": d["conjunto"], "id": d["id"], "spans": spans,
                                 "segundos": round(dt, 4)}, ensure_ascii=False) + "\n")
            if k % 100 == 0:
                print(f"{k}/{len(docs)} {t_total / (k + 1):.3f} s/doc", flush=True)
    print(f"FIM {len(docs)} documentos, {t_total / max(1, len(docs)):.3f} s/doc", flush=True)
    if a.dados and not a.saida:
        info = HfApi().upload_file(path_or_fileobj=str(saida), path_in_repo="bench/saida/gliner.jsonl",
                                   repo_id=a.dados, repo_type="dataset",
                                   commit_message="bench: spans do GLiNER zero-shot (estresse + reais)")
        print("REVISAO", info.oid, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
