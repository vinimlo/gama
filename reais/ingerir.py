# -*- coding: utf-8 -*-
"""Ingestão de ementas reais com proveniência por registro.

Fonte: `celsowm/jurisprudencias_br` (781,5 mil decisões, CC-BY-4.0, coletadas pelo
Juriscraper). Amostra estratificada por tribunal, gravada em JSONL com a proveniência de
cada registro (fonte, licença, revisão do dataset, tribunal, classe).

Uso: ROBUSTEZ fora da distribuição (onde o Gama e a régua discordam em texto real),
nunca a base de resolução — processo real vindo daqui que coincidisse com um número
inventado da organização viraria `real` e acionaria τ.

    docker compose run --rm lab python -m reais.ingerir --n 3000
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random

FONTE = "celsowm/jurisprudencias_br"
LICENCA = "CC-BY-4.0"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--saida", default="/app/corpus/reais/amostra.jsonl")
    ap.add_argument("--semente", type=int, default=5)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.dataset_info(FONTE)
    shards = sorted(f.rfilename for f in info.siblings
                    if f.rfilename.endswith(".parquet") and not f.rfilename.startswith("full/"))
    rng = random.Random(a.semente)
    # Os shards vêm ordenados por tribunal: ler todos, com reservatório por tribunal
    # (memória limitada a `reserva` registros por tribunal, não 781 mil).
    reserva = max(200, a.n)
    por_trib = collections.defaultdict(list)
    vistos = collections.Counter()
    for s in shards:
        arq = hf_hub_download(FONTE, s, repo_type="dataset", revision=info.sha)
        t = pq.read_table(arq, columns=["tribunal", "classe", "numero_processo", "relator",
                                        "data_julgamento", "ementa", "documento_id"])
        for r in t.to_pylist():
            if not (r["ementa"] and len(r["ementa"]) > 200):
                continue
            k = r["tribunal"]
            vistos[k] += 1
            if len(por_trib[k]) < reserva:
                por_trib[k].append(r)
            else:
                j = rng.randrange(vistos[k])
                if j < reserva:
                    por_trib[k][j] = r
    trib = sorted(por_trib)
    cota = max(1, a.n // len(trib))
    amostra = []
    for t in trib:
        amostra += rng.sample(por_trib[t], min(cota, len(por_trib[t])))
    out = pathlib.Path(a.saida)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(amostra):
            fh.write(json.dumps({"id": f"real_{i:05d}", "texto": r["ementa"],
                                 "proveniencia": {"fonte": FONTE, "licenca": LICENCA,
                                                  "revisao": info.sha, "tribunal": r["tribunal"],
                                                  "classe": r["classe"], "documento_id": r["documento_id"]}},
                                ensure_ascii=False) + "\n")
    print(f"{len(amostra)} ementas de {len(trib)} tribunais ({dict((t, min(cota, len(por_trib[t]))) for t in trib)}) "
          f"-> {out}  [{FONTE}@{info.sha[:7]}, {LICENCA}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
