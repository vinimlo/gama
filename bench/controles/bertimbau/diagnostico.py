# -*- coding: utf-8 -*-
"""Diagnóstico do H3 (exploratório, depois dos números): onde o BERTimbau difere do v1.2.

1. Confiança dos spans crus por conjunto: quantos ficam abaixo de 0,95 e de 0,98 (a guarda troca
   os de baixo pela régua; a calibração oficial separa a faixa alta em 0,98, medida no Gama).
2. Recall cru nas 305 e nas 172 separado por citação do ouro que tem ou não token [UNK] no
   tokenizador do BERTimbau ("nº", "1º": o vocabulário não tem "º" depois de letra ou dígito).
   Usa `bench.pontuar.extracao` só sobre o subconjunto do ouro (o FP não se aplica aqui).

    docker compose run --rm -e HF_HOME=/app/saidas/bench/controles/bertimbau/hf gama \\
        python -m bench.controles.bertimbau.diagnostico --bertimbau <jsonl> --v12 <jsonl> --v13 <jsonl>
"""
from __future__ import annotations

import argparse
import json

from transformers import AutoTokenizer

from ... import pontuar
from .. import avaliar

BERT =("neuralmind/bert-base-portuguese-cased", "94d69c95f98f7d5b2a8700c420230ae10def0baa")  # o de checar_janelas.py

AQUI = avaliar.SAIDA / "bertimbau"


def confiancas(linhas: dict) -> dict:
    out = {}
    for c in avaliar.CONJUNTOS:
        textos, _ = avaliar.textos_e_ouro(c)
        cs = sorted(s[5] for d in textos for s in linhas[(c, d)]["spans"] if s[5] is not None)
        out[c] = {"spans": len(cs), "abaixo_095": sum(x < 0.95 for x in cs), "abaixo_098": sum(x < 0.98 for x in cs),
                  "min": round(cs[0], 4) if cs else None, "mediana": round(cs[len(cs) // 2], 4) if cs else None}
    return out


def recall_por_unk(tok, arquivos: dict) -> dict:
    out = {}
    for c in ("reais", "novas"):
        textos, ouro = avaliar.textos_e_ouro(c)
        com, sem = {}, {}
        for d, t in textos.items():
            enc = tok(t, return_offsets_mapping=True, add_special_tokens=False)
            unk = [o for i, o in zip(enc["input_ids"], enc["offset_mapping"]) if i == tok.unk_token_id]
            com[d] = [g for g in ouro.get(d, []) if any(s < g[1] and g[0] < e for s, e in unk)]
            sem[d] = [g for g in ouro.get(d, []) if g not in com[d]]
        out[c] = {"ouro_com_unk": sum(map(len, com.values())), "ouro_sem_unk": sum(map(len, sem.values()))}
        for nome, arq in arquivos.items():
            _, linhas = avaliar.ler([arq])
            pred = avaliar.triplas(avaliar.spans(linhas, c))
            for rot, g in (("com_unk", com), ("sem_unk", sem)):
                r = pontuar.extracao(g, pred, True)
                tp = sum(v["tp"] for v in r["por_tipo"].values())
                fn = sum(v["fn"] for v in r["por_tipo"].values())
                out[c][f"recall_cru_{nome}_{rot}"] = round(tp / max(1, tp + fn), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bertimbau", required=True)
    ap.add_argument("--v12", required=True)
    ap.add_argument("--v13", required=True)
    a = ap.parse_args()
    arqs = {"bertimbau": a.bertimbau, "v12": a.v12, "v13": a.v13}
    out = {"confianca_crua": {n: confiancas(avaliar.ler([p])[1]) for n, p in arqs.items()},
           "recall_por_unk": recall_por_unk(AutoTokenizer.from_pretrained(BERT[0], revision=BERT[1]), arqs)}
    (AQUI / "diagnostico.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
