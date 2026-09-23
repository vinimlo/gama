# -*- coding: utf-8 -*-
"""Catalogo de erros por categoria -- o motor de melhoria.

Casa predicao x gabarito pelo mesmo criterio da metrica oficial (IoU >= 0,5,
guloso por maior IoU) e classifica cada divergencia num balde acionavel:

  LINK_ERRADO    classe `real` certa, id_canonico errado  -> so precisao
  CLASSE_ERRADA  span casado, classe divergente           -> custa DUAS vezes
  NAO_EXTRAIDO   citacao do gabarito sem par              -> recall
  ESPURIO        predicao sem par                         -> precisao

A separacao importa porque o custo e diferente: link errado bate so em
fp[real]; classe errada bate em fn[esperada] E fp[predita].
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"


def iou(a, b):
    i = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return i / ((a[1] - a[0]) + (b[1] - b[0]) - i) if i else 0.0


def casar(golds, preds):
    cand = []
    for gi, g in enumerate(golds):
        for pi, p in enumerate(preds):
            v = iou((g["inicio"], g["fim"]), (p["inicio"], p["fim"]))
            if v >= 0.5:
                cand.append((-v, gi, pi))
    cand.sort()
    gu, pu, pares = set(), set(), []
    for _, gi, pi in cand:
        if gi in gu or pi in pu:
            continue
        gu.add(gi)
        pu.add(pi)
        pares.append((gi, pi))
    return pares, [i for i in range(len(golds)) if i not in gu], \
        [i for i in range(len(preds)) if i not in pu]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--limite", type=int, default=40)
    args = ap.parse_args(argv)
    pasta = pathlib.Path(args.pred)

    gold = collections.defaultdict(list)
    for r in csv.DictReader(open(DADOS / "goldenset_offsets.csv", encoding="utf-8-sig")):
        gold[r["documento_id"]].append(
            {"inicio": int(r["inicio"]), "fim": int(r["fim"]),
             "classe": r["classificacao"], "ids": set(r["id_canonico"].split()),
             "trecho": r["trecho"].replace("\\n", " "), "nivel": r["nivel"]})

    baldes = collections.defaultdict(list)
    for doc in sorted(gold):
        arq = pasta / f"{doc}.json"
        preds = []
        if arq.exists():
            for c in json.loads(arq.read_text(encoding="utf-8")).get("citacoes", []):
                resol = c.get("resolucao") or {}
                preds.append({"inicio": c["inicio"], "fim": c["fim"],
                              "classe": c["classificacao"],
                              "id": str(resol.get("id_canonico", "") or ""),
                              "trecho": c["trecho"].replace("\n", " ")})
        pares, sem_gold, sem_pred = casar(gold[doc], preds)
        for gi, pi in pares:
            g, p = gold[doc][gi], preds[pi]
            if g["classe"] != p["classe"]:
                baldes["CLASSE_ERRADA"].append(
                    (doc, g["nivel"], f'{g["classe"]}->{p["classe"]}', g["trecho"]))
            elif g["classe"] == "real" and p["id"] not in g["ids"]:
                baldes["LINK_ERRADO"].append(
                    (doc, g["nivel"], f'{p["id"] or "-"} != {"/".join(g["ids"])}', g["trecho"]))
        for gi in sem_gold:
            g = gold[doc][gi]
            baldes["NAO_EXTRAIDO"].append((doc, g["nivel"], g["classe"], g["trecho"]))
        for pi in sem_pred:
            p = preds[pi]
            baldes["ESPURIO"].append((doc, "?", p["classe"], p["trecho"]))

    total = sum(len(v) for v in baldes.values())
    print(f"TOTAL DE ERROS: {total} em 192 citacoes\n")
    for balde in ("CLASSE_ERRADA", "LINK_ERRADO", "NAO_EXTRAIDO", "ESPURIO"):
        itens = baldes.get(balde, [])
        if not itens:
            continue
        print(f"--- {balde}  ({len(itens)}) ---")
        for doc, nivel, detalhe, trecho in itens[:args.limite]:
            print(f"  n{nivel} {doc:12} {detalhe:34} {trecho[:58]!r}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
