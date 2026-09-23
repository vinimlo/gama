# -*- coding: utf-8 -*-
"""Ida e volta da tokenização: rotular -> decodificar reproduz o gabarito com IoU 1,0?

    docker compose run --rm lab python -m treino.gate --modelo jhu-clsp/mmBERT-base

Roda sobre o dev set da organização e sobre um goldenset gerado. Qualquer span que
não volte idêntico é listado — e bloqueia o treino daquele tokenizador.
"""
from __future__ import annotations

import argparse
import collections
import csv
import pathlib

from transformers import AutoTokenizer

from gama.extratores.bio import decodificar, rotular

ROT = {"incompleta": "VAGA"}


def carregar(pasta: pathlib.Path) -> dict:
    """{doc: (texto, [(ini, fim, tipo)])} a partir de goldenset_offsets.csv + txt/."""
    spans = collections.defaultdict(list)
    for r in csv.DictReader(open(pasta / "goldenset_offsets.csv", encoding="utf-8-sig")):
        tipo = ROT.get(r["classificacao"]) or ("LEI" if r["tipo"] == "lei" else "JURIS")
        spans[r["documento_id"]].append((int(r["inicio"]), int(r["fim"]), tipo))
    def ler(d):
        with open(pasta / "txt" / f"{d}.txt", encoding="utf-8", newline="") as fh:
            return fh.read()
    return {d: (ler(d), sp) for d, sp in spans.items()}


def checar(tok, docs: dict) -> tuple[int, int, list]:
    total = erros = 0
    exemplos = []
    for doc, (texto, spans) in docs.items():
        enc = tok(texto, return_offsets_mapping=True, add_special_tokens=True)
        offs = [tuple(o) for o in enc["offset_mapping"]]
        volta = set(decodificar(offs, rotular(offs, spans), texto))
        for sp in spans:
            total += 1
            if sp not in volta:
                erros += 1
                if len(exemplos) < 12:
                    perto = [v for v in volta if v[0] < sp[1] and sp[0] < v[1]]
                    exemplos.append((doc, texto[sp[0]:sp[1]], [texto[a:b] for a, b, _ in perto]))
        for v in volta - set(spans):
            if not any(v[0] < b and a < v[1] for a, b, _ in spans):
                erros += 1
    return total, erros, exemplos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--dev", default="/app/dados")
    ap.add_argument("--sint", default="/app/corpus/goldenset/v0")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.modelo)
    ok = True
    for nome, pasta in (("dev", a.dev), ("sintético", a.sint)):
        total, erros, ex = checar(tok, carregar(pathlib.Path(pasta)))
        print(f"{a.modelo} | {nome}: {total - erros}/{total} spans voltam idênticos")
        for e in ex:
            print("   ", e)
        ok &= erros == 0
    print("PORTÃO", "ABERTO" if ok else "FECHADO")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
