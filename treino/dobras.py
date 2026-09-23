# -*- coding: utf-8 -*-
"""Experimento honesto: goldenset de cada dobra gerado SEM os documentos avaliados.

Para cada dobra d dos 26 documentos da org: bancos extraídos das OUTRAS dobras
(+ expansão por LLM, na variante 'llm'), goldenset gerado, e a dobra d vira o
conjunto de avaliação (gabarito filtrado) — pontuado depois pelo harness oficial.
É a melhor estimativa que temos do cego: frases de molde que o modelo nunca viu.

    docker compose run --rm lab python -m treino.dobras --n 1500
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

from gama.indice import construir
from geracao import fichas as F
from geracao.bancos import dobras_do_dev, extrair
from geracao.gerar import escrever
from geracao.montar import Montador, mistura_do_dev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--dados", default="/app/dados")
    ap.add_argument("--db", default="/app/dados/desafio1_bracis.db")
    ap.add_argument("--llm", default="/app/corpus/bancos/llm_v1.json")
    ap.add_argument("--saida", default="/app/corpus/dobras")
    a = ap.parse_args()
    dados, saida = pathlib.Path(a.dados), pathlib.Path(a.saida)
    idx = construir(a.db)
    fichas, sumulas, disps = F.carregar(a.db)
    mistura = mistura_do_dev(dados)
    llm = json.loads(pathlib.Path(a.llm).read_text(encoding="utf-8"))
    for d, fora in enumerate(dobras_do_dev(dados)):
        # gabarito da dobra avaliada
        (saida / f"dobra{d}").mkdir(parents=True, exist_ok=True)
        with open(dados / "goldenset_offsets.csv", encoding="utf-8-sig", newline="") as fe, \
                open(saida / f"dobra{d}" / "gabarito.csv", "w", encoding="utf-8", newline="") as fs:
            r, w = csv.reader(fe), csv.writer(fs)
            w.writerow(next(r))
            w.writerows(x for x in r if x[1] in fora)
        (saida / f"dobra{d}" / "docs.txt").write_text("\n".join(sorted(fora)) + "\n")
        for variante in ("org", "llm"):
            bancos = extrair(dados, excluir=fora)
            if variante == "llm":
                for k, v in llm.items():
                    bancos.setdefault(k, []).extend(v)
            m = Montador(bancos, fichas, sumulas, disps, mistura, idx)
            rel = escrever(m, idx, a.n, 100 + d, saida / f"dobra{d}" / variante)
            print(f"dobra {d} {variante}: {rel['documentos']} docs, {rel['citacoes']} citações, "
                  f"descartes {rel['descartes']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
