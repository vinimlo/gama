# -*- coding: utf-8 -*-
"""Roda a metrica OFICIAL (vendor/kaggle_metric.py) sobre as nossas saidas.

O script do vendor e copia exata do que roda no servidor do Kaggle, entao o
numero impresso aqui e o numero do leaderboard -- nivel a nivel, incluindo o
matching por IoU. Nada aqui reimplementa a metrica; so monta os dois
DataFrames no formato que ela exige.

  solution   : documento_id, nivel, citacoes  ("inicio,fim,classe,doc_ids")
  submission : documento_id, citacoes         ("inicio,fim,classe,id,conf")
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "src"))

import pandas as pd  # noqa: E402

from vendor.kaggle_metric import avaliar  # noqa: E402

DADOS = RAIZ / "dados"


def montar_solution(caminho_gold: pathlib.Path) -> pd.DataFrame:
    linhas = collections.defaultdict(list)
    nivel = {}
    for r in csv.DictReader(open(caminho_gold, encoding="utf-8-sig")):
        doc = r["documento_id"]
        nivel[doc] = int(r["nivel"])
        ids = r["id_canonico"].strip().replace(" ", ":") or "-"
        linhas[doc].append(f'{r["inicio"]},{r["fim"]},{r["classificacao"]},{ids}')
    return pd.DataFrame(
        [{"documento_id": d, "nivel": nivel[d], "citacoes": "|".join(c)}
         for d, c in sorted(linhas.items())])


def montar_submission(pasta_json: pathlib.Path, docs: list) -> pd.DataFrame:
    linhas = []
    for doc in docs:
        arq = pasta_json / f"{doc}.json"
        partes = []
        if arq.exists():
            for c in json.loads(arq.read_text(encoding="utf-8")).get("citacoes", []):
                resol = c.get("resolucao") or {}
                idc = str(resol.get("id_canonico", "") or "").strip() or "-"
                conf = c.get("confianca")
                conf_s = "-" if conf is None else f"{float(conf):.4f}"
                partes.append(
                    f'{int(c["inicio"])},{int(c["fim"])},{c["classificacao"]},{idc},{conf_s}')
        linhas.append({"documento_id": doc, "citacoes": "|".join(partes) or "-"})
    return pd.DataFrame(linhas)


def relatorio(res: dict) -> None:
    print("=" * 62)
    for nivel, d in sorted(res["niveis"].items()):
        f1 = d["f1_por_classe"]
        print(f"NIVEL {nivel}  (peso {1 if nivel == 1 else 2}x)")
        print(f"   F1 real       {f1.get('real', 0):.4f}")
        print(f"   F1 inventada  {f1.get('inventada', 0):.4f}")
        print(f"   F1 incompleta {f1.get('incompleta', 0):.4f}")
        print(f"   macro-F1      {d['macro_f1']:.4f}")
        print(f"   tau (grave)   {d['tau']:.4f}   -> s = {d['s']:.4f}")
        print(f"   bonus calib.  {d['b']:.4f}   -> score = {d['score']:.4f}")
    print("-" * 62)
    print(f"SCORE FINAL   {res['score_final']:.5f}"
          f"     (teto pratico 1.10000)")
    print("=" * 62)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="pasta com os .json preditos")
    ap.add_argument("--gold", default=str(DADOS / "goldenset_offsets.csv"))
    args = ap.parse_args(argv)

    solution = montar_solution(pathlib.Path(args.gold))
    submission = montar_submission(pathlib.Path(args.pred),
                                   solution["documento_id"].tolist())
    relatorio(avaliar(solution, submission))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
