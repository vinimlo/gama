# -*- coding: utf-8 -*-
"""Pontua um extrator num conjunto rotulado qualquer, pela métrica OFICIAL.

    python -m treino.avaliar_conjunto --conjunto /app/corpus/goldenset/estresse_glm \\
        --extrator neural --modelo vinimlo/gama [--revisao SHA]

Usado para desempatar candidatos quando o dev set satura (todos em F1 = 1,0).
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil

from gama.extratores import carregar
from gama.indice import construir

from .avaliar_dobras import pontuar, rodar


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", required=True)
    ap.add_argument("--extrator", default="neural")
    ap.add_argument("--modelo", default=None, help="repo do Hub ou pasta local")
    ap.add_argument("--revisao", default=None)
    ap.add_argument("--nome", default=None)
    ap.add_argument("--limite", type=int, default=None)
    a = ap.parse_args()
    conj = pathlib.Path(a.conjunto)
    gold = conj / "goldenset_offsets.csv"
    docs = sorted({r["documento_id"] for r in csv.DictReader(open(gold, encoding="utf-8-sig"))})
    if a.limite:
        # Estratificado por nível: ordem alfabética põe todo syn_n1_* antes de syn_n2_*, e
        # um corte simples mediria só texto limpo (erro cometido e corrigido em 22/09).
        n1 = [d for d in docs if "_n1_" in d]
        n2 = [d for d in docs if "_n1_" not in d]
        docs = sorted(n1[: a.limite // 2] + n2[: a.limite - a.limite // 2])
        filtrado = pathlib.Path("/tmp/gold_filtrado.csv")
        with open(gold, encoding="utf-8-sig", newline="") as fe, open(filtrado, "w", encoding="utf-8", newline="") as fs:
            r, w = csv.reader(fe), csv.writer(fs)
            w.writerow(next(r))
            w.writerows(x for x in r if x[1] in set(docs))
        gold = filtrado
    local = None
    if a.modelo and not pathlib.Path(a.modelo).exists():
        from huggingface_hub import snapshot_download
        local = pathlib.Path("/app/corpus/modelos/tmp_conj")
        if local.exists():
            shutil.rmtree(local)
        snapshot_download(a.modelo, revision=a.revisao, local_dir=str(local))
    pasta_modelo = str(local or a.modelo) if a.modelo else None
    idx = construir("/app/dados/desafio1_bracis.db")
    nome = a.nome or f"{a.extrator}:{a.modelo or ''}"
    saida = pathlib.Path("/app/saidas/conjuntos") / nome.replace("/", "_").replace(":", "_")
    rodar(carregar(a.extrator, pasta_modelo), idx, docs, conj, saida)
    res = pontuar(gold, saida)
    print(nome, json.dumps(res, ensure_ascii=False))
    arq = conj / "resultados.json"
    anterior = json.loads(arq.read_text()) if arq.exists() else {}
    arq.write_text(json.dumps({**anterior, nome: res}, ensure_ascii=False, indent=1))
    if local:
        shutil.rmtree(local)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
