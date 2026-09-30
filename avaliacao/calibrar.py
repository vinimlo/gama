# -*- coding: utf-8 -*-
"""D-006: confiança = probabilidade de acerto medida por balde.

Roda o pipeline em conjuntos ROTULADOS, casa cada predição com o gabarito como a
métrica oficial (1-para-1, IoU >= 0,5) e mede, por balde (via × classe × faixa da
confiança do extrator), a fração de pares casados em que classe E link estão
certos — exatamente o y do Brier em vendor/kaggle_metric.py.

Estimativa: média a posteriori Beta(1,1), (acertos+1)/(n+2), com teto. O excesso de
Brier é (c − p)², plano perto do ótimo: o teto protege contra um cego um pouco mais
difícil que os nossos dados, quase sem custo se ele não for.

    python -m avaliacao.calibrar --conjunto /app/dados --conjunto /app/corpus/goldenset/estresse_glm \\
        --extrator neural --modelos /models --saida /app/src/gama/calibracao.json
"""
from __future__ import annotations

import argparse
import collections
import re
import csv
import json
import pathlib

from gama.extratores import CatalogoDeExtratores
from gama.indice import Indice
from gama.pipeline import Documento, Pipeline

TETO = 0.995


def _gabarito(pasta: pathlib.Path) -> dict:
    g = collections.defaultdict(list)
    for r in csv.DictReader(open(pasta / "goldenset_offsets.csv", encoding="utf-8-sig")):
        g[r["documento_id"]].append((int(r["inicio"]), int(r["fim"]), r["classificacao"],
                                     r["id_canonico"] or None))
    return g


def _iou(a, b, c, d) -> float:
    inter = max(0, min(b, d) - max(a, c))
    return inter / (max(b, d) - min(a, c)) if inter else 0.0


def casar(preds: list, golds: list) -> list[tuple]:
    """1-para-1 guloso, IDÊNTICO a vendor/kaggle_metric._casar: candidatos
    (-IoU, índice do gold, índice da predição) em ordem crescente (revisão independente, rodada 1, achado 3)."""
    cand = sorted((-_iou(p.inicio, p.fim, g[0], g[1]), j, i) for i, p in enumerate(preds)
                  for j, g in enumerate(golds) if _iou(p.inicio, p.fim, g[0], g[1]) >= 0.5)
    usados_p, usados_g, pares = set(), set(), []
    for _, j, i in cand:
        if i in usados_p or j in usados_g:
            continue
        usados_p.add(i); usados_g.add(j)
        pares.append((preds[i], golds[j]))
    return pares


def _norm_id(v) -> str:
    v = str(v).strip()
    return (v.lstrip("0") or "0") if v.isdigit() else v


def mesmo_id(pred, gold) -> bool:
    """Como a métrica: o gabarito pode listar vários ids aceitos ("123 456" no CSV,
    "123:456" no transporte); espaço e zero à esquerda normalizados (revisão
    independente: rodada 1, achado 4; rodada 2, achado 6)."""
    if pred is None or gold is None:
        return False
    aceitos = {_norm_id(x) for x in re.split(r"[\s:|,;]+", str(gold).strip()) if x}
    return _norm_id(pred) in aceitos


def medir(conjuntos: list[pathlib.Path], extrator, idx, limite: int | None) -> dict:
    stats = collections.defaultdict(lambda: [0, 0])          # balde -> [acertos, n]
    for pasta in conjuntos:
        gab = _gabarito(pasta)
        docs = sorted(gab)
        if limite:
            # estratificado por nível (ver treino/avaliar_conjunto.py)
            n1 = [d for d in docs if "_n1_" in d]
            n2 = [d for d in docs if "_n1_" not in d]
            docs = sorted(n1[: limite // 2] + n2[: limite - limite // 2])
        for d in docs:
            texto = Documento.ler(pasta / "txt" / f"{d}.txt").texto
            for p, g in casar(Pipeline(idx, extrator).processar(texto), gab[d]):
                y = p.classificacao == g[2] and (g[2] != "real" or mesmo_id(p.id_canonico, g[3]))
                s = stats["|".join(p.balde)]
                s[0] += int(y)
                s[1] += 1
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", action="append", required=True, help="pasta com txt/ + goldenset_offsets.csv")
    ap.add_argument("--extrator", default="neural")
    ap.add_argument("--modelos", default="/models")
    ap.add_argument("--db", default="/app/dados/desafio1_bracis.db")
    ap.add_argument("--limite", type=int, default=None, help="documentos por conjunto")
    ap.add_argument("--saida", default=None)
    a = ap.parse_args()
    idx = Indice.do_banco(a.db)
    extrator = CatalogoDeExtratores(a.modelos).carregar(a.extrator)
    stats = medir([pathlib.Path(c) for c in a.conjunto], extrator, idx, a.limite)
    tabela = {}
    for balde, (k, n) in sorted(stats.items()):
        tabela[balde] = round(min(TETO, (k + 1) / (n + 2)), 4)
        print(f"{balde:34s} {k:5d}/{n:<5d} acc={k / n:.4f} -> conf {tabela[balde]}")
    if a.saida:
        pathlib.Path(a.saida).write_text(json.dumps(tabela, indent=1, sort_keys=True), encoding="utf-8")
        print("tabela em", a.saida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
