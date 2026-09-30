# -*- coding: utf-8 -*-
"""Mede SÓ o recall de span — a etapa cujo erro não tem recuperação.

Uma citação sem span não chega na classificação: vira erro de recall e a
predição órfã vira falso positivo. Por isso esta métrica é medida isolada,
antes da nota, em toda mudança do extrator.
"""
from __future__ import annotations

import collections
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from gama.extratores.regua import ExtratorRegua  # noqa: E402

DADOS = pathlib.Path(__file__).resolve().parent.parent / "dados"


def iou(a, b):
    i = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return i / ((a[1] - a[0]) + (b[1] - b[0]) - i) if i else 0.0


def medir(mostrar: int = 0):
    rows = list(csv.DictReader(open(DADOS / "goldenset_offsets.csv", encoding="utf-8-sig")))
    por_doc = collections.defaultdict(list)
    for r in rows:
        por_doc[r["documento_id"]].append(r)

    tot = casado = fp = 0
    perdidos = collections.Counter()
    fp_forma = collections.Counter()
    ex_perd, ex_fp = [], []
    por_nivel = collections.Counter()

    for doc, golds in sorted(por_doc.items()):
        texto = (DADOS / "txt" / f"{doc}.txt").read_text(encoding="utf-8")
        preds = ExtratorRegua().extrair(texto)
        usados = set()
        for g in golds:
            tot += 1
            gi = (int(g["inicio"]), int(g["fim"]))
            melhor = None
            for k, p in enumerate(preds):
                if k in usados:
                    continue
                v = iou(gi, (p.inicio, p.fim))
                if v >= 0.5 and (melhor is None or v > melhor[0]):
                    melhor = (v, k)
            if melhor:
                casado += 1
                usados.add(melhor[1])
                por_nivel[(g["nivel"], True)] += 1
            else:
                perdidos[g["classificacao"]] += 1
                por_nivel[(g["nivel"], False)] += 1
                if len(ex_perd) < mostrar:
                    ex_perd.append((doc, g["classificacao"], g["trecho"][:66]))
        for k, p in enumerate(preds):
            if k not in usados:
                fp += 1
                fp_forma[p.forma] += 1
                if len(ex_fp) < mostrar:
                    ex_fp.append((doc, p.forma, p.trecho.replace("\n", " ")[:66]))

    print(f"RECALL DE SPAN : {casado}/{tot} = {casado / tot:.1%}")
    for n in ("1", "2"):
        ok, no = por_nivel[(n, True)], por_nivel[(n, False)]
        if ok + no:
            print(f"   nível {n}     : {ok}/{ok + no} = {ok / (ok + no):.1%}")
    print(f"FALSOS POSITIVOS: {fp}   {dict(fp_forma.most_common())}")
    if perdidos:
        print(f"perdidos por classe: {dict(perdidos)}")
    for d, c, t in ex_perd:
        print(f"  PERDIDO {d} [{c:10}] {t!r}")
    for d, f, t in ex_fp:
        print(f"  FP      {d} [{f:9}] {t!r}")
    return casado / tot, fp


if __name__ == "__main__":
    medir(mostrar=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
