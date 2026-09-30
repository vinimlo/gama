# -*- coding: utf-8 -*-
"""Diagnóstico (não é o número do candidato): a métrica oficial do GLiNER fine-tunado com o
score dele como confiança do modelo.

O candidato sai no formato curto [ini, fim, rótulo, score]; na métrica oficial ele entra sem
confiança de modelo (faixa "regra" da calibração), como o GLiNER zero-shot publicado. Aqui os
mesmos spans são regravados no formato completo [ini, fim, tipo, forma, "", score] (tipo e forma
por `gama.formas.span_de`, o mesmo caminho de `bench.pontuar.para_span`), e o score cai nas faixas
alta/baixa da `calibracao.json`, que foi MEDIDA PARA O GAMA, não para este modelo. Serve só para
separar, na diferença para o Gama no estresse, o que é extração do que é faixa de calibração.

    docker compose run --rm gama python -m bench.controles.gliner_ft.diagnostico_confianca \\
        --spans saidas/bench/controles/_hub/bench/saida/controles/gliner_ft/modelo_gliner_ft.jsonl
    (depois: python -m bench.controles.avaliar pontuar --nome gliner_ft_diag_conf --spans <saída>)
"""
from __future__ import annotations

import argparse
import json
import pathlib

from bench.controles import avaliar as av
from gama.formas import DetectorDeForma


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", required=True)
    ap.add_argument("--saida", default="saidas/bench/controles/gliner_ft/diag_formato_completo.jsonl")
    a = ap.parse_args()
    textos = {}
    for c in av.CONJUNTOS:
        t, _ = av.textos_e_ouro(c)
        textos.update({(c, d): x for d, x in t.items()})
    destino = pathlib.Path(a.saida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(a.spans, encoding="utf-8") as fe, destino.open("w", encoding="utf-8") as fs:
        for linha in fe:
            r = json.loads(linha)
            if "meta" in r:
                r["meta"] = {**r["meta"], "diagnostico": "formato completo, score como confiança (calibração do Gama)"}
            elif (r["conjunto"], r["id"]) in textos:
                t = textos[(r["conjunto"], r["id"])]
                ss = [DetectorDeForma().span(t, s[0], s[1], s[2], s[3]) for s in r["spans"]]
                r["spans"] = [[s.inicio, s.fim, s.tipo, s.forma, s.digitos, s.confianca] for s in ss]
                n += 1
            fs.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(n, "documentos ->", destino)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
