# -*- coding: utf-8 -*-
"""Contraste principal do H2: a sonda (encoder congelado) contra o v1.2 (mesma receita, encoder treinado).

O harness (`bench.controles.avaliar`) compara todo candidato com o Gama v1.3 com a guarda. A pergunta
do H2 é outra, "adaptar o encoder acrescenta além de treinar a cabeça?", e o plano manda começar
pelo extrator sem guarda. Aqui só se trocam os dois lados do mesmo bootstrap do harness
(`avaliar.bootstrap`: pareado por ementa, 2.000 reamostras, semente 0): sonda - v1.2, cru com cru e
guarda@0,95 com guarda@0,95, nas 305 e nas 172. No estresse, o F1 de extração dos dois (VAGA separada).
Nada é reimplementado: leitura, aparo, guarda, F1 e bootstrap são os do harness.

    docker compose run --rm gama python -m bench.controles.sonda.comparar \\
        --sonda saidas/bench/controles/sonda/modelo_sonda_<lr>.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib

from bench.controles import avaliar

V12 = avaliar.BENCH / "modelo_professor.jsonl"        # vinimlo/gama@ad06ffd, os spans da validação do harness


def _variantes(arq: pathlib.Path) -> dict:
    _, linhas = avaliar.ler([arq])
    out = {}
    for c in avaliar.CONJUNTOS:
        sp = avaliar.spans(linhas, c)
        if sp is None:
            continue
        out[c] = {"cru": sp, "guarda@0.95": avaliar.com_guarda(sp, avaliar.regua(c), avaliar.LIMIAR_PRODUCAO)}
    return out


def comparar(sonda: pathlib.Path, v12: pathlib.Path = V12) -> dict:
    a, b = _variantes(sonda), _variantes(v12)
    out = {"sonda": str(sonda), "v12": str(v12), "codigo": avaliar.impressao(), "conjuntos": {}}
    for c in avaliar.CONJUNTOS:
        if c not in a or c not in b:
            continue
        res = {}
        for v in ("cru", "guarda@0.95"):
            r = {"sonda_f1": avaliar.extracao(c, a[c][v])["f1"], "v12_f1": avaliar.extracao(c, b[c][v])["f1"]}
            if c != "estresse":
                r["sonda_menos_v12"] = avaliar.bootstrap(avaliar.por_doc(c, a[c][v]), avaliar.por_doc(c, b[c][v]))
            res[v] = r
            print(c, v, json.dumps(r, ensure_ascii=False), flush=True)
        out["conjuntos"][c] = res
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sonda", required=True)
    ap.add_argument("--v12", default=str(V12))
    ap.add_argument("--saida", default=str(avaliar.SAIDA / "sonda" / "comparacao_v12.json"))
    a = ap.parse_args()
    res = comparar(pathlib.Path(a.sonda), pathlib.Path(a.v12))
    destino = pathlib.Path(a.saida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", destino)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
