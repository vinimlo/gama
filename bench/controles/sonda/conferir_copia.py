# -*- coding: utf-8 -*-
"""Confere que as funções copiadas de `treino/treinar.py` para `treinar_sonda.py` são literais (AST igual).

    docker compose run --rm gama python -m bench.controles.sonda.conferir_copia
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys

APP = pathlib.Path(__file__).resolve().parents[3]
COPIADAS = ["ROT", "_bio", "carregar", "_deslocamento", "especiais", "exemplos", "prever", "f1_spans"]


def _defs(arq: pathlib.Path) -> dict:
    out = {}
    for no in ast.parse(arq.read_text(encoding="utf-8")).body:
        if isinstance(no, ast.FunctionDef):
            out[no.name] = ast.dump(no)
        elif isinstance(no, ast.Assign) and len(no.targets) == 1 and isinstance(no.targets[0], ast.Name):
            out[no.targets[0].id] = ast.dump(no)
    return out


def main() -> int:
    orig = _defs(APP / "treino" / "treinar.py")
    copia = _defs(APP / "bench" / "controles" / "sonda" / "treinar_sonda.py")
    res = {n: n in orig and orig.get(n) == copia.get(n) for n in COPIADAS}
    print(json.dumps(res, ensure_ascii=False))
    return 0 if all(res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
