# -*- coding: utf-8 -*-
"""Sorteia um teste real intocado e separa o que sobra para a destilação.

As 305 ementas do ouro atual ajudaram a escolher a guarda (D-008), então não servem mais
de teste limpo. Este sorteio tira N ementas novas, ao acaso (sem enriquecer por
divergência), de fora das 360 que `reais.anotar` já tinha selecionado, e sem texto
repetido de nenhuma delas. O resto, também sem repetição, é o conjunto sem rótulo que a
destilação pode ver.

    docker compose run --rm gama python -m reais.sortear_teste
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re

RAIZ = pathlib.Path("/app/corpus/reais")


def _chave(texto: str) -> str:
    return re.sub(r"\s+", " ", texto).strip().lower()


def selecionadas_antes(preds: dict, divergentes: int = 300, concordantes: int = 60) -> set:
    """Reproduz a seleção de `reais.anotar` (mesma regra, mesma semente)."""
    div, conc = [], []
    for i, p in preds.items():
        g = [tuple(s[:2]) for s in p["gama"]]
        r = [tuple(s[:2]) for s in p["regua"]]
        (div if set(g) != set(r) else conc).append(i)
    rng = random.Random(3)
    return set(rng.sample(sorted(div), min(divergentes, len(div)))) | \
        set(rng.sample(sorted(conc), min(concordantes, len(conc))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--semente", type=int, default=2026)
    ap.add_argument("--saida", default=str(RAIZ / "novo"))
    a = ap.parse_args()
    textos = {r["id"]: r["texto"] for r in map(json.loads, open(RAIZ / "amostra.jsonl", encoding="utf-8"))}
    preds = {r["id"]: r for r in map(json.loads, open(RAIZ / "predicoes.jsonl", encoding="utf-8"))}
    antes = selecionadas_antes(preds)
    vistos = {_chave(textos[i]) for i in antes}
    livres, chaves = [], set()
    for i in sorted(textos):
        k = _chave(textos[i])
        if i in antes or k in vistos or k in chaves:
            continue
        chaves.add(k)
        livres.append(i)
    teste = sorted(random.Random(a.semente).sample(livres, a.n))
    destilacao = [i for i in livres if i not in set(teste)]
    saida = pathlib.Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "ids.json").write_text(json.dumps(teste, indent=0))
    (RAIZ / "destilacao_ids.json").write_text(json.dumps(destilacao, indent=0))
    print(json.dumps({"ementas": len(textos), "ja_selecionadas": len(antes),
                      "livres_sem_repeticao": len(livres), "teste_novo": len(teste),
                      "destilacao": len(destilacao)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
