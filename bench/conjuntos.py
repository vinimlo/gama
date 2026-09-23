# -*- coding: utf-8 -*-
"""Os três conjuntos do benchmark e seus gabaritos, lidos do jeito que o pipeline lê.

    dev       26 documentos da organização (gabarito oficial); não saem da máquina
    estresse  600 documentos sintéticos com redação nunca treinada, N1 + N2
    reais     305 ementas reais (STF, STJ, TJRJ; CC-BY-4.0), citações adjudicadas

Gera as entradas dos extratores crus (`python -m bench.conjuntos`): uma linha JSON por
documento, `{"conjunto", "id", "texto"}`. Só estresse e reais sobem para o HF Jobs.
"""
from __future__ import annotations

import csv
import json
import pathlib

from gama.pipeline import ler_texto

RAIZ = pathlib.Path("/app")
DEV = RAIZ / "dados"
ESTRESSE = RAIZ / "corpus" / "goldenset" / "estresse_glm"
REAIS_TEXTOS = RAIZ / "corpus" / "reais" / "amostra.jsonl"
REAIS_OURO = RAIZ / "corpus" / "reais" / "ouro_real.jsonl"
SAIDA = RAIZ / "saidas" / "bench"


def _rotulo(r: dict) -> str:
    """Tipo de extração de uma linha do gabarito: JURIS, LEI ou VAGA."""
    if r["classificacao"] == "incompleta":
        return "VAGA"
    return "LEI" if r["tipo"] == "lei" else "JURIS"


def _com_gabarito_csv(pasta: pathlib.Path) -> tuple[dict, dict]:
    gold: dict[str, list] = {}
    for r in csv.DictReader(open(pasta / "goldenset_offsets.csv", encoding="utf-8-sig")):
        gold.setdefault(r["documento_id"], []).append((int(r["inicio"]), int(r["fim"]), _rotulo(r)))
    textos = {d: ler_texto(pasta / "txt" / f"{d}.txt") for d in sorted(gold)}
    return textos, gold


def carregar(nome: str) -> tuple[dict, dict]:
    """(textos por id, gabarito de extração por id: [(inicio, fim, rótulo)])."""
    if nome == "dev":
        return _com_gabarito_csv(DEV)
    if nome == "estresse":
        return _com_gabarito_csv(ESTRESSE)
    if nome == "reais":
        ouro = {}
        for linha in REAIS_OURO.read_text(encoding="utf-8").splitlines():
            r = json.loads(linha)
            ouro[r["id"]] = [tuple(s) for s in r["spans"]]
        textos = {}
        for linha in REAIS_TEXTOS.read_text(encoding="utf-8").splitlines():
            r = json.loads(linha)
            if r["id"] in ouro:
                textos[r["id"]] = r["texto"]
        return {k: textos[k] for k in sorted(ouro)}, ouro
    raise ValueError(nome)


def gabarito_oficial(nome: str) -> pathlib.Path | None:
    """CSV para a métrica oficial (só onde há `id_canonico`, isto é, dev e estresse)."""
    pasta = {"dev": DEV, "estresse": ESTRESSE}.get(nome)
    return pasta / "goldenset_offsets.csv" if pasta else None


def main() -> int:
    SAIDA.mkdir(parents=True, exist_ok=True)
    for arquivo, nomes in (("entrada_dev.jsonl", ["dev"]), ("entrada_remota.jsonl", ["estresse", "reais"])):
        with open(SAIDA / arquivo, "w", encoding="utf-8") as fh:
            for nome in nomes:
                textos, _ = carregar(nome)
                for i, t in textos.items():
                    fh.write(json.dumps({"conjunto": nome, "id": i, "texto": t}, ensure_ascii=False) + "\n")
                print(nome, len(textos), "documentos ->", arquivo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
