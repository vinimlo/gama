# -*- coding: utf-8 -*-
"""Minera citacoes REAIS do acervo -- a unica fonte de variacao que nao saiu
da nossa cabeca.

Por que isto existe. O extrator marca 100% de recall no conjunto de
desenvolvimento, mas foi ajustado a ele. Um gerador sintetico nosso nao
corrige esse vies: ele produz o ruido que NOS imaginamos, e a forma de
superficie que quebra o extrator e justamente a que nao imaginamos.

Os 996 acordaos do acervo sao 67,7 M de caracteres de escrita juridica real,
e cada um cita dezenas de outros feitos. Isso da:
  - avaliacao de recall com milhares de casos reais;
  - corpus de treino com rotulo derivado do indice;
  - catalogo de formas de superficie que ninguem enumerou a mao.

O rotulo e FRACO mas REAL: se a chave de um registro aparece no texto de
outro, ali existe uma citacao escrita por um humano.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import re
import sqlite3
import sys

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from gama.extrair import extrair  # noqa: E402
from gama.indice import construir  # noqa: E402
from gama.normalizar import chave_processo  # noqa: E402

# Numeros que PARECEM processo mas nao sao. Sem este filtro a medicao infla:
# "DJe 24.11.2014" vira chave 24112014, que pode colidir com algum registro.
DATA = re.compile(
    r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b"
    r"|\b\d{1,2}\s+de\s+\w+\s+de\s+\d{4}\b"
)
CONTEXTO_DATA = re.compile(
    r"(?:DJ|DJe|DJU|PSESS|publicad[oa]|julgad[oa]\s+em|sess[aã]o\s+de"
    r"|divulgad[oa]|em)\s*[.:]?\s*$",
    re.I,
)
# Candidato a numero de processo dentro do corpo.
NUMERO = re.compile(r"\d{1,3}(?:\.\d{3})+|\b\d{5,}\b")

JANELA_CABECALHO = 400   # o proprio numero do feito vive aqui


def _zonas_de_data(texto: str) -> list:
    return [(m.start(), m.end()) for m in DATA.finditer(texto)]


def _dentro(pos: int, zonas: list) -> bool:
    return any(a <= pos < b for a, b in zonas)


def ocorrencias_reais(texto: str, doc_id: int, idx) -> list:
    """Posicoes onde ha citacao real a outro feito, com o id alvo.

    Filtra data por padrao E por contexto ("DJe 24.11.2014"), e ignora o
    cabecalho, onde mora o numero do proprio documento.
    """
    zonas = _zonas_de_data(texto)
    achados = []
    for m in NUMERO.finditer(texto):
        pos = m.start()
        if pos < JANELA_CABECALHO or _dentro(pos, zonas):
            continue
        if CONTEXTO_DATA.search(texto[max(0, pos - 24):pos]):
            continue
        chave = chave_processo(m.group(0))
        if len(chave) < 5:
            continue
        cands = [c for c in idx.candidatos_processo(chave) if c != doc_id]
        if not cands:
            continue
        achados.append({"pos": pos, "fim_num": m.end(),
                        "numero": m.group(0), "alvos": cands})
    return achados


def medir(idx, docs, amostra: int, seed: int = 0):
    """Recall do extrator sobre citacoes reais. Devolve (stats, perdidas)."""
    rng = random.Random(seed)
    escolhidos = rng.sample(docs, min(amostra, len(docs)))
    achou = perdeu = 0
    formas = collections.Counter()
    exemplos = []
    for doc_id, texto in escolhidos:
        spans = extrair(texto)
        cobertos = [(s.inicio, s.fim) for s in spans]
        for oc in ocorrencias_reais(texto, doc_id, idx):
            if any(a <= oc["pos"] < b for a, b in cobertos):
                achou += 1
            else:
                perdeu += 1
                ctx = texto[max(0, oc["pos"] - 46):oc["fim_num"] + 14]
                ctx = re.sub(r"\s+", " ", ctx).strip()
                formas[re.sub(r"\d", "#", ctx[:60])] += 1
                if len(exemplos) < 25:
                    exemplos.append(ctx)
    total = achou + perdeu
    return {"total": total, "achou": achou, "perdeu": perdeu,
            "recall": achou / total if total else 0.0}, formas, exemplos


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(RAIZ / "dados" / "desafio1_bracis.db"))
    ap.add_argument("--amostra", type=int, default=150)
    ap.add_argument("--exemplos", type=int, default=16)
    ap.add_argument("--dump", help="salva as ocorrencias mineradas em JSONL")
    args = ap.parse_args(argv)

    idx = construir(args.db)
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    docs = con.execute(
        "SELECT id, texto FROM documentos WHERE natureza='acordao'").fetchall()
    con.close()

    stats, formas, exemplos = medir(idx, docs, args.amostra)
    print(f"=== RECALL EM CITACOES REAIS DO ACERVO ({args.amostra} acordaos) ===")
    print(f"  citacoes reais encontradas : {stats['total']}")
    print(f"  o extrator pegou           : {stats['achou']} ({stats['recall']:.1%})")
    print(f"  o extrator perdeu          : {stats['perdeu']}")
    print(f"\n--- formas mais perdidas (digitos mascarados como #) ---")
    for forma, n in formas.most_common(args.exemplos):
        print(f"  {n:3}x  {forma!r}")
    print(f"\n--- exemplos literais ---")
    for e in exemplos[:args.exemplos]:
        print(f"  {e!r}")

    if args.dump:
        destino = pathlib.Path(args.dump)
        destino.parent.mkdir(parents=True, exist_ok=True)
        with destino.open("w", encoding="utf-8") as f:
            for doc_id, texto in docs:
                for oc in ocorrencias_reais(texto, doc_id, idx):
                    f.write(json.dumps({"doc": doc_id, "pos": oc["pos"],
                                        "numero": oc["numero"],
                                        "alvos": oc["alvos"],
                                        "ctx": texto[max(0, oc["pos"] - 90):
                                                     oc["fim_num"] + 30]},
                                       ensure_ascii=False) + "\n")
        print(f"\ndump -> {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
