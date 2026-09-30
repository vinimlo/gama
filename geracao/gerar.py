# -*- coding: utf-8 -*-
"""Gera o goldenset sintético no formato da organização.

    python -m geracao.gerar --n 3000 --semente 7 --saida /app/corpus/goldenset/v1

Saída (mesmo esquema do Kaggle, para o harness oficial ler sem adaptação):
    txt/syn_n{1,2}_NNNNN.txt
    goldenset_offsets.csv   nivel,documento_id,citacao_id,inicio,fim,trecho,tipo,classificacao,id_canonico
    meta.jsonl              gênero de origem, matéria, split (treino | estresse), bancos usados
    relatorio.json          portões: descartes, discordâncias do resolver por fonte

Determinismo: tudo sai de `random.Random(semente + i)` por documento — o mesmo
comando gera os mesmos bytes.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import random

from gama.indice import Indice
from gama.ruido import InjetorDeRuido

from . import fichas as F
from .montar import Documento, Montador, mistura_do_dev
from .verificar import verificar


def carregar_bancos(caminhos: list[str]) -> dict:
    """Une bancos (o da organização + expansões por LLM)."""
    total: dict = collections.defaultdict(list)
    for c in caminhos:
        b = json.loads(pathlib.Path(c).read_text(encoding="utf-8"))
        for k, v in b.items():
            total[k].extend(v)
    return dict(total)


def split_de(doc_id: str) -> str:
    h = int(hashlib.sha1(doc_id.encode()).hexdigest(), 16)
    return "estresse" if h % 10 == 0 else "treino"


def gerar_um(m: Montador, i: int, semente: int, nivel: int, idx) -> tuple:
    rng = random.Random(semente * 1_000_003 + i)
    doc = m.montar(rng, nivel)
    if nivel == 2:
        inten = rng.uniform(0.45, 1.0)
        novo, remap = InjetorDeRuido(rng, inten).aplicar(doc.texto, [(a, b) for a, b, _ in doc.spans])
        doc = Documento(novo, [(na, nb, c) for (na, nb), (_, _, c) in zip(remap, doc.spans)],
                        {**doc.meta, "ruido": round(inten, 3)})
    return doc, verificar(doc.texto, doc.spans, idx, nivel)


def escrever(m: Montador, idx, n: int, semente: int, saida: pathlib.Path) -> dict:
    """Gera n documentos aprovados nos portões e grava no formato da organização."""
    (saida / "txt").mkdir(parents=True, exist_ok=True)
    linhas, metas = [], []
    descartes = collections.Counter()
    exemplos_fatais: list = []
    disc = collections.Counter()
    exemplos_disc = collections.defaultdict(list)
    por_fonte = collections.Counter()
    feitos = 0
    i = 0
    while feitos < n:
        nivel = 1 if feitos % 2 == 0 else 2
        doc, laudo = gerar_um(m, i, semente, nivel, idx)
        i += 1
        if laudo.fatal:
            descartes[laudo.fatal[0].split(" ")[0]] += 1
            if len(exemplos_fatais) < 15:
                exemplos_fatais.append(laudo.fatal[:2])
            continue
        doc_id = f"syn_n{nivel}_{feitos:05d}"
        (saida / "txt" / f"{doc_id}.txt").write_text(doc.texto, encoding="utf-8")
        for k, (ini_, fim, c) in enumerate(sorted(doc.spans, key=lambda s: s[0])):
            por_fonte[c.fonte] += 1
            linhas.append([nivel, doc_id, f"g{k + 1}", ini_, fim, doc.texto[ini_:fim], c.tipo,
                           c.classe, c.id_canonico or ""])
        for fonte, trecho, esperado, obtido in laudo.resolver:
            disc[fonte] += 1
            if len(exemplos_disc[fonte]) < 6:
                exemplos_disc[fonte].append((trecho, esperado, obtido))
        metas.append({"documento_id": doc_id, "split": split_de(doc_id), "semente": semente,
                      "i": i - 1, **doc.meta})
        feitos += 1

    with open(saida / "goldenset_offsets.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["nivel", "documento_id", "citacao_id", "inicio", "fim", "trecho", "tipo",
                    "classificacao", "id_canonico"])
        w.writerows(linhas)
    (saida / "meta.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in metas),
                                      encoding="utf-8")
    rel = {
        "documentos": feitos, "tentativas": i, "citacoes": len(linhas),
        "descartes": dict(descartes), "exemplos_descarte": exemplos_fatais,
        "resolver_discorda": {k: f"{disc[k]}/{por_fonte[k]}" for k in por_fonte},
        "exemplos_discordancia": exemplos_disc,
        "classes": dict(collections.Counter(r[7] for r in linhas)),
    }
    (saida / "relatorio.json").write_text(json.dumps(rel, ensure_ascii=False, indent=1), encoding="utf-8")
    return rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--semente", type=int, default=7)
    ap.add_argument("--saida", default="/app/corpus/goldenset/v0")
    ap.add_argument("--bancos", default="/app/corpus/bancos/org.json",
                    help="lista separada por vírgula")
    ap.add_argument("--db", default="/app/dados/desafio1_bracis.db")
    ap.add_argument("--dados", default="/app/dados")
    ap.add_argument("--fracao-llm", type=float, default=None,
                    help="sobrepõe montar.FRACAO_LLM em todos os bancos (estresse)")
    a = ap.parse_args()
    if a.fracao_llm is not None:
        from . import montar
        for k in montar.FRACAO_LLM:
            montar.FRACAO_LLM[k] = a.fracao_llm

    idx = Indice.do_banco(a.db)
    fichas, sumulas, disps = F.carregar(a.db)
    m = Montador(carregar_bancos(a.bancos.split(",")), fichas, sumulas, disps,
                 mistura_do_dev(pathlib.Path(a.dados)), idx)
    rel = escrever(m, idx, a.n, a.semente, pathlib.Path(a.saida))
    print(json.dumps({k: rel[k] for k in ("documentos", "tentativas", "citacoes", "descartes",
                                          "classes", "resolver_discorda")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
