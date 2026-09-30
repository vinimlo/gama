# -*- coding: utf-8 -*-
"""Cobertura do índice sobre as 996 fichas de acórdão — o ponto cego do dev set.

O dev set cita ~95 fichas; o conjunto cego vai citar outras. Uma ficha sem número
próprio, ou cujo número colide com outra ficha de mesma cadeia de classe, faz
qualquer citação a ela virar `inventada` ou link chutado. Isto mede isso sem
depender de nenhum extrator.
"""
from __future__ import annotations

import collections
import csv
import pathlib
import re
import sqlite3
import sys

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from gama.cabecalho import CadeiaDeClasse, LeitorDeNumeroProprio  # noqa: E402
from gama.normalizar import NumeroDeProcesso  # noqa: E402

DB = RAIZ / "dados" / "desafio1_bracis.db"
def nucleo(trecho: str) -> str:
    """Chave do número de uma citação (ver NumeroDeProcesso.do_trecho)."""
    return NumeroDeProcesso.do_trecho(trecho).chave


def construir():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute("SELECT id, tribunal, texto FROM documentos WHERE natureza='acordao'").fetchall()
    con.close()
    prop, por_chave = {}, collections.defaultdict(list)
    for doc_id, trib, texto in rows:
        p = LeitorDeNumeroProprio().ler(texto, trib)
        prop[doc_id] = (trib, p)
        for k in p.chaves:
            por_chave[k].append(doc_id)
    return rows, prop, por_chave


def main() -> int:
    rows, prop, por_chave = construir()
    tot = collections.Counter(t for _, t, _ in rows)
    sem = collections.Counter(t for d, (t, p) in prop.items() if not p.chaves)
    print("=== NUMERO PROPRIO POR TRIBUNAL ===")
    for t in sorted(tot):
        print(f"  {t}: {tot[t] - sem[t]}/{tot[t]}")
    for d, (t, p) in prop.items():
        if not p.chaves:
            print(f"   SEM [{t}] {d} ({p.fonte})")
    amb = {k: v for k, v in por_chave.items() if len(v) > 1}
    mesma_cadeia = {k: v for k, v in amb.items() if len({prop[d][1].cadeia for d in v}) < len(v)}
    print(f"\n=== AMBIGUIDADE ===\n  chaves compartilhadas: {len(amb)} | "
          f"separaveis pela cadeia: {len(amb) - len(mesma_cadeia)} | mesma cadeia: {len(mesma_cadeia)}")

    print("\n=== GABARITO: 'real' de acordao ===")
    stat = collections.Counter()
    for r in csv.DictReader(open(RAIZ / "dados" / "goldenset_offsets.csv", encoding="utf-8-sig")):
        if r["classificacao"] != "real" or r["tipo"] != "jurisprudencia":
            continue
        gid = int(r["id_canonico"])
        if gid not in prop:
            continue
        cands = por_chave.get(nucleo(r["trecho"]), [])
        if cands == [gid]:
            stat["unico_correto"] += 1
        elif gid in cands:
            cc = CadeiaDeClasse.ler(r["trecho"].replace("\\n", "\n"))
            filtr = [d for d in cands if prop[d][1].cadeia == cc]
            stat["amb_resolvido" if filtr == [gid] else "amb_nao_resolvido"] += 1
            if filtr != [gid]:
                print(f"   AMB {r['trecho']!r} cadeia={cc} -> {[(d, prop[d][1].cadeia) for d in cands]}")
        else:
            stat["nao_achou"] += 1
            print(f"   NAO ACHOU {r['trecho']!r} gid={gid} fonte={prop[gid][1].fonte} chaves={prop[gid][1].chaves} nucleo={nucleo(r['trecho'])}")
    print("  ", dict(stat))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
