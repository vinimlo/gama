# -*- coding: utf-8 -*-
"""Convenções de borda do gabarito — onde a organização começa e termina um span.

Por que isto existe. O goldenset que vamos gerar treina o extrator neural. Se as
bordas dele divergirem das da organização (incluir o artigo "o" antes, deixar a UF
de fora, cortar "do STJ" da súmula), o modelo aprende a borda errada e perde IoU sem
erro visível. As 192 citações do gabarito são a única fonte de verdade sobre isso.

Para cada citação, olhamos o span E o contexto imediato no .txt — o que ficou fora
é tão informativo quanto o que ficou dentro.
"""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import re
import sys

RAIZ = pathlib.Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"
sys.path.insert(0, str(RAIZ / "src"))
from gama.formas import forma  # noqa: E402

UF = r"(?:AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)"
RE_UF_FINAL = re.compile(r"[/\-–—(]\s*" + UF + r"\s*\)?\s*$")
RE_MARCA_NUM = re.compile(r"\b[nN][ºo°.]\s|\bN[ºo°]|\bNo\b")
RE_TRIB_FINAL = re.compile(r"\bd[oa]\s+(?:STF|STJ|TST|TSE|STM)\s*$")


# Regras derivadas das 192 citações do gabarito (22/09). Valem nos dois sentidos:
# o gabarito não pode violar nenhuma (teste), e o goldenset gerado também não
# (verificador do gerador sintético) — senão o modelo aprende a borda errada.
ARTIGO_OU_PREPOSICAO = {"o", "a", "os", "as", "no", "na", "nos", "nas", "do", "da",
                        "dos", "das", "ao", "à", "pelo", "pela", "em", "de"}


def violacoes_de_borda(span: str) -> list:
    """Lista as regras de borda que o span viola (vazia = conforme)."""
    v = []
    if not span or span != span.strip():
        v.append("espaco_na_borda")                  # nunca começa/termina em espaço
    elif span[-1] in ".,;:":
        v.append("pontuacao_final_dentro")           # 0/192 no gabarito
    primeiro = re.findall(r"\S+", span)[:1]
    if primeiro and primeiro[0].lower() in ARTIGO_OU_PREPOSICAO:
        v.append("artigo_ou_preposicao_inicial")     # "o REsp" -> o "o" fica fora
    fm = forma(span)
    if fm == "artigo" and not span[:1].islower():
        v.append("artigo_comeca_maiusculo")          # 28/28 começam em "art"
    if fm == "processo" and not RE_UF_FINAL.search(span):
        v.append("processo_sem_uf")                  # 86/86 numeros classicos terminam com UF
    return v


def _palavras(s: str, n: int, do_fim: bool) -> str:
    toks = re.findall(r"\S+", s)
    toks = toks[-n:] if do_fim else toks[:n]
    return " ".join(toks)


def analisar() -> dict:
    rows = list(csv.DictReader(open(DADOS / "goldenset_offsets.csv", encoding="utf-8-sig")))
    textos = {}
    por_forma = collections.defaultdict(list)
    for r in rows:
        doc = r["documento_id"]
        if doc not in textos:
            textos[doc] = (DADOS / "txt" / f"{doc}.txt").read_text(encoding="utf-8")
        texto = textos[doc]
        i, f = int(r["inicio"]), int(r["fim"])
        span = texto[i:f]
        antes = texto[max(0, i - 40):i]
        depois = texto[f:f + 25]
        por_forma[forma(span)].append({
            "doc": doc, "nivel": r["nivel"], "classe": r["classificacao"],
            "span": span, "antes": antes, "depois": depois,
        })

    relatorio = {}
    for fm, itens in sorted(por_forma.items()):
        n = len(itens)
        c = collections.Counter()
        prim, ult, ant, dep = (collections.Counter() for _ in range(4))
        for it in itens:
            s = it["span"]
            c["com_quebra_de_linha"] += "\n" in s
            c["termina_com_uf"] += bool(RE_UF_FINAL.search(s))
            c["tem_marca_de_numero"] += bool(RE_MARCA_NUM.search(s))
            c["termina_com_tribunal"] += bool(RE_TRIB_FINAL.search(s))
            c["termina_em_pontuacao"] += s[-1] in ".,;:"
            c["comeca_com_minuscula"] += s[0].islower()
            prim[_palavras(s, 1, False)] += 1
            ult[_palavras(s, 1, True)] += 1
            ant[_palavras(it["antes"], 2, True).lower()] += 1
            dep[(it["depois"][:1] or "<fim>").replace("\n", "\\n")] += 1
        relatorio[fm] = {
            "n": n,
            "frequencias": {k: f"{v}/{n}" for k, v in sorted(c.items())},
            "primeira_palavra": prim.most_common(8),
            "ultima_palavra": ult.most_common(8),
            "duas_palavras_antes": ant.most_common(8),
            "primeiro_char_depois": dep.most_common(6),
            "exemplos": [{"antes": x["antes"][-28:], "span": x["span"],
                          "depois": x["depois"][:14]} for x in itens[:3]],
        }
    return relatorio


def main() -> int:
    rel = analisar()
    saida = RAIZ / "saidas" / "convencoes.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(rel, ensure_ascii=False, indent=2), encoding="utf-8")
    for fm, d in rel.items():
        print(f"\n=== {fm.upper()}  (n={d['n']}) ===")
        for k, v in d["frequencias"].items():
            print(f"   {k:24} {v}")
        print(f"   1a palavra : {d['primeira_palavra'][:6]}")
        print(f"   ult palavra: {d['ultima_palavra'][:6]}")
        print(f"   antes      : {d['duas_palavras_antes'][:6]}")
        print(f"   char depois: {d['primeiro_char_depois']}")
    print(f"\n-> {saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
