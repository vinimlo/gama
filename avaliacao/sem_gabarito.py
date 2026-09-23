# -*- coding: utf-8 -*-
"""Ferramenta do dia do conjunto cego: onde procurar erro SEM gabarito.

Compara duas saídas (em geral: neural e régua) sobre os mesmos textos e lista, por
prioridade de inspeção:

1. DIVERGÊNCIA de extração — span que só um dos extratores achou. A melhor pista que
   existe sem gabarito: onde os dois concordam, o erro é raro.
2. NÚMERO SOLTO — sequência de 3+ dígitos fora de qualquer span, fora do cabeçalho e
   que não é data, fls., OAB ou valor: citação que pode ter escapado.
3. RELATOR SEM VAGA — "Rel.", "relatoria" fora de qualquer span: vaga não detectada.
4. BAIXA CONFIANÇA — spans com confiança < limiar.
5. DISTRIBUIÇÃO — classes por documento vs. o dev (≈ 7,4 citações/doc, 3:2:1).

    python -m avaliacao.sem_gabarito --txt IN --a saidas/json_neural --b saidas/json_regua \\
        --saida saidas/inspecao.md
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re

_NUM = re.compile(r"\d[\d.\-/]{2,}\d")
_NAO_CITACAO = re.compile(
    r"(?:fls?\.\s*|OAB/[A-Z]{2}\s*|R\$\s*|\bde\s+\w+\s+de\s+|nº\s*\d{4}\.\d)", re.I)
_RELATOR = re.compile(r"\bRel\.|\brelatoria\b|\bRelator[a]?\b(?!\s*:)", re.I)


def _spans(pasta: pathlib.Path) -> dict:
    out = {}
    for arq in sorted(pasta.glob("*.json")):
        d = json.loads(arq.read_text(encoding="utf-8"))
        out[d["documento_id"]] = d["citacoes"]
    return out


def _cruza(c, lista) -> bool:
    return any(c["inicio"] < o["fim"] and o["inicio"] < c["fim"] for o in lista)


def _fim_cabecalho(texto: str) -> int:
    i = texto.find("\n\n\n")
    return i if i > 0 else min(len(texto), 600)


def inspecionar(txt: pathlib.Path, a: dict, b: dict, limiar: float) -> dict:
    rel = collections.defaultdict(list)
    classes = collections.Counter()
    for doc, cits in sorted(a.items()):
        texto = (txt / f"{doc}.txt").read_text(encoding="utf-8")
        outro = b.get(doc, [])
        for c in cits:
            classes[c["classificacao"]] += 1
            if not _cruza(c, outro):
                rel["so_A"].append((doc, c["trecho"], c["classificacao"], c.get("confianca")))
            if c.get("confianca") is not None and c["confianca"] < limiar:
                rel["baixa_confianca"].append((doc, c["trecho"], c["classificacao"], c["confianca"]))
        for c in outro:
            if not _cruza(c, cits):
                rel["so_B"].append((doc, c["trecho"], c["classificacao"], c.get("confianca")))
        corte = _fim_cabecalho(texto)
        todos = cits + outro
        for m in _NUM.finditer(texto):
            if m.start() < corte or re.fullmatch(r"(?:19|20)\d{2}", m.group()):
                continue
            antes = texto[max(0, m.start() - 12):m.start()]
            if _NAO_CITACAO.search(antes + m.group()):
                continue
            if not any(c["inicio"] <= m.start() < c["fim"] for c in todos):
                rel["numero_solto"].append((doc, texto[max(0, m.start() - 40):m.end() + 10].replace("\n", " ")))
        for m in _RELATOR.finditer(texto):
            if m.start() < corte:
                continue
            if not any(c["inicio"] <= m.start() < c["fim"] for c in todos):
                rel["relator_sem_vaga"].append((doc, texto[max(0, m.start() - 60):m.end() + 30].replace("\n", " ")))
    n = max(1, len(a))
    rel["distribuicao"] = [(k, v, round(v / n, 2)) for k, v in classes.most_common()]
    return rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", required=True)
    ap.add_argument("--a", required=True, help="saída principal (ex.: neural)")
    ap.add_argument("--b", required=True, help="saída de comparação (ex.: régua)")
    ap.add_argument("--limiar", type=float, default=0.6)
    ap.add_argument("--saida", default=None)
    x = ap.parse_args()
    rel = inspecionar(pathlib.Path(x.txt), _spans(pathlib.Path(x.a)), _spans(pathlib.Path(x.b)), x.limiar)
    linhas = ["# Inspeção sem gabarito", ""]
    titulos = {"so_A": "Só na saída A", "so_B": "Só na saída B", "numero_solto": "Número solto",
               "relator_sem_vaga": "Relator sem vaga", "baixa_confianca": "Baixa confiança",
               "distribuicao": "Distribuição (classe, total, por documento)"}
    for k, t in titulos.items():
        itens = rel.get(k, [])
        linhas += [f"## {t} ({len(itens)})", ""] + [f"- {i}" for i in itens[:200]] + [""]
    texto = "\n".join(linhas)
    if x.saida:
        pathlib.Path(x.saida).write_text(texto, encoding="utf-8")
    print(texto[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
