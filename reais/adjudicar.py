# -*- coding: utf-8 -*-
"""Adjudicação por critério escrito e medida de robustez em texto real.

Entrada: `corpus/reais/prata.jsonl` (reais.anotar). Saída:
  - precisão/recall de Gama e régua contra o rótulo PRATA (os dois LLMs concordam),
    por tipo, com IoU >= 0,5 — a robustez fora da distribuição;
  - divergências entre os LLMs resolvidas pelos critérios de `wiki/conceitos/convencoes-de-borda.md`
    quando o critério decide sozinho;
  - `corpus/reais/ambiguos.jsonl`: só o que o critério escrito NÃO decide — isso, e só
    isso, vai para decisão humana, tomada por padrão e não item a item.

Critérios (na ordem):
  1. span sem dígito e sem relator/ano -> não é citação (referência genérica);
  2. span com "Rel." / "relator" DEPOIS de um número -> a borda certa para no número (UF);
  3. tipo: "art."/"artigo" no início -> LEI; relator + ano sem número -> VAGA; resto -> JURIS;
  4. sobrou divergência de borda com IoU >= 0,5 entre as versões -> fica a mais curta que
     ainda contém classe + número (convenção do gabarito);
  5. o resto é ambíguo.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re


def _iou(a, b) -> float:
    i = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return i / (max(a[1], b[1]) - min(a[0], b[0])) if i else 0.0


def _tipo(trecho: str) -> str:
    t = trecho.strip().lower()
    if re.search(r"\barts?\b\.?\s*\d|\bartigos?\s*\d", t):
        return "LEI"
    sem_ano = re.sub(r"\b(19|20)\d{2}\b", "", t)
    if re.search(r"rel(\.|ator|atoria)", t) and not re.search(r"\d", sem_ano):
        return "VAGA"
    return "JURIS"


def criterio(texto: str, span) -> tuple | None:
    """Aplica os critérios 1-3 a um span candidato. None = não é citação."""
    a, b, _ = span
    trecho = texto[a:b]
    sem_ano = re.sub(r"\b(19|20)\d{2}\b", "", trecho)
    if not re.search(r"\d", trecho) and not re.search(r"rel(\.|ator|atoria)", trecho, re.I):
        return None                                                    # 1
    m = re.search(r"\d[\d.\-/]*\s*[-/]?\s*[A-Z]{2}\b", trecho)
    rel = re.search(r",?\s*rel(\.|ator)", trecho, re.I)
    if m and rel and rel.start() >= m.end():                           # 2
        b = a + m.end()
    # 3: tipo LEI vindo do anotador é mantido ("arts. 543-A e 543-B do CPC"); só JURIS/VAGA
    # passam pelo critério de tipo (revisão independente, rodada 3, achado 4)
    return (a, b, "LEI" if span[2] == "LEI" else _tipo(texto[a:b]))


_DIPLOMA = re.compile(r"(?i)\b(cpc|cpp|clt|cf|cdc|ristf|ristj|c[oó]digo|lei|constitui|regimento|estatuto)")
_NUMERO = re.compile(r"\d[\d.]*")


def preferencia(candidatos: list, texto: str):
    """Critérios de convenção (wiki/conceitos/convencoes-de-borda.md) quando os votos divergem só
    na borda. Devolve a lista escolhida (um ou mais spans) ou None se nada decide.
      - LEI: a versão que inclui o nome do diploma ("RISTF, art. 131" > "art. 131");
      - Tema: a versão com o qualificador ("Tema 660 da ... RG" > "Tema 660");
      - lista de súmulas: um span por número (o dev cita uma súmula por span).
    """
    trechos = [texto[a:b] for a, b, _ in candidatos]
    # Convenção votada, padrão 1: lista com classe compartilhada ("Rcls 36.958 e
    # 40.652", "CPC, artigos 219, 224 e 1.003") = UMA citação por número.
    if any(len(_NUMERO.findall(tr)) >= 2 and re.search(r"(?i)\s(e|,)\s", tr) for tr in trechos):
        simples = [c for c, tr in zip(candidatos, trechos) if len(_NUMERO.findall(tr)) == 1]
        if simples:
            return simples
    if all(t == "LEI" for _, _, t in candidatos):
        com = [c for c, tr in zip(candidatos, trechos) if _DIPLOMA.search(tr)]
        if com:
            return [max(com, key=lambda c: c[1] - c[0])]
    if any(re.match(r"(?i)\s*te(m|rn)a", tr) for tr in trechos):
        com = [c for c, tr in zip(candidatos, trechos) if re.search(r"(?i)repercuss|\brg\b", tr)]
        if com:
            return [max(com, key=lambda c: c[1] - c[0])]
    if any(re.search(r"(?i)s[úu]mulas", tr) for tr in trechos):
        simples = [c for c, tr in zip(candidatos, trechos) if len(_NUMERO.findall(tr)) == 1]
        if simples:
            return simples
    # Convenções votadas, padrões 2 e 3: qualificador entre classe e número entra
    # ("Recurso Especial repetitivo 1.495.146/MG"); nome por extenso + sigla começa no
    # extenso ("Arguição de Descumprimento de Preceito Fundamental - ADPF nº 130/DF").
    # Regra: envelope das versões (do início da classe ao número/UF), com "Rel." cortado.
    if all(t == "JURIS" for _, _, t in candidatos):
        env = (min(a for a, _, _ in candidatos), max(b for _, b, _ in candidatos), "JURIS")
        r = criterio(texto, env)
        if r and _NUMERO.search(texto[r[0]:r[1]]):
            return [r]
    return None


_VOTOS: dict = {}


def _terceiro(i: str, texto: str, modelo: str):
    """Anotação do terceiro LLM (mesmo prompt e alinhamento exato de reais.anotar)."""
    if i not in _VOTOS:
        from geracao.expandir import _json
        from geracao.llm import chat
        from .anotar import PROMPT, alinhar
        try:
            marcado = _json(chat(modelo, [{"role": "user", "content": PROMPT.format(texto=texto)}],
                                 temperatura=0.0, semente=0)).get("texto", "")
            _VOTOS[i] = alinhar(texto, marcado) if isinstance(marcado, str) else None
        except RuntimeError:
            _VOTOS[i] = None
    return _VOTOS[i]


def metricas(itens: list, fonte: str, textos: dict) -> dict:
    tp, fp, fn = collections.Counter(), collections.Counter(), collections.Counter()
    for it in itens:
        prata = [tuple(s) for s in it["prata"]]
        preds = [(s[0], s[1], "LEI" if s[2] == "artigo" else "VAGA" if s[2] == "vaga" else "JURIS")
                 for s in it[fonte]]
        usados = set()
        for g in prata:
            k = next((k for k, p in enumerate(preds) if k not in usados and p[2] == g[2] and _iou(g, p) >= 0.5), None)
            if k is None:
                fn[g[2]] += 1
            else:
                usados.add(k); tp[g[2]] += 1
        for k, p in enumerate(preds):
            if k not in usados and not any(_iou(p, d) >= 0.5 for d in map(tuple, it["divergente_llm"])):
                fp[p[2]] += 1
    out = {}
    for t in ("JURIS", "LEI", "VAGA"):
        p = tp[t] / max(1, tp[t] + fp[t]); r = tp[t] / max(1, tp[t] + fn[t])
        out[t] = {"precisao": round(p, 3), "recall": round(r, 3), "n": tp[t] + fn[t]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prata", default="/app/corpus/reais/prata.jsonl")
    ap.add_argument("--amostra", default="/app/corpus/reais/amostra.jsonl")
    ap.add_argument("--ambiguos", default="/app/corpus/reais/ambiguos.jsonl")
    ap.add_argument("--terceiro", default="glm", help="LLM de desempate (voto 2 de 3); '' desliga")
    a = ap.parse_args()
    textos = {r["id"]: r["texto"] for r in map(json.loads, open(a.amostra, encoding="utf-8"))}
    itens = [json.loads(x) for x in open(a.prata, encoding="utf-8")]
    decididos, ambiguos = 0, []
    for it in itens:
        texto = textos[it["id"]]
        resolvidos = []
        for s in it["divergente_llm"]:
            r = criterio(texto, s)
            if r is None:
                decididos += 1
                continue
            if any(_iou(r, p) >= 0.5 for p in map(tuple, it["prata"])):
                decididos += 1
                continue
            conflito = [o for o in it["divergente_llm"] if o != s and _iou(o, s) > 0]
            if conflito:
                escolha = preferencia([tuple(s)] + [tuple(o) for o in conflito], texto)
                if escolha:
                    resolvidos.extend(escolha)
                    decididos += 1
                    continue
                # Terceiro voto: o span que o terceiro LLM confirma vence;
                # só o empate de três vias vai para decisão humana.
                voto = _terceiro(it["id"], texto, a.terceiro) if a.terceiro else None
                if voto is not None:
                    candidatos = [s] + conflito
                    ganha = [c for c in candidatos if any(_iou(c, v) >= 0.8 and c[2] == v[2] for v in voto)]
                    if len(ganha) == 1:
                        resolvidos.append(tuple(ganha[0]))
                        decididos += 1
                        continue
                    if not ganha and not any(_iou(c, v) > 0 for c in candidatos for v in voto):
                        decididos += 1          # terceiro diz que não é citação: fica fora
                        continue
                ambiguos.append({"id": it["id"], "trecho": texto[s[0]:s[1]], "tipo": s[2],
                                 "contexto": texto[max(0, s[0] - 80):s[1] + 80]})
            else:
                resolvidos.append(r)
                decididos += 1
        it["prata"] = sorted(set(map(tuple, it["prata"])) | set(resolvidos))
        # Só o que segue ambíguo desculpa previsão nas métricas; divergência já decidida
        # conta como qualquer rótulo (revisão independente, rodada 3, achado 6).
        restantes = {x["trecho"] for x in ambiguos if x["id"] == it["id"]}
        it["divergente_llm"] = [s for s in it["divergente_llm"] if texto[s[0]:s[1]] in restantes]
    with open(a.ambiguos, "w", encoding="utf-8") as fh:
        for x in ambiguos:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    # Conjunto real adjudicado: prata (2 LLMs concordam) + divergências decididas por
    # critério escrito e pelas convenções votadas. Serve para medir qualquer versão futura.
    with open(pathlib.Path(a.prata).with_name("ouro_real.jsonl"), "w", encoding="utf-8") as fh:
        for it in itens:
            fh.write(json.dumps({"id": it["id"], "spans": it["prata"],
                                 "proveniencia": it["proveniencia"]}, ensure_ascii=False) + "\n")
    rel = {"ementas": len(itens), "divergencias_llm_decididas_por_criterio": decididos,
           "ambiguas_para_o_vinicius": len(ambiguos),
           "gama": metricas(itens, "gama", textos), "regua": metricas(itens, "regua", textos)}
    print(json.dumps(rel, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
