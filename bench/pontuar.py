# -*- coding: utf-8 -*-
"""Pontua os quatro extratores nos três conjuntos, com o mesmo resolver e a mesma métrica.

Todo extrator vira uma lista de spans por documento, gravada em `saidas/bench/`. A linha
`gama_guarda` é o Gama como a solução o roda: os mesmos spans, pela guarda de produção. Daí em
diante o caminho é o de produção para todos: `aparar`, `processar` (resolver + classe +
confiança) e a métrica oficial. Régua e Gama guardam o Span completo (forma, dígitos,
confiança); GLiNER e Qwen entram como o extrator neural entra, por `span_de`, sem
confiança de modelo (faixa "regra", os mesmos priores da régua).

Estresse e reais são extraídos no HF Jobs (`bench/extrair_*.py`, L4) e baixados para
`saidas/bench/`. O dev é da organização e não sai da máquina: nele só rodam régua e Gama,
aqui. Os extratores crus ficam sem dev.

    docker compose run --rm gama python -m bench.conjuntos
    docker compose run --rm gama python -m bench.pontuar extrair --extrator regua
    docker compose run --rm gama python -m bench.pontuar extrair --extrator gama
    docker compose run --rm gama python -m bench.pontuar          # -> bench/resultados.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import shutil
import time

from gama.extratores import carregar as carregar_extrator
from gama.extratores.guarda import guardar
from gama.formas import span_de
from gama.indice import construir
from gama.pipeline import SCHEMA, processar
from gama.span import Span, aparar_todos
from treino.avaliar_dobras import pontuar as pontuar_oficial

from . import conjuntos

SAIDA = conjuntos.SAIDA
RESULTADOS = pathlib.Path("/app/bench/resultados.json")
EXTRATORES = ["regua", "gliner", "qwen", "gama", "gama_guarda"]
CONJUNTOS = ["dev", "estresse", "reais"]
DB = "/app/dados/desafio1_bracis.db"


def _ler_spans(arquivo: pathlib.Path) -> tuple[dict, dict]:
    meta, linhas = {}, {}
    for x in arquivo.read_text(encoding="utf-8").splitlines():
        r = json.loads(x)
        if "meta" in r:
            meta = r["meta"]
        else:
            linhas[(r["conjunto"], r["id"])] = r
    return meta, linhas


def spans_de(extrator: str) -> tuple[dict, dict]:
    """(meta por origem, {(conjunto, id): linha}) juntando o arquivo local e o remoto."""
    metas, todas = {}, {}
    for arq in sorted(SAIDA.glob(f"{extrator}*.jsonl")):
        meta, linhas = _ler_spans(arq)
        metas[arq.stem] = meta
        todas.update(linhas)
    return metas, todas


def spans_com_guarda() -> tuple[dict, dict]:
    """O Gama como a solução o usa: os spans gravados do modelo e da régua passam pela guarda
    de produção (`src/gama/extratores/guarda.py`). Tempo = a soma dos dois."""
    mg, gama = spans_de("gama")
    mr, regua = spans_de("regua")
    textos = {}
    for nome in CONJUNTOS:
        textos.update({(nome, d): t for d, t in conjuntos.carregar(nome)[0].items()})
    linhas = {}
    for k, lg in gama.items():
        if k not in regua or k not in textos:
            continue
        t = textos[k]
        ss = guardar(aparar_todos([para_span(t, x) for x in lg["spans"]], t),
                     aparar_todos([para_span(t, x) for x in regua[k]["spans"]], t))
        linhas[k] = {"segundos": lg["segundos"] + regua[k]["segundos"],
                     "spans": [[s.inicio, s.fim, s.tipo, s.forma, s.digitos, s.confianca] for s in ss]}
    return {"gama": mg, "regua": mr}, linhas


def para_span(texto: str, s: list) -> Span:
    if len(s) == 6:                                   # Span completo (régua, Gama)
        a, b, tipo, forma, digitos, conf = s
        return Span(a, b, texto[a:b], tipo, forma, digitos, conf)
    return span_de(texto, s[0], s[1], s[2], None)     # (inicio, fim, rótulo[, score])


def rotulo(s: Span) -> str:
    return "LEI" if s.tipo == "lei" else ("VAGA" if s.forma == "vaga" else "JURIS")


def extracao(gold: dict, pred: dict, juntar_vaga: bool) -> dict:
    """F1 por tipo e micro, casamento 1-para-1 com o mesmo tipo e IoU >= 0,5."""
    def norm(t):
        return "JURIS" if juntar_vaga and t == "VAGA" else t
    tp, fp, fn = collections.Counter(), collections.Counter(), collections.Counter()
    for doc, gs in gold.items():
        ps = [(a, b, norm(t)) for a, b, t in pred.get(doc, [])]
        usados = set()
        for a, b, t in gs:
            t = norm(t)
            k = next((k for k, (pa, pb, pt) in enumerate(ps) if k not in usados and pt == t
                      and max(0, min(b, pb) - max(a, pa)) / (max(b, pb) - min(a, pa)) >= 0.5), None)
            if k is None:
                fn[t] += 1
            else:
                usados.add(k)
                tp[t] += 1
        for k, (_, _, pt) in enumerate(ps):
            if k not in usados:
                fp[pt] += 1

    def f1(t, f, n):
        return round(2 * t / max(1, 2 * t + f + n), 4)
    tipos = sorted(set(tp) | set(fp) | set(fn))
    return {"f1": f1(sum(tp.values()), sum(fp.values()), sum(fn.values())),
            "por_tipo": {t: {"f1": f1(tp[t], fp[t], fn[t]), "tp": tp[t], "fp": fp[t], "fn": fn[t]}
                         for t in tipos}}


class Fixo:
    """Extrator que devolve spans já calculados (o documento corrente é trocado por fora)."""
    nome = "fixo"

    def __init__(self):
        self.atual: list[Span] = []

    def extrair(self, texto: str) -> list[Span]:
        return self.atual


def oficial(nome: str, extrator: str, textos: dict, spans: dict, idx) -> dict:
    pasta = SAIDA / "json" / nome / extrator
    if pasta.exists():
        shutil.rmtree(pasta)
    pasta.mkdir(parents=True)
    fixo = Fixo()
    for d, texto in textos.items():
        fixo.atual = spans[d]
        cits = processar(texto, idx, fixo)
        doc = {"schema_version": SCHEMA, "documento_id": d,
               "citacoes": [c.para_json(i + 1) for i, c in enumerate(cits)]}
        (pasta / f"{d}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return pontuar_oficial(conjuntos.gabarito_oficial(nome), pasta)


def extrair_local(extrator: str, nomes: list[str]) -> int:
    """Roda régua ou Gama aqui (por padrão, só no dev) e grava os spans completos."""
    ext = carregar_extrator("regua" if extrator == "regua" else "neural-cru", "/models")
    meta = {"extrator": extrator, "dispositivo": "cpu (container, 4 vCPUs)"}
    for nome in nomes:
        textos, _ = conjuntos.carregar(nome)
        with open(SAIDA / f"{extrator}_{nome}.jsonl", "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"meta": meta}) + "\n")
            t_total = 0.0
            for d, texto in textos.items():
                t0 = time.perf_counter()
                ss = ext.extrair(texto)
                dt = time.perf_counter() - t0
                t_total += dt
                fh.write(json.dumps({"conjunto": nome, "id": d, "segundos": round(dt, 4), "spans": [
                    [s.inicio, s.fim, s.tipo, s.forma, s.digitos, s.confianca] for s in ss]},
                    ensure_ascii=False) + "\n")
        print(extrator, nome, len(textos), f"{t_total / max(1, len(textos)):.3f} s/doc", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", nargs="?", default="pontuar", choices=["pontuar", "extrair"])
    ap.add_argument("--extrator", choices=["regua", "gama"])
    ap.add_argument("--conjuntos", default="dev")
    a = ap.parse_args()
    if a.acao == "extrair":
        return extrair_local(a.extrator, a.conjuntos.split(","))

    idx = construir(DB)
    saida = {"meta": {}, "resultados": {}}
    for extrator in EXTRATORES:
        metas, linhas = spans_com_guarda() if extrator == "gama_guarda" else spans_de(extrator)
        saida["meta"][extrator] = metas
        res = saida["resultados"][extrator] = {}
        for nome in CONJUNTOS:
            textos, gold = conjuntos.carregar(nome)
            faltam = [d for d in textos if (nome, d) not in linhas]
            if faltam:
                print(f"{extrator} {nome}: faltam {len(faltam)} documentos, conjunto pulado", flush=True)
                continue
            spans = {d: [para_span(textos[d], s) for s in linhas[(nome, d)]["spans"]] for d in textos}
            pred = {d: [(s.inicio, s.fim, rotulo(s)) for s in aparar_todos(ss, textos[d])]
                    for d, ss in spans.items()}
            r = {"documentos": len(textos),
                 "s_por_doc": round(sum(linhas[(nome, d)]["segundos"] for d in textos) / len(textos), 4),
                 "extracao": extracao(gold, pred, juntar_vaga=nome == "reais")}
            if conjuntos.gabarito_oficial(nome):
                r["oficial"] = oficial(nome, extrator, textos, spans, idx)
            res[nome] = r
            print(extrator, nome, json.dumps({k: v for k, v in r.items() if k != "extracao"} |
                                             {"extracao_f1": r["extracao"]["f1"]}), flush=True)
    RESULTADOS.write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", RESULTADOS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
