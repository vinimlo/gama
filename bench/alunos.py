# -*- coding: utf-8 -*-
"""Régua de zero perda para os alunos destilados: o aluno só passa se a solução com ele
entregar o mesmo que a solução com o professor.

    extrair-dev  roda um modelo no dev, em CPU, aqui (o dev não sai da máquina)
    comparar     aluno x professor pelo caminho de produção inteiro (extrator, guarda,
                 resolver, confiança calibrada): JSON final documento a documento no dev,
                 no estresse e no final_v1; F1 de extração com a guarda nas 305 ementas do
                 ouro antigo e nas 200 do teste novo, com bootstrap pareado; tempo e tamanho

    docker compose run --rm gama python -m bench.alunos extrair-dev --nome a --pasta /app/corpus/modelos/a
    docker compose run --rm gama python -m bench.alunos comparar --alunos a,b,b_controle

Spans dos conjuntos remotos em `saidas/bench/modelo_<nome>.jsonl` (`bench/extrair_alunos.py`).
Resultado em `bench/alunos.json`.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import time

from gama.extratores.guarda import Guarda
from gama.extratores.regua import ExtratorRegua
from gama.indice import Indice
from gama.pipeline import Pipeline
from gama.span import Span
from treino.avaliar_dobras import pontuar as pontuar_oficial

from . import conjuntos
from .pontuar import DB, SAIDA, Fixo, extracao, para_span, rotulo

RESULTADOS = pathlib.Path("/app/bench/alunos.json")
NOVO_OURO = pathlib.Path("/app/corpus/reais/novo/ouro_real.jsonl")
V1 = pathlib.Path("/app/corpus/goldenset/v1")


def ler(nome: str) -> tuple[dict, dict]:
    meta, linhas = {}, {}
    for arq in (SAIDA / f"modelo_{nome}.jsonl", SAIDA / f"modelo_{nome}_dev.jsonl"):
        if not arq.exists():
            continue
        for x in arq.read_text(encoding="utf-8").splitlines():
            r = json.loads(x)
            if "meta" in r:
                meta[arq.stem] = r["meta"]
            else:
                linhas[(r["conjunto"], r["id"])] = r
    return meta, linhas


def extrair_dev(nome: str, pasta: str) -> None:
    from gama.extratores.neural import ExtratorNeural
    ext = ExtratorNeural(pasta)
    textos, _ = conjuntos.carregar("dev")
    with open(SAIDA / f"modelo_{nome}_dev.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": {"extrator": nome, "pasta": pasta, "dispositivo": "cpu (container, 4 vCPUs)"}}) + "\n")
        total = 0.0
        for d, t in textos.items():
            t0 = time.perf_counter()
            ss = ext.extrair(t)
            dt = time.perf_counter() - t0
            total += dt
            fh.write(json.dumps({"conjunto": "dev", "id": d, "segundos": round(dt, 4), "spans": [
                [s.inicio, s.fim, s.tipo, s.forma, s.digitos, s.confianca] for s in ss]}, ensure_ascii=False) + "\n")
    print(nome, "dev", f"{total / len(textos):.3f} s/doc em CPU")


def _textos_e_ouro(conjunto: str) -> tuple[dict, dict]:
    if conjunto in ("dev", "estresse", "reais"):
        return conjuntos.carregar(conjunto)
    if conjunto == "v1":
        return conjuntos._com_gabarito_csv(V1)
    if conjunto == "novas":
        ouro = {r["id"]: [tuple(s) for s in r["spans"]] for r in map(json.loads, NOVO_OURO.open(encoding="utf-8"))}
        textos = {r["id"]: r["texto"] for r in map(json.loads, open("/app/corpus/reais/amostra.jsonl", encoding="utf-8"))
                  if r["id"] in ouro}
        return {k: textos[k] for k in sorted(ouro)}, ouro
    raise ValueError(conjunto)


def _gabarito_oficial(conjunto: str):
    return V1 / "goldenset_offsets.csv" if conjunto == "v1" else conjuntos.gabarito_oficial(conjunto)


def solucao(textos: dict, linhas: dict, conjunto: str, regua, idx) -> tuple[dict, dict]:
    """(spans finais depois da guarda, JSON final) por documento, como a produção entrega."""
    spans, jsons = {}, {}
    fixo = Fixo()
    for d, t in textos.items():
        modelo = Span.aparar_todos([para_span(t, s) for s in linhas[(conjunto, d)]["spans"]], t)
        spans[d] = Guarda().aplicar(modelo, Span.aparar_todos(regua.extrair(t), t))
        fixo.atual = spans[d]
        jsons[d] = [c.para_json(i + 1) for i, c in enumerate(Pipeline(idx, fixo).processar(t))]
    return spans, jsons


def _oficial(conjunto: str, nome: str, jsons: dict) -> float:
    pasta = SAIDA / "json" / f"alunos_{conjunto}" / nome
    pasta.mkdir(parents=True, exist_ok=True)
    for d, cits in jsons.items():
        (pasta / f"{d}.json").write_text(json.dumps({"schema_version": "1.2", "documento_id": d, "citacoes": cits},
                                                    ensure_ascii=False), encoding="utf-8")
    return pontuar_oficial(_gabarito_oficial(conjunto), pasta)["final"]


def _por_doc(gold: dict, pred: dict) -> dict:
    return {d: extracao({d: gold[d]}, {d: pred.get(d, [])}, True)["por_tipo"] for d in gold}


def _f1(contagens: dict, docs: list) -> float:
    tp = fp = fn = 0
    for d in docs:
        for v in contagens[d].values():
            tp, fp, fn = tp + v["tp"], fp + v["fp"], fn + v["fn"]
    return 2 * tp / max(1, 2 * tp + fp + fn)


def comparar(alunos: list[str]) -> dict:
    idx = Indice.do_banco(DB)
    regua = ExtratorRegua()
    modelos = {n: ler(n) for n in ["professor", *alunos]}
    out = {}
    base = {}
    for conjunto in ("dev", "estresse", "v1", "reais", "novas"):
        if conjunto == "novas" and not NOVO_OURO.exists():
            continue
        textos, gold = _textos_e_ouro(conjunto)
        for nome, (meta, linhas) in modelos.items():
            if any((conjunto, d) not in linhas for d in textos):
                continue
            spans, jsons = solucao(textos, linhas, conjunto, regua, idx)
            r = {}
            if conjunto in ("dev", "estresse", "v1"):
                r["oficial"] = _oficial(conjunto, nome, jsons)
                if nome != "professor":
                    ref_spans, ref_json = base[conjunto]
                    r["docs_json_diferente"] = sum(jsons[d] != ref_json[d] for d in textos)
                    r["docs_spans_diferentes"] = sum(
                        [(s.inicio, s.fim, s.forma) for s in spans[d]] != [(s.inicio, s.fim, s.forma) for s in ref_spans[d]]
                        for d in textos)
                else:
                    base[conjunto] = (spans, jsons)
            else:
                pred = {d: [(s.inicio, s.fim, rotulo(s)) for s in ss] for d, ss in spans.items()}
                cont = _por_doc(gold, pred)
                r["f1_com_guarda"] = round(_f1(cont, list(gold)), 4)
                if nome == "professor":
                    base[conjunto] = cont
                else:
                    docs = sorted(gold)
                    rng = random.Random(0)
                    difs = sorted(_f1(cont, s) - _f1(base[conjunto], s)
                                  for s in ([rng.choice(docs) for _ in docs] for _ in range(2000)))
                    r["dif_para_professor"] = round(_f1(cont, docs) - _f1(base[conjunto], docs), 4)
                    r["ic95"] = [round(difs[50], 4), round(difs[1949], 4)]
            segs = [linhas[(conjunto, d)]["segundos"] for d in textos]
            r["s_por_doc"] = round(sum(segs) / len(segs), 4)
            out.setdefault(nome, {})[conjunto] = r
            print(nome, conjunto, json.dumps(r), flush=True)
    for nome, (meta, _) in modelos.items():
        out.setdefault(nome, {})["meta"] = meta
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=["extrair-dev", "comparar"])
    ap.add_argument("--nome")
    ap.add_argument("--pasta")
    ap.add_argument("--alunos", default="a,b,b_controle")
    a = ap.parse_args()
    if a.acao == "extrair-dev":
        extrair_dev(a.nome, a.pasta)
        return 0
    res = comparar(a.alunos.split(","))
    RESULTADOS.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", RESULTADOS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
