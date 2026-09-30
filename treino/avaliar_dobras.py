# -*- coding: utf-8 -*-
"""Validação por dobras: cada modelo de dobra avaliado nos documentos da org que ele NUNCA viu.

Para a dobra d: os documentos da org da dobra d (gabarito filtrado em
corpus/dobras/dobra{d}/) passam pelo pipeline inteiro — régua e cada variante
treinada sem eles — e são pontuados pela métrica OFICIAL. Os pesos descem um de
cada vez e são apagados depois (cada um tem ~1,2 GB).

    docker compose run --rm lab python -m treino.avaliar_dobras
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

from avaliacao.harness import avaliar, montar_solution, montar_submission
from gama.indice import Indice
from gama.pipeline import Documento, Pipeline
from gama.extratores import CatalogoDeExtratores

SCHEMA = "1.2"


def rodar(extrator, idx, docs: list[str], dados: pathlib.Path, saida: pathlib.Path) -> None:
    if saida.exists():
        shutil.rmtree(saida)
    saida.mkdir(parents=True)
    for d in docs:
        texto = Documento.ler(dados / "txt" / f"{d}.txt").texto
        cits = Pipeline(idx, extrator).processar(texto)
        doc = {"schema_version": SCHEMA, "documento_id": d,
               "citacoes": [c.para_json(i + 1) for i, c in enumerate(cits)]}
        (saida / f"{d}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def pontuar(gold: pathlib.Path, pred: pathlib.Path) -> dict:
    sol = montar_solution(gold)
    res = avaliar(sol, montar_submission(pred, sol["documento_id"].tolist()))
    return {"final": round(res["score_final"], 5),
            **{f"n{n}": {"score": round(v["score"], 4), "f1": {k: round(x, 3) for k, x in v["f1_por_classe"].items()},
                         "tau": round(v["tau"], 3)} for n, v in res["niveis"].items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default="/app/dados")
    ap.add_argument("--dobras", default="/app/corpus/dobras")
    ap.add_argument("--repo", default="vinimlo/gama-d{d}-{v}")
    ap.add_argument("--variantes", default="org,llm")
    ap.add_argument("--dobra", default="0,1", help="quais dobras avaliar")
    ap.add_argument("--nome", default="mmbert", help="prefixo das chaves no resultado")
    ap.add_argument("--sem-regua", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download

    dados, dobras = pathlib.Path(a.dados), pathlib.Path(a.dobras)
    idx = Indice.do_banco(dados / "desafio1_bracis.db")
    tabela = {}
    for d in (int(x) for x in a.dobra.split(",")):
        pasta = dobras / f"dobra{d}"
        docs = pasta.joinpath("docs.txt").read_text().split()
        gold = pasta / "gabarito.csv"
        if not a.sem_regua:
            rodar(CatalogoDeExtratores().carregar("regua"), idx, docs, dados, pasta / "pred_regua")
            tabela[f"d{d}:regua"] = pontuar(gold, pasta / "pred_regua")
            print(f"d{d} regua", tabela[f"d{d}:regua"], flush=True)
        for v in a.variantes.split(","):
            repo = a.repo.format(d=d, v=v)
            local = pathlib.Path("/app/corpus/modelos/tmp")
            if local.exists():
                shutil.rmtree(local)
            snapshot_download(repo, local_dir=str(local))
            for modo in ("neural", "uniao"):
                chave = f"{a.nome}:d{d}:{v}:{modo}"
                rodar(CatalogoDeExtratores(str(local)).carregar(modo), idx, docs, dados, pasta / f"pred_{v}_{modo}")
                tabela[chave] = pontuar(gold, pasta / f"pred_{v}_{modo}")
                print(chave, tabela[chave], flush=True)
            shutil.rmtree(local)
    arq = dobras / "resultado.json"
    anterior = json.loads(arq.read_text()) if arq.exists() else {}
    arq.write_text(json.dumps({**anterior, **tabela}, ensure_ascii=False, indent=1))
    print("\nRESUMO (score final, métrica oficial, docs nunca vistos):")
    for k, v in tabela.items():
        print(f"  {k:12s} {v['final']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
