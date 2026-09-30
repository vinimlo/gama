# -*- coding: utf-8 -*-
"""Escolha da configuração (só na validação interna) e AUROC refeito a partir dos arquivos gravados.

    escolher --subpastas lr_1e-05,lr_2e-05
        lê _modelo/<subpasta>/treino.json (baixado do repo do modelo) e escolhe a de maior AUROC nos
        candidatos das ementas da validação interna, medido no modelo salvo em bf16 (empate: menor
        perda; depois a ordem dada). Nada das 305, das 172 ou do estresse entra -> escolha.json
    auroc --subpasta S --scores scores/decision_1_0_eos_0_8b.jsonl
        AUROC refeito dos arquivos: validação interna (validacao_scores.jsonl do modelo, rótulos
        conferidos contra validacao.jsonl dos dados) e 305 / 172 / estresse (scores x candidatos.jsonl
        x ouro do harness, pela `nucleo.casa`), por origem e nos elegíveis A e B -> auroc.json

    docker compose run --rm gama python -m bench.controles.verificador_treinado.decision_1_0_eos_0_8b.analise <acao>
"""
from __future__ import annotations

import argparse
import json
import pathlib

from bench.controles import avaliar
from bench.controles.verificador import nucleo as vn

from . import job as J
from .politicas import SAIDA

DADOS = avaliar.SAIDA / "verificador_treinado" / "dados"


def cmd_escolher(a) -> None:
    linhas = []
    for i, s in enumerate(a.subpastas.split(",")):
        t = json.loads((SAIDA / "_modelo" / s / "treino.json").read_text(encoding="utf-8"))
        v = t["validacao_final_bf16"]
        linhas.append({"subpasta": s, "lr": t["hiperparametros"]["lr"], "melhor_passo": t["melhor_passo"],
                       "passos_feitos": t["passos_feitos"], "auroc_ementas": v["ementas"]["auroc"],
                       "perda": v["perda"], "auroc_todos": v["todos"]["auroc"],
                       "auroc_ementas_elegivel_A": v["ementas_elegivel_A"]["auroc"],
                       "auroc_final_v3": v["final_v3"]["auroc"], "ordem": i})
    melhor = max(linhas, key=lambda x: (x["auroc_ementas"], -x["perda"], -x["ordem"]))
    out = {"criterio": "maior AUROC nos candidatos das ementas da validação interna (modelo salvo em bf16); "
                       "empate: menor perda; depois a ordem dada. Só a validação interna dos dados de treino.",
           "tabela": linhas, "escolhida": melhor["subpasta"]}
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    (SAIDA / "escolha.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_auroc(a) -> None:
    out = {}
    # validação interna
    va = {r["cid"]: r for r in J.jsonl(DADOS / "validacao.jsonl")}
    vs = J.jsonl(SAIDA / "_modelo" / a.subpasta / "validacao_scores.jsonl")
    assert len(vs) == len(va) and all(va[r["cid"]]["rotulo"] == r["rotulo"] for r in vs), "validação não confere"
    regs = [va[r["cid"]] for r in vs]
    out["validacao_interna"] = J.metricas(regs, [r["score"] for r in vs])
    # 305, 172, estresse
    _, sc = vn.ler_scores(a.scores)
    _, docs = vn.ler_candidatos(vn.CANDIDATOS)
    info = {}
    for linha in pathlib.Path(vn.CANDIDATOS).read_text(encoding="utf-8").splitlines():
        r = json.loads(linha)
        if "meta" not in r:
            info[r["cid"]] = r
    for c in vn.CONJUNTOS:
        _, gold = avaliar.textos_e_ouro(c)
        pares = {x.cid: (sc[x.cid], int(vn.casa(x.span, gold[d], c != "estresse")))
                 for d, (G, R) in docs[c].items() for x in G + R}
        sub = {"todos": list(pares)}
        for o in ("gama", "regua"):
            sub[o] = [k for k in pares if info[k]["origem"] == o]
        sub["gama_forte"] = [k for k in pares if info[k]["forte"]]
        sub["elegivel_A"] = [k for k in pares if info[k]["elegivel"]["A"]]
        sub["elegivel_B"] = [k for k in pares if info[k]["elegivel"]["B"]]
        out[c] = {n: {"n": len(ks), "positivos": sum(pares[k][1] for k in ks),
                      "auroc": vn.auroc([pares[k] for k in ks])} for n, ks in sub.items()}
    out["scores"] = {"arquivo": a.scores, "sha256": vn.sha256(a.scores)}
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    (SAIDA / "auroc.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="acao", required=True)
    p = sub.add_parser("escolher")
    p.add_argument("--subpastas", required=True)
    p = sub.add_parser("auroc")
    p.add_argument("--subpasta", required=True)
    p.add_argument("--scores", default=str(SAIDA / "scores" / f"{J.NOME}.jsonl"))
    a = ap.parse_args()
    {"escolher": cmd_escolher, "auroc": cmd_auroc}[a.acao](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
