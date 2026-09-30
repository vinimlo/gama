# -*- coding: utf-8 -*-
"""Quanto da política C do gliner_decide é decidido pelo desempate, não pelo score.

Os scores do fine-tune saturam em exatamente 1,0 (float32) na maior parte dos candidatos. Na C, entre
dois candidatos sobrepostos com o mesmo score, vence o do Gama (`nucleo.politica`: -score, depois
Gama antes da régua). Aqui a C roda de novo, no mesmo tau do relatório, com o desempate invertido só
nos empates exatos (régua antes do Gama), e conta documentos mudados contra a produção e o F1. Também
conta os pares Gama x régua sobrepostos com score igual. Só leitura dos arquivos gravados.

    docker compose run --rm gama python -m bench.controles.verificador_treinado.gliner_decide.desempate \\
        --variante ft --scores <gliner_decide_ft.jsonl>
"""
from __future__ import annotations

import argparse
import json

from bench.controles import avaliar
from bench.controles.verificador import nucleo as vn

from .__main__ import SAIDA, _escrever


def politica_c(G: list, R: list, score: dict, tau: float, regua_primeiro: bool) -> list:
    ok = sorted((c for c in G + R if score[c.cid] >= tau),
                key=lambda c: (-score[c.cid], (c.origem == "gama") if regua_primeiro else (c.origem != "gama"),
                               c.span.inicio, c.span.fim))
    escolhidos = []
    for c in ok:
        if not any(vn._cruza(c.span, o) for o in escolhidos):
            escolhidos.append(c.span)
    return sorted(escolhidos, key=lambda s: s.inicio)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variante", required=True, choices=("zs", "ft"))
    ap.add_argument("--scores", required=True)
    a = ap.parse_args()
    rel = json.loads((SAIDA / f"gliner_decide_{a.variante}.json").read_text(encoding="utf-8"))
    tau = rel["politicas"]["C"]["tau"]
    _, docs = vn.ler_candidatos(vn.CANDIDATOS)
    _, sc = vn.ler_scores(a.scores)
    out = {"tau_C": tau, "scores": {"arquivo": a.scores, "sha256": vn.sha256(a.scores)}, "conjuntos": {}}
    for c in vn.CONJUNTOS:
        prod = vn.producao(docs[c])
        cands = [x for G, R in docs[c].values() for x in G + R]
        pares = empates = 0
        for G, R in docs[c].values():
            for g in G:
                for r in R:
                    if vn._cruza(g.span, r.span) and sc[g.cid] >= tau and sc[r.cid] >= tau:
                        pares += 1
                        empates += sc[g.cid] == sc[r.cid]
        linha = {"candidatos": len(cands), "score_igual_a_1": sum(sc[x.cid] == 1.0 for x in cands),
                 "pares_gama_regua_sobrepostos_acima_de_tau": pares, "desses_com_score_igual": empates}
        for nome, inv in (("gama_primeiro", False), ("regua_primeiro", True)):
            finais = {d: politica_c(G, R, sc, tau, inv) for d, (G, R) in docs[c].items()}
            linha[nome] = {"docs_mudados_vs_producao": sum(vn._chave(finais[d]) != vn._chave(prod[d]) for d in finais),
                           "f1": avaliar.extracao(c, finais)["f1"]}
        out["conjuntos"][c] = linha
        print(c, json.dumps(linha, ensure_ascii=False), flush=True)
    rel_c = rel["politicas"]["C"]["conjuntos"]
    out["confere_com_relatorio"] = all(
        out["conjuntos"][c]["gama_primeiro"]["docs_mudados_vs_producao"] == rel_c[c]["docs_mudados_vs_producao"]
        and out["conjuntos"][c]["gama_primeiro"]["f1"] == rel_c[c]["extracao"]["f1"] for c in vn.CONJUNTOS)
    print("confere com o relatório:", out["confere_com_relatorio"], flush=True)
    _escrever(SAIDA / f"desempate_{a.variante}.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
