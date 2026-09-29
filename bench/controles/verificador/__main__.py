# -*- coding: utf-8 -*-
"""CLI do verificador de candidatos. Tudo no container:

    docker compose run --rm gama python -m bench.controles.verificador <acao> [...]

    candidatos                      grava saidas/bench/controles/verificador/candidatos.jsonl (Gama v1.3 +
                                    régua, aparados, com o estado para o modelo de decisão) e confere que
                                    relidos eles reproduzem os spans e a guarda de produção do harness
    oraculo                         scores/oraculo.jsonl: 1 se o candidato casa com o ouro, senão 0 (teto)
    de-spans --nome N --spans A     scores/N.jsonl a partir de spans de outro verificador no formato do
                                    harness: score = maior score de um span previsto do mesmo tipo com
                                    IoU >= 0,5, senão 0 (ex.: o GLiNER fine-tunado)
    escolher --bruto B              AUROC de cada formulação do Laya nas 305 (só nelas) -> escolha.json
    rodar --nome N --scores S       políticas A, B e C com tau escolhido nas 305, medidas nas 172 e no
          [--campo score]           estresse (docs mudados e nota oficial) -> N.json e spans/N__<pol>.jsonl
    rodar --nome v0 --v0            V0: a guarda de produção com o limiar varrido (só a política A)
    refazer                         refaz cada relatório do zero (refeito/) e repassa os spans finais pelo
                                    harness (`avaliar.avaliar`, variante cru) -> refeito.json
    tabela                          junta os relatórios em resumo.json

Formato do arquivo de scores: linha 1 opcional {"meta": {...}}; depois {"cid": <cid>, "score": <float>}
por candidato (o cid vem de candidatos.jsonl). Candidato sem score é erro.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from gama.indice import construir

from .. import avaliar
from ... import pontuar
from . import nucleo as nu


def _escrever(destino: pathlib.Path, obj) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", destino, flush=True)


def cmd_candidatos(a) -> None:
    info = nu.construir_candidatos(pathlib.Path(a.saida) if a.saida else nu.CANDIDATOS)
    _, docs = nu.ler_candidatos(nu.CANDIDATOS if not a.saida else a.saida)
    info["conferencia"] = nu.conferir(docs)
    print("conferência", info["conferencia"], flush=True)
    _escrever(nu.SAIDA / "candidatos.json", info)


def cmd_oraculo(a) -> None:
    meta_c, docs = nu.ler_candidatos(a.candidatos)
    sc = nu.scores_oraculo(docs)
    info = nu.gravar_scores(nu.SAIDA / "scores" / "oraculo.jsonl",
                            {"verificador": "oraculo", "nota": "teto: 1 se casa com o ouro (mesmo tipo, IoU >= 0,5)",
                             "candidatos_sha256": nu.sha256(a.candidatos)}, sc)
    print(info, flush=True)


def cmd_de_spans(a) -> None:
    _, docs = nu.ler_candidatos(a.candidatos)
    meta, sc = nu.scores_de_spans(docs, [pathlib.Path(s) for s in a.spans])
    meta.update({"verificador": a.nome, "regra": "maior score de span previsto do mesmo tipo (VAGA junto de "
                 "JURIS) com IoU >= 0,5, senão 0", "candidatos_sha256": nu.sha256(a.candidatos)})
    print(nu.gravar_scores(nu.SAIDA / "scores" / f"{a.nome}.jsonl", meta, sc), flush=True)


def cmd_escolher(a) -> None:
    """Escolha da formulação do Laya SÓ nas 305: maior AUROC (score x casa com o ouro) sobre todos os
    candidatos das 305; empate -> a primeira na ordem declarada (f1, f2, f3)."""
    _, docs = nu.ler_candidatos(a.candidatos)
    linhas =[json.loads(x) for x in pathlib.Path(a.bruto).read_text(encoding="utf-8").splitlines()]
    meta = linhas[0]["meta"]
    p = {r["cid"]: r["p"] for r in linhas[1:]}
    _, gold = avaliar.textos_e_ouro("reais")
    cands = [x for G, R in docs["reais"].values() for x in G + R]
    doc_de = {x.cid: d for d, (G, R) in docs["reais"].items() for x in G + R}
    faltam = [x.cid for x in cands if x.cid not in p]
    if faltam:
        raise SystemExit(f"{len(faltam)} candidatos das 305 sem resposta no bruto")
    extra = [cid for cid in p if not cid.startswith("reais/")]
    if extra:
        raise SystemExit(f"o bruto tem {len(extra)} candidatos fora das 305: a escolha só pode ver as 305")
    rot = {x.cid: int(nu.casa(x.span, gold[doc_de[x.cid]], True)) for x in cands}
    forms = list(meta["formulacoes"])
    elegiveis = _elegiveis_a(docs["reais"])
    tabela = {}
    for f in forms:
        pares = [(p[x.cid][f], rot[x.cid]) for x in cands]
        fracos = [(p[x.cid][f], rot[x.cid]) for x in cands if x.cid in elegiveis]
        tabela[f] = {"auroc_todos": nu.auroc(pares), "auroc_elegiveis_A": nu.auroc(fracos),
                     "media_p": round(sum(v for v, _ in pares) / len(pares), 4),
                     "p_min": min(v for v, _ in pares), "p_max": max(v for v, _ in pares)}
    melhor = max(tabela[f]["auroc_todos"] for f in forms)
    escolhida = next(f for f in forms if tabela[f]["auroc_todos"] == melhor)
    out = {"criterio": "maior AUROC sobre todos os candidatos das 305 (rótulo: casa com o ouro, mesmo tipo, "
                       "IoU >= 0,5, VAGA junto de JURIS); empate -> primeira na ordem declarada",
           "conjunto": "reais (305) apenas", "candidatos": len(cands), "positivos": sum(rot.values()),
           "formulacoes": meta["formulacoes"], "tabela": tabela, "escolhida": escolhida,
           "bruto": {"arquivo": str(a.bruto), "sha256": nu.sha256(a.bruto), "meta": meta}}
    print(json.dumps(tabela, ensure_ascii=False), "escolhida", escolhida, flush=True)
    _escrever(nu.SAIDA / "escolha.json", out)


def _elegiveis_a(docs_c: dict) -> set:
    out = set()
    for G, R in docs_c.values():
        _, _, w, x = nu.grupos([c.span for c in G], [c.span for c in R])
        ids = {id(s) for s in w + x}
        out |= {c.cid for c in G + R if id(c.span) in ids}
    return out


def cmd_rodar(a) -> None:
    t0 = time.perf_counter()
    meta_c, docs = nu.ler_candidatos(a.candidatos)
    conf = nu.conferir(docs)
    if not conf["ok"]:
        raise SystemExit(f"candidatos não reproduzem o harness: {conf}")
    prod = {c: nu.producao(docs[c]) for c in nu.CONJUNTOS}
    oficial = None if a.sem_oficial else nu.Oficial(construir(pontuar.DB))
    out = {"verificador": a.nome, "candidatos": {"arquivo": str(a.candidatos), "sha256": nu.sha256(a.candidatos),
                                                   "meta": {k: v for k, v in meta_c.items() if k != "codigo"}},
           "conferencia_harness": conf, "codigo": avaliar.impressao(), "grade": nu.GRADE,
           "criterio_tau": "maior F1 exato nas 305; empate -> maior tau; medido nas 172 sem reescolha",
           "politicas": {}}
    if oficial is not None:
        out["producao_estresse_oficial"] = oficial("producao_guarda@0.95", prod["estresse"])
    if a.v0:
        out["scores"] = {"verificador": "v0", "nota": "confiança do Gama; a régua não tem sinal (entra onde o "
                         "Gama cai abaixo de tau): a guarda de produção com o limiar varrido"}
        out["politicas"]["V0-A"] = nu.avaliar_politica(a.nome, "V0-A", docs, None, oficial, prod)
    else:
        meta_s, sc = nu.ler_scores(a.scores, a.campo)
        falta = nu._faltando(docs, sc)
        if falta:
            raise SystemExit(f"{falta} candidatos sem score em {a.scores}")
        out["scores"] = {"arquivo": str(a.scores), "campo": a.campo, "sha256": nu.sha256(a.scores), "meta": meta_s}
        out["auroc"] = {}
        for c in nu.CONJUNTOS:
            _, gold = avaliar.textos_e_ouro(c)
            pares = [(sc[x.cid], int(nu.casa(x.span, gold[d], c != "estresse")))
                     for d, (G, R) in docs[c].items() for x in G + R]
            out["auroc"][c] = {"candidatos": len(pares), "positivos": sum(y for _, y in pares),
                               "auroc": nu.auroc(pares)}
        for pol in a.politicas.split(","):
            out["politicas"][pol] = nu.avaliar_politica(a.nome, pol, docs, sc, oficial, prod)
    out["segundos"] = round(time.perf_counter() - t0, 1)
    _escrever(pathlib.Path(a.saida) if a.saida else nu.SAIDA / f"{a.nome}.json", out)


def cmd_refazer(a) -> None:
    """Refaz do zero cada relatório gravado (mesmos candidatos e scores, processo novo) em refeito/ e
    compara com o gravado; depois repassa os spans finais de cada política por `avaliar.avaliar` (a
    função de entrada do harness, variante `cru`) e compara F1 e IC95 nas 305 e nas 172."""
    comparacao = {}
    for arq in sorted(nu.SAIDA.glob("*.json")):
        if arq.name in ("candidatos.json", "escolha.json", "resumo.json", "refeito.json"):
            continue
        antigo = json.loads(arq.read_text(encoding="utf-8"))
        if "politicas" not in antigo:
            continue
        ns = argparse.Namespace(candidatos=antigo["candidatos"]["arquivo"], nome=antigo["verificador"],
                                v0="V0-A" in antigo["politicas"], scores=antigo["scores"].get("arquivo"),
                                campo=antigo["scores"].get("campo", "score"), politicas=",".join(antigo["politicas"]),
                                sem_oficial=False, saida=str(nu.SAIDA / "refeito" / arq.name))
        cmd_rodar(ns)
        novo = json.loads(pathlib.Path(ns.saida).read_text(encoding="utf-8"))
        difs = [k for k in ("politicas", "auroc", "conferencia_harness", "producao_estresse_oficial")
                if antigo.get(k) != novo.get(k)]
        harness = {}
        for pol, res in novo["politicas"].items():
            rel = avaliar.avaliar(f"verificador_{antigo['verificador']}__{pol}", [pathlib.Path("/app") / res["spans"]["arquivo"]],
                                  com_oficial=False)
            cru = rel["variantes"]["cru"]
            harness[pol] = {c: {"f1_harness": cru[c]["extracao"]["f1"], "f1_relatorio": res["conjuntos"][c]["extracao"]["f1"],
                                "ic95_harness": cru[c].get("vs_gama_v13_guarda", {}).get("ic95"),
                                "ic95_relatorio": res["conjuntos"][c].get("vs_gama_v13_guarda", {}).get("ic95")}
                            for c in nu.CONJUNTOS}
            harness[pol]["iguais"] = all(v["f1_harness"] == v["f1_relatorio"] and v["ic95_harness"] == v["ic95_relatorio"]
                                         for v in harness[pol].values() if isinstance(v, dict))
        comparacao[arq.name] = {"relatorio_refeito_igual": not difs, "campos_diferentes": difs,
                                "harness_cru_igual": all(h["iguais"] for h in harness.values()), "harness": harness}
        print(arq.name, "refeito igual:", not difs, difs, "| harness cru igual:",
              comparacao[arq.name]["harness_cru_igual"], flush=True)
    _escrever(nu.SAIDA / "refeito.json", comparacao)


def cmd_tabela(a) -> None:
    linhas = []
    for arq in sorted(nu.SAIDA.glob("*.json")):
        if arq.name in ("candidatos.json", "escolha.json", "resumo.json"):
            continue
        r = json.loads(arq.read_text(encoding="utf-8"))
        for pol, res in r.get("politicas", {}).items():
            linhas.append({**res["resumo"], "relatorio": arq.name})
    _escrever(nu.SAIDA / "resumo.json", linhas)
    for x in linhas:
        print(x, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.controles.verificador", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="acao", required=True)
    p = sub.add_parser("candidatos")
    p.add_argument("--saida")
    for nome in ("oraculo", "de-spans", "escolher", "rodar", "refazer", "tabela"):
        p = sub.add_parser(nome)
        p.add_argument("--candidatos", default=str(nu.CANDIDATOS))
        if nome == "de-spans":
            p.add_argument("--nome", required=True)
            p.add_argument("--spans", action="append", required=True)
        if nome == "escolher":
            p.add_argument("--bruto", required=True)
        if nome == "rodar":
            p.add_argument("--nome", required=True)
            p.add_argument("--scores")
            p.add_argument("--campo", default="score")
            p.add_argument("--v0", action="store_true")
            p.add_argument("--politicas", default="A,B,C")
            p.add_argument("--sem-oficial", action="store_true")
            p.add_argument("--saida")
    a = ap.parse_args()
    if a.acao == "rodar" and not a.v0 and not a.scores:
        ap.error("rodar exige --scores (ou --v0)")
    nu.SAIDA.mkdir(parents=True, exist_ok=True)
    {"candidatos": cmd_candidatos, "oraculo": cmd_oraculo, "de-spans": cmd_de_spans, "escolher": cmd_escolher,
     "rodar": cmd_rodar, "refazer": cmd_refazer, "tabela": cmd_tabela}[a.acao](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
