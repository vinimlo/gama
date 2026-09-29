# -*- coding: utf-8 -*-
"""Pontuação e comparações do GLiNER 2.5 fine-tunado, no container, a partir dos arquivos gravados.

    pontuar   o harness validado (`bench.controles.avaliar.avaliar`, sem mudança) com a pasta de
              saída trocada para saidas/bench/controles/gliner25/ft/ (relatório e JSON da métrica
              oficial ficam aqui, fora das pastas dos outros experimentos). Variantes: cru,
              guarda@0.95 e guarda no limiar escolhido nas 305 (a grade e o critério do harness).
    comparar  bootstrap pareado por ementa (o do harness: 2.000 reamostras, semente 0) do 2.5
              fine-tunado contra: o GLiNER v2.1 fine-tunado na MESMA variante (cru x cru, guarda@0.95
              x guarda@0.95, cada um no próprio limiar escolhido), o Gama v1.3 cru e o Gama v1.3 com a
              guarda (este já vem do harness e é repetido para conferência).
    auroc     o score separa erro de acerto? Nos spans crus (aparados) nas 305 e nas 172, erro =
              previsão sem par no casamento 1 para 1 de `bench.pontuar.extracao` (VAGA em JURIS,
              IoU >= 0,5), a definição de `bench/controles/offline/risco_cobertura.py`, cujas funções
              são importadas (casar_extracao, auroc, faixas). AUROC com IC95 por documento e a
              diferença pareada 2.5 - v2.1 (mesmas reamostras de documentos). O score da 2.5 é o
              gravado (4 casas, como o da v2.1) e, à parte, o de precisão cheia do arquivo de
              candidatos. Também a fração de falsos positivos com score >= 0,95 e >= 0,99.

    docker compose run --rm gama python -m bench.controles.gliner25.ft_analisar pontuar \\
        --nome gliner25_ft --spans <modelo_gliner25_ft.jsonl>
    docker compose run --rm gama python -m bench.controles.gliner25.ft_analisar comparar \\
        --relatorio saidas/bench/controles/gliner25/ft/gliner25_ft.json
    docker compose run --rm gama python -m bench.controles.gliner25.ft_analisar auroc \\
        --spans <modelo_gliner25_ft.jsonl> --candidatos <candidatos_gliner25_ft.jsonl>
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random

from bench import pontuar
from bench.controles import avaliar as A
from bench.controles.offline import risco_cobertura as RC

AQUI = A.SAIDA / "gliner25" / "ft"
V21_SPANS = A.SAIDA / "_hub/bench/saida/controles/gliner_ft/modelo_gliner_ft.jsonl"
V21_RELATORIO = A.SAIDA / "gliner_ft.json"
REAMOSTRAS, SEMENTE = A.REAMOSTRAS, A.SEMENTE


def limiar_de(variante: str) -> float | None:
    return None if variante == "cru" else float(variante.split("@")[1])


def finais(linhas: dict, conjunto: str, variante: str) -> dict:
    sp = A.spans(linhas, conjunto)
    t = limiar_de(variante)
    return sp if t is None else A.com_guarda(sp, A.regua(conjunto), t)


# ---------------------------------------------------------------- pontuar

def cmd_pontuar(a) -> int:
    AQUI.mkdir(parents=True, exist_ok=True)
    A.SAIDA = AQUI                                   # só o destino muda; REGUA/REFERENCIA são absolutos
    res = A.avaliar(a.nome, [pathlib.Path(s) for s in a.spans], [], True)
    destino = pathlib.Path(a.saida) if a.saida else AQUI / f"{a.nome}.json"
    destino.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", destino)
    return 0


# ---------------------------------------------------------------- comparar

def cmd_comparar(a) -> int:
    rel = json.loads(pathlib.Path(a.relatorio).read_text(encoding="utf-8"))
    rel21 = json.loads(V21_RELATORIO.read_text(encoding="utf-8"))
    _, l25 = A.ler(rel["arquivos"])
    _, l21 = A.ler([V21_SPANS])
    v25 = list(rel["variantes"])
    v21 = list(rel21["variantes"])
    par = {"cru": "cru", "guarda@0.95": "guarda@0.95"}
    esc25, esc21 = rel["varredura_reais"]["variante"], rel21["varredura_reais"]["variante"]
    out = {"relatorio": a.relatorio, "relatorio_v21": str(V21_RELATORIO), "codigo": A.impressao(),
           "pares": {**par, f"{esc25} (escolhido 2.5)": f"{esc21} (escolhido v2.1)"}, "contrastes": {}}
    for c in ("reais", "novas"):
        v13 = A.spans(pontuar._ler_spans(A.REFERENCIA[c])[1], c)
        ref_cru, ref_guarda = A.por_doc(c, v13), A.referencia(c)
        for v in v25:
            f25 = finais(l25, c, v)
            cont = A.por_doc(c, f25)
            outro = par.get(v, esc21 if v == esc25 else None)
            x = {"f1": A.extracao(c, f25)["f1"],
                 "vs_gama_v13_cru": A.bootstrap(cont, ref_cru),
                 "vs_gama_v13_guarda": A.bootstrap(cont, ref_guarda)}
            if outro in v21:
                f21 = finais(l21, c, outro)
                x["gliner_v21_ft_variante"] = outro
                x["gliner_v21_ft_f1"] = A.extracao(c, f21)["f1"]
                x["vs_gliner_v21_ft"] = A.bootstrap(cont, A.por_doc(c, f21))
            out["contrastes"].setdefault(v, {})[c] = x
            print(c, v, json.dumps({k: (y["dif"], y["ic95"]) if isinstance(y, dict) else y
                                    for k, y in x.items()}, ensure_ascii=False), flush=True)
    destino = pathlib.Path(a.saida) if a.saida else AQUI / "gliner25_ft_comparacao.json"
    destino.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", destino)
    return 0


# ---------------------------------------------------------------- auroc

def precisao_cheia(candidatos: pathlib.Path) -> dict:
    """{(conjunto, id): {(ini, fim, ROT, score4): score}} a partir dos candidatos do piso."""
    from bench.controles.gliner25.ft_treinar import ROTULOS
    out = {}
    with candidatos.open(encoding="utf-8") as fh:
        for linha in fh:
            r = json.loads(linha)
            if "meta" in r:
                continue
            m = {}
            for jan in r["janelas"]:
                a = jan["janela"][0]
                for nome, cs in jan["c"].items():
                    for s, f, sc in cs:
                        chave = (a + s, a + f, ROTULOS[nome], round(sc, 4))
                        m[chave] = max(m.get(chave, 0.0), sc)
            out[(r["conjunto"], r["id"])] = m
    return out


def itens(linhas: dict, conjunto: str, cheia: dict | None = None) -> tuple[list, int]:
    """Um item por span cru aparado: {doc, conf, erro}. `cheia` troca o score gravado pelo de
    precisão cheia (casando o span gravado com o candidato de mesmo trecho, rótulo e score)."""
    sp = A.spans(linhas, conjunto)
    _, gold = A.textos_e_ouro(conjunto)
    brutos = {d: {(s[0], s[1]): s for s in linhas[(conjunto, d)]["spans"]} for d in sp}
    out, sem_par = [], 0
    for d, ss in sp.items():
        acertos = RC.casar_extracao(gold.get(d, []), [(s.inicio, s.fim, pontuar.rotulo(s)) for s in ss])
        for s, ok in zip(ss, acertos):
            conf = s.confianca
            if cheia is not None:
                g = next((x for (i, f), x in brutos[d].items() if i <= s.inicio and s.fim <= f), None)
                v = cheia.get((conjunto, d), {}).get((g[0], g[1], g[2], g[3])) if g else None
                if v is None:
                    sem_par += 1
                else:
                    conf = v
            out.append({"doc": d, "conf": conf, "erro": int(not ok)})
    return out, sem_par


def auroc_pareado(a: list, b: list) -> dict:
    """AUROC(a) - AUROC(b), reamostrando os mesmos documentos para os dois."""
    pa, pb = collections.defaultdict(list), collections.defaultdict(list)
    for it in a:
        pa[it["doc"]].append(it)
    for it in b:
        pb[it["doc"]].append(it)
    docs = sorted(set(pa) | set(pb))
    rng = random.Random(SEMENTE)
    difs = []
    for _ in range(REAMOSTRAS):
        amostra = [rng.choice(docs) for _ in docs]
        x = RC.auroc([i for d in amostra for i in pa[d]])
        y = RC.auroc([i for d in amostra for i in pb[d]])
        if x is not None and y is not None:
            difs.append(x - y)
    difs.sort()
    ic = [round(difs[int(0.025 * len(difs))], 4), round(difs[int(0.975 * len(difs)) - 1], 4)] \
        if len(difs) == REAMOSTRAS else None
    return {"dif": round(RC.auroc(a) - RC.auroc(b), 4), "ic95": ic,
            "frac_dif_menor_ou_igual_zero": round(sum(x <= 0 for x in difs) / max(1, len(difs)), 4),
            "reamostras_validas": len(difs), "semente": SEMENTE, "unidade": "documento"}


def resumo_score(its: list) -> dict:
    fps = [i for i in its if i["erro"]]
    tps = [i for i in its if not i["erro"]]
    r = RC.resumir(its, None, com_faixas=False, com_boot=True)
    r = {k: r[k] for k in ("previsoes", "erros", "risco_total", "auroc", "auroc_ic95", "aurc", "aurc_oraculo")
         if k in r}
    r["fp_score_ge_095"] = sum(i["conf"] >= 0.95 for i in fps)
    r["fp_score_ge_099"] = sum(i["conf"] >= 0.99 for i in fps)
    r["fp_score_ge_0999"] = sum(i["conf"] >= 0.999 for i in fps)
    r["tp_score_ge_099"] = sum(i["conf"] >= 0.99 for i in tps)
    r["empates_no_topo"] = sum(i["conf"] >= 0.99995 for i in its)
    r["faixas"] = RC.faixas(its)
    return r


def cmd_auroc(a) -> int:
    _, l25 = A.ler([pathlib.Path(a.spans)])
    _, l21 = A.ler([V21_SPANS])
    cheia = precisao_cheia(pathlib.Path(a.candidatos)) if a.candidatos else None
    out = {"definicao": "spans crus aparados; erro = previsão sem par no casamento 1 para 1 de "
                        "bench.pontuar.extracao (VAGA em JURIS, IoU >= 0,5); AUROC = P(score de um acerto > "
                        "score de um erro), empate 1/2; IC95 por bootstrap de documentos, 2.000 reamostras, "
                        "semente 0", "spans": a.spans, "candidatos": a.candidatos, "conjuntos": {}}
    for c in ("reais", "novas"):
        v13 = pontuar._ler_spans(A.REFERENCIA[c])[1]
        i25, _ = itens(l25, c)
        i21, _ = itens(l21, c)
        i13, _ = itens(v13, c)
        x = {"gliner25_ft": resumo_score(i25), "gliner_v21_ft": resumo_score(i21), "gama_v13": resumo_score(i13),
             "dif_auroc_25_menos_v21": auroc_pareado(i25, i21),
             "dif_auroc_25_menos_v13": auroc_pareado(i25, i13)}
        if cheia is not None:
            ic, sem_par = itens(l25, c, cheia)
            x["gliner25_ft_precisao_cheia"] = resumo_score(ic)
            x["gliner25_ft_precisao_cheia"]["spans_sem_candidato"] = sem_par
            x["dif_auroc_25cheia_menos_v21"] = auroc_pareado(ic, i21)
        out["conjuntos"][c] = x
        print(c, json.dumps({k: (v.get("auroc"), v.get("auroc_ic95")) if "auroc" in v else (v["dif"], v["ic95"])
                             for k, v in x.items()}, ensure_ascii=False), flush=True)
    destino = pathlib.Path(a.saida) if a.saida else AQUI / "gliner25_ft_auroc.json"
    destino.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", destino)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pontuar")
    p.add_argument("--nome", required=True)
    p.add_argument("--spans", action="append", required=True)
    p.add_argument("--saida")
    c = sub.add_parser("comparar")
    c.add_argument("--relatorio", required=True)
    c.add_argument("--saida")
    u = sub.add_parser("auroc")
    u.add_argument("--spans", required=True)
    u.add_argument("--candidatos")
    u.add_argument("--saida")
    a = ap.parse_args()
    return {"pontuar": cmd_pontuar, "comparar": cmd_comparar, "auroc": cmd_auroc}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
