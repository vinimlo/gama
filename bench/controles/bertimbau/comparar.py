# -*- coding: utf-8 -*-
"""H3: BERTimbau (receita do v1.2) contra o Gama v1.2 e o v1.3, do zero, a partir dos JSONL gravados.

Tudo sai de `bench.controles.avaliar` (nada de métrica nova):
  1. o relatório padrão do harness para o BERTimbau (cru, guarda@0.95, guarda@t* escolhido nas
     305 e medido nas 172; diferença para o v1.3 com guarda, IC95 pareado);
  2. o mesmo relatório para o v1.2 (spans publicados da L4, `saidas/bench/modelo_professor.jsonl`)
     e para o v1.3 (`gama.jsonl` nas 305 e no estresse, `modelo_base12.jsonl` nas 172), para as
     comparações pareadas de igual para igual que o harness não faz: BERTimbau menos v1.2 e
     menos v1.3, cru contra cru, guarda@0.95 contra guarda@0.95 e cada um no seu t*;
  3. tempo por documento na L4: BERTimbau, v1.2 e v1.3 extraídos no MESMO job (mesma máquina),
     média, mediana e p95 por conjunto; e a reprodução dos spans publicados do v1.2 e do v1.3
     pelos extraídos agora (documentos com spans diferentes, maior diferença de confiança).

    docker compose run --rm gama python -m bench.controles.bertimbau.comparar \\
        --bertimbau saidas/bench/controles/_hub/bench/saida/controles/bertimbau/modelo_bertimbau.jsonl \\
        --mesmo-job-v12 .../modelo_v12.jsonl --mesmo-job-v13 .../modelo_v13.jsonl

Saídas: `saidas/bench/controles/bertimbau/comparacao.json` e os relatórios do harness em
`saidas/bench/controles/bertimbau/{bertimbau_s13,v12_ref,v13_ref}.json`.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from gama.indice import Indice

from ... import pontuar
from .. import avaliar

AQUI = avaliar.SAIDA / "bertimbau"
REF_V12 = [avaliar.BENCH / "modelo_professor.jsonl"]
# v1.3: o arquivo de referência do harness em cada conjunto (gama.jsonl não tem as 172)
REF_V13 = [avaliar.BENCH / "modelo_base12.jsonl", avaliar.BENCH / "gama.jsonl"]


def _variantes_comparaveis(rel: dict) -> dict:
    """{cru, guarda@0.95, guarda@t*} -> nome da variante no relatório."""
    out = {"cru": "cru", "guarda@0.95": avaliar.variante(avaliar.LIMIAR_PRODUCAO)}
    if rel["varredura_reais"].get("escolhido") is not None:
        out["guarda@t*"] = rel["varredura_reais"]["variante"]
    return out


def _finais(linhas: dict, conjunto: str, t: float | None) -> dict:
    sp = avaliar.spans(linhas, conjunto)
    return sp if t is None else avaliar.com_guarda(sp, avaliar.regua(conjunto), t)


def _limiar(nome_variante: str) -> float | None:
    return None if nome_variante == "cru" else float(nome_variante.split("@")[1])


def pareados(cands: dict, rels: dict) -> dict:
    """BERTimbau menos cada referência, por variante comparável, nas 305 e nas 172."""
    _, lb = avaliar.ler(cands["bertimbau"])
    out = {}
    for ref in ("v12", "v13"):
        _, lr = avaliar.ler(cands[ref])
        vb, vr = _variantes_comparaveis(rels["bertimbau"]), _variantes_comparaveis(rels[ref])
        for chave in ("cru", "guarda@0.95", "guarda@t*"):
            if chave not in vb or chave not in vr:
                continue
            for c in ("reais", "novas"):
                cb = avaliar.por_doc(c, _finais(lb, c, _limiar(vb[chave])))
                cr = avaliar.por_doc(c, _finais(lr, c, _limiar(vr[chave])))
                b = avaliar.bootstrap(cb, cr)
                out[f"bertimbau_menos_{ref}|{chave}|{c}"] = {
                    "bertimbau": vb[chave], ref: vr[chave], **b,
                    "f1_bertimbau": round(avaliar.alunos._f1(cb, sorted(cb)), 4),
                    f"f1_{ref}": round(avaliar.alunos._f1(cr, sorted(cr)), 4)}
    return out


def tempos(arquivos: dict) -> dict:
    """Segundos por documento (L4) por conjunto: média, mediana, p95; e sem o 1º documento."""
    out = {}
    for nome, arq in arquivos.items():
        meta, linhas = pontuar._ler_spans(pathlib.Path(arq))
        por = {}
        for c in avaliar.CONJUNTOS:
            textos, _ = avaliar.textos_e_ouro(c)
            v = sorted(linhas[(c, d)]["segundos"] for d in textos if (c, d) in linhas)
            if not v:
                continue
            por[c] = {"docs": len(v), "media": round(sum(v) / len(v), 4), "mediana": round(v[len(v) // 2], 4),
                      "p95": round(v[int(0.95 * len(v)) - 1], 4), "max": round(v[-1], 4)}
        todos = [linhas[k]["segundos"] for k in linhas]
        out[nome] = {"arquivo": str(arq), "dispositivo": meta.get("dispositivo"), "modelo": meta.get("modelo"),
                     "parametros": meta.get("parametros"), "max_len": meta.get("max_len"),
                     "vram_pico_mib": meta.get("vram_pico_mib"), "por_conjunto": por,
                     "todos_os_docs_do_arquivo": {"docs": len(todos), "media": round(sum(todos) / len(todos), 4)}}
    return out


def reproducao(novo: str, publicados: list) -> dict:
    """Spans extraídos agora (mesmo job do BERTimbau) contra os publicados, por documento."""
    _, ln = pontuar._ler_spans(pathlib.Path(novo))
    lp = {}
    for p in publicados:
        lp.update(pontuar._ler_spans(pathlib.Path(p))[1])
    out = {}
    for c in avaliar.CONJUNTOS:
        textos, _ = avaliar.textos_e_ouro(c)
        comuns = [d for d in textos if (c, d) in ln and (c, d) in lp]
        dif_conf = 0.0
        n_dif = 0
        for d in comuns:
            a, b = ln[(c, d)]["spans"], lp[(c, d)]["spans"]
            if [s[:4] for s in a] != [s[:4] for s in b]:
                n_dif += 1
                continue
            for x, y in zip(a, b):
                dif_conf = max(dif_conf, abs((x[5] or 0) - (y[5] or 0)))
        out[c] = {"docs": len(comuns), "docs_com_spans_diferentes": n_dif,
                  "max_dif_confianca_nos_iguais": round(dif_conf, 8)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bertimbau", required=True)
    ap.add_argument("--mesmo-job-v12", required=True)
    ap.add_argument("--mesmo-job-v13", required=True)
    ap.add_argument("--nome", default="bertimbau_s13")
    a = ap.parse_args()
    AQUI.mkdir(parents=True, exist_ok=True)
    idx = Indice.do_banco(pontuar.DB)
    cands = {"bertimbau": [pathlib.Path(a.bertimbau)], "v12": REF_V12, "v13": REF_V13}
    nomes = {"bertimbau": a.nome, "v12": "v12_ref", "v13": "v13_ref"}
    rels = {}
    for k, arqs in cands.items():
        rels[k] = avaliar.avaliar(nomes[k], arqs, idx=idx)
        (AQUI / f"{nomes[k]}.json").write_text(json.dumps(rels[k], ensure_ascii=False, indent=1), encoding="utf-8")
    out = {
        "resumo": {k: rels[k]["resumo"] for k in rels},
        "varredura_reais": {k: {x: rels[k]["varredura_reais"].get(x) for x in ("escolhido", "variante")} for k in rels},
        "pareados": pareados(cands, rels),
        "tempos_l4_mesmo_job": tempos({"bertimbau": a.bertimbau, "v12": a.mesmo_job_v12, "v13": a.mesmo_job_v13}),
        "tempos_l4_publicados": tempos({"v12": REF_V12[0], "v13": REF_V13[0]}),
        "reproducao_spans_publicados": {"v12": reproducao(a.mesmo_job_v12, REF_V12),
                                        "v13": reproducao(a.mesmo_job_v13, REF_V13)},
        "codigo": avaliar.impressao(),
        "relatorios": {k: str(AQUI / f"{nomes[k]}.json") for k in rels},
    }
    (AQUI / "comparacao.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in out["pareados"].items():
        print(k, v["f1_bertimbau"], v[f"f1_{k.split('|')[0].rsplit('_', 1)[1]}"], v["dif"], v["ic95"], flush=True)
    print("->", AQUI / "comparacao.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
