# -*- coding: utf-8 -*-
"""Escolha da variante secundária e contrastes do GLiNER 2.5 zero-shot com a v2.1 zero-shot.

Tudo pelo harness validado (`bench.controles.avaliar`): leitura, aparar, guarda de produção,
F1 de extração e bootstrap (pareado por ementa, 2.000 reamostras, semente 0).

escolher   lê os relatórios do harness das três tentativas (só as 305) e aplica o critério fixado
           em `extrair_gliner25.py` antes de qualquer saída da 2.5: maior F1 cru nas 305 (IoU >= 0,5,
           VAGA em JURIS); empate -> maior F1 exato; depois o limiar mais perto de 0,5.
contrastar 2.5 menos v2.1, ambos zero-shot, cru e com a guarda@0,95, nas 305 e nas 172; no estresse,
           as referências vagas achadas (tp de VAGA, de 714) e o F1 por tipo. A v2.1 é
           `saidas/bench/gliner.jsonl` (estresse + 305, a gravação do benchmark) e, nas 172, a
           referência rodada no job A. Confere também que a v2.1 rodada de novo nas 305 reproduz a
           gravação do benchmark.

    docker compose run --rm gama python -m bench.controles.gliner25.comparar escolher \\
        --relatorios saidas/bench/controles/gliner25/gliner25_t1_desc_050.json ... --saida ...
    docker compose run --rm gama python -m bench.controles.gliner25.comparar contrastar \\
        --g25 <jsonl da 2.5> --v21-novas <jsonl da referência v2.1> --saida ...
"""
from __future__ import annotations

import argparse
import json
import pathlib

from bench import pontuar
from bench.controles import avaliar as av

LIMIAR_TENTATIVA = {"t1_desc_050": 0.5, "t2_desc_030": 0.3, "t3_desc_070": 0.7}


def escolher(relatorios: list[str]) -> dict:
    linhas = []
    for arq in relatorios:
        rel = json.loads(pathlib.Path(arq).read_text(encoding="utf-8"))
        meta = next(iter(rel["meta"].values()))
        nome = meta["variante"]
        assert rel["presentes"]["reais"]["faltam"] == 0, arq
        assert all(rel["presentes"][c]["documentos"] == 0 for c in ("estresse", "novas")), \
            f"{arq}: a tentativa só pode ter as 305"
        e = rel["variantes"]["cru"]["reais"]["extracao"]
        linhas.append({"variante": nome, "limiar": meta["limiar_decisao"], "f1": e["f1"], "f1_exato": e["f1_exato"],
                       "p": e["p"], "r": e["r"], "tp": e["tp"], "fp": e["fp"], "fn": e["fn"],
                       "f1_guarda095": rel["variantes"]["guarda@0.95"]["reais"]["extracao"]["f1"],
                       "relatorio": arq})
    melhor = max(linhas, key=lambda x: (x["f1"], x["f1_exato"], -abs(x["limiar"] - 0.5)))
    return {"criterio": "maior F1 cru nas 305 (IoU >= 0,5, VAGA em JURIS); empate: F1 exato; depois limiar "
                        "mais perto de 0,5", "tentativas": linhas, "escolhida": melhor["variante"]}


def _spans(arquivos: list[str], conjunto: str):
    _, linhas = av.ler(arquivos)
    return av.spans(linhas, conjunto)


def contrastar(g25: list[str], v21_novas: str) -> dict:
    v21 = {"estresse": [str(av.BENCH / "gliner.jsonl")], "reais": [str(av.BENCH / "gliner.jsonl")],
           "novas": [v21_novas]}
    out = {"g25": g25, "v21": v21, "codigo": av.impressao(), "conjuntos": {}}
    for c in av.CONJUNTOS:
        a, b = _spans(g25, c), _spans(v21[c], c)
        if a is None or b is None:
            out["conjuntos"][c] = "faltam documentos"
            continue
        res = {}
        for v, t in (("cru", None), ("guarda@0.95", av.LIMIAR_PRODUCAO)):
            fa = a if t is None else av.com_guarda(a, av.regua(c), t)
            fb = b if t is None else av.com_guarda(b, av.regua(c), t)
            ea, eb = av.extracao(c, fa), av.extracao(c, fb)
            r = {"f1_g25": ea["f1"], "f1_v21": eb["f1"],
                 "por_tipo_g25": {k: {m: x[m] for m in ("f1", "tp", "fp", "fn", "p", "r")} for k, x in ea["por_tipo"].items()},
                 "por_tipo_v21": {k: {m: x[m] for m in ("f1", "tp", "fp", "fn", "p", "r")} for k, x in eb["por_tipo"].items()}}
            if c != "estresse":
                r["g25_menos_v21"] = av.bootstrap(av.por_doc(c, fa), av.por_doc(c, fb))
            else:
                r["vagas_achadas_g25"] = ea["por_tipo"]["VAGA"]["tp"]
                r["vagas_achadas_v21"] = eb["por_tipo"]["VAGA"]["tp"]
                r["vagas_ouro"] = ea["por_tipo"]["VAGA"]["tp"] + ea["por_tipo"]["VAGA"]["fn"]
            res[v] = r
        out["conjuntos"][c] = res
    # a v2.1 rodada de novo nas 305 (arquivo da referência) contra a gravação do benchmark
    _, orig = pontuar._ler_spans(av.BENCH / "gliner.jsonl")
    _, novo = pontuar._ler_spans(pathlib.Path(v21_novas))
    ids = [k for k in orig if k[0] == "reais" and k in novo]
    if ids:
        chave = lambda ss: [tuple(s[:3]) for s in ss]  # noqa: E731
        dif = sum(chave(orig[k]["spans"]) != chave(novo[k]["spans"]) for k in ids)
        maxs = max((abs(p[3] - q[3]) for k in ids if chave(orig[k]["spans"]) == chave(novo[k]["spans"])
                    for p, q in zip(orig[k]["spans"], novo[k]["spans"])), default=0.0)
        f_novo = av.extracao("reais", _spans([v21_novas], "reais"))["f1"] if len(ids) == 305 else None
        out["conferencia_v21_reais"] = {"docs": len(ids), "spans_diferentes": dif, "max_dif_score": round(maxs, 6),
                                        "f1_rodada_nova": f_novo,
                                        "f1_gravacao_benchmark": av.extracao("reais", _spans([str(av.BENCH / "gliner.jsonl")], "reais"))["f1"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("escolher")
    e.add_argument("--relatorios", nargs="+", required=True)
    e.add_argument("--saida", required=True)
    c = sub.add_parser("contrastar")
    c.add_argument("--g25", action="append", required=True)
    c.add_argument("--v21-novas", required=True)
    c.add_argument("--saida", required=True)
    a = ap.parse_args()
    res = escolher(a.relatorios) if a.cmd == "escolher" else contrastar(a.g25, a.v21_novas)
    pathlib.Path(a.saida).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res if a.cmd == "escolher" else {k: v for k, v in res.items() if k != "codigo"},
                     ensure_ascii=False, indent=1)[:6000])
    print("->", a.saida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
