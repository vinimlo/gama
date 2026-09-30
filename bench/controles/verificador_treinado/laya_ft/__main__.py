# -*- coding: utf-8 -*-
"""CLI local do candidato laya_ft. Tudo no container, em CPU:

    docker compose run --rm gama python -m bench.controles.verificador_treinado.laya_ft <acao> [...]

    politicas     políticas A, B e C pelo CLI de bench/controles/verificador (`rodar`: tau escolhido
                  nas 305, medido nas 172 e no estresse, IC95 pareado contra a guarda, AUROC nos três
                  conjuntos), com a raiz de saída do CLI (`nucleo.SAIDA`) apontada para a pasta deste
                  candidato -> laya_ft.json, spans/laya_ft__<pol>.jsonl, json/estresse/...
    refazer       o `refazer` e o `tabela` do mesmo CLI nesta pasta: relatório refeito do zero a partir
                  dos arquivos gravados e spans finais repassados pelo harness -> refeito/, refeito.json,
                  resumo.json
    auroc         AUROC na validação interna (scores dos pesos gravados) no todo e por fatia, mais as
                  AUROC do relatório nos três conjuntos; confere a pergunta f3 com a do zero-shot -> auroc.json
    paridade      a cópia mínima do forward (laya_min, CPU, fp32) contra os scores do HF Jobs numa
                  amostra fixa de candidatos -> paridade.json
    dev           checagem no dev (26 docs, não sai da máquina): Gama v1.3 cru e régua aqui, candidatos
                  como os do verificador, score pela cópia mínima, políticas com o tau de laya_ft.json;
                  documentos mudados contra a produção e nota oficial -> dev.json (+ dev/)

Entradas baixadas no host com o hf CLI (revisões no relatório):
    hf download vinimlo/gama-goldenset bench/saida/controles/verificador_treinado/laya_ft.jsonl \\
        --repo-type dataset --revision <oid> --local-dir saidas/bench/controles/verificador_treinado/laya_ft/_hub
    hf download vinimlo/gama-exp-verif-laya_ft --revision <oid> \\
        --local-dir saidas/bench/controles/verificador_treinado/laya_ft/_modelo
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

from bench.controles import avaliar
from bench.controles.verificador import nucleo as vn

SAIDA = avaliar.SAIDA / "verificador_treinado" / "laya_ft"     # /app/saidas/bench/controles/verificador_treinado/laya_ft
SCORES = SAIDA / "_hub" / "bench" / "saida" / "controles" / "verificador_treinado" / "laya_ft.jsonl"
MODELO = SAIDA / "_modelo"
VALIDACAO_SCORES = MODELO / "avaliacao" / "validacao_scores.jsonl"
DADOS = avaliar.SAIDA / "verificador_treinado" / "dados"
NOME = "laya_ft"
THREADS = 4                                                    # = cpus do serviço no compose


def _escrever(destino: pathlib.Path, obj) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", destino, flush=True)


def _jsonl(arq) -> tuple[dict, list]:
    meta, linhas = {}, []
    for x in pathlib.Path(arq).read_text(encoding="utf-8").splitlines():
        r = json.loads(x)
        if "meta" in r:
            meta = r["meta"]
        else:
            linhas.append(r)
    return meta, linhas


def _cli_verificador(argv: list) -> int:
    """O CLI de bench/controles/verificador sem mudança, com a raiz de saída nesta pasta."""
    from bench.controles.verificador import __main__ as vm
    vn.SAIDA = SAIDA
    sys.argv = ["python -m bench.controles.verificador", *argv]
    return vm.main()


def cmd_politicas(a) -> None:
    _cli_verificador(["rodar", "--nome", NOME, "--scores", str(SCORES)])


def cmd_refazer(a) -> None:
    _cli_verificador(["refazer"])
    _cli_verificador(["tabela"])


def cmd_auroc(a) -> None:
    from bench.controles.verificador import laya_job
    from . import treinar_job
    if treinar_job.PERGUNTA != laya_job.FORMULACOES["f3"]["pergunta"]:
        raise SystemExit("a pergunta do treino não é a f3 do zero-shot")
    meta_v, sv = _jsonl(VALIDACAO_SCORES)
    _, val = _jsonl(DADOS / "validacao.jsonl")
    info = {r["cid"]: r for r in val}
    if {r["cid"] for r in sv} != set(info) or any(r["rotulo"] != info[r["cid"]]["rotulo"] for r in sv):
        raise SystemExit("scores de validação não batem com validacao.jsonl")
    fatias = collections.defaultdict(list)
    for r in sv:
        x = info[r["cid"]]
        par = (r["score"], int(r["rotulo"]))
        fatias["todos"].append(par)
        fatias[f"conjunto={x['conjunto']}"].append(par)
        fatias[f"{x['conjunto']}/{x['origem']}"].append(par)
        if x["elegivel"]["A"]:
            fatias["elegivel_A"].append(par)
        if x["elegivel"]["B"]:
            fatias["elegivel_B"].append(par)
        if x["forte"]:
            fatias["gama_forte"].append(par)
    val_auroc = {k: {"n": len(v), "positivos": sum(y for _, y in v), "auroc": vn.auroc(v)}
                 for k, v in sorted(fatias.items())}
    rel = json.loads((SAIDA / f"{NOME}.json").read_text(encoding="utf-8"))
    treino = json.loads((MODELO / "treino.json").read_text(encoding="utf-8"))
    # fatias dos conjuntos de avaliação: rótulo = scores/oraculo.jsonl do verificador (casa com o ouro)
    _, orac = _jsonl(vn.SAIDA / "scores" / "oraculo.jsonl")
    _, sc = _jsonl(SCORES)
    rot, s = {r["cid"]: int(r["score"]) for r in orac}, {r["cid"]: r["score"] for r in sc}
    fat_c = collections.defaultdict(list)
    for linha in vn.CANDIDATOS.read_text(encoding="utf-8").splitlines()[1:]:
        r = json.loads(linha)
        par = (s[r["cid"]], rot[r["cid"]])
        for k in ("A", "B"):
            if r["elegivel"][k]:
                fat_c[f"{r['conjunto']}/elegivel_{k}"].append(par)
        fat_c[f"{r['conjunto']}/{r['origem']}"].append(par)
    conj_fatias = {k: {"n": len(v), "positivos": sum(y for _, y in v), "auroc": vn.auroc(v)}
                   for k, v in sorted(fat_c.items())}
    out = {"validacao_interna": val_auroc, "conjuntos": rel["auroc"], "conjuntos_fatias": conj_fatias,
           "validacao_no_job_de_treino": treino["validacao_pesos_gravados"],
           "scores_validacao": {"arquivo": vn._rel(VALIDACAO_SCORES), "sha256": vn.sha256(VALIDACAO_SCORES),
                                "modelo": meta_v.get("modelo")},
           "pergunta_igual_f3_zero_shot": True}
    print(json.dumps({k: v["auroc"] for k, v in val_auroc.items()}), rel["auroc"], flush=True)
    _escrever(SAIDA / "auroc.json", out)


def _pontuador(modelo=MODELO):
    from . import laya_min, treinar_job
    return laya_min.Pontuador(str(modelo), treinar_job.PERGUNTA, threads=THREADS)


def cmd_paridade(a) -> None:
    meta_s, ls = _jsonl(SCORES)
    gpu = {r["cid"]: r["score"] for r in ls}
    _, cands = _jsonl(vn.CANDIDATOS)
    amostra = cands[:: a.passo]
    est = list(dict.fromkeys(r["estado"] for r in amostra))
    pont = _pontuador()
    t0 = time.perf_counter()
    sc, trunc = pont(est)
    seg = time.perf_counter() - t0
    cpu = dict(zip(est, sc))
    difs = [abs(cpu[r["estado"]] - gpu[r["cid"]]) for r in amostra]
    grade = vn.GRADE
    lados = sum(any((cpu[r["estado"]] >= t) != (gpu[r["cid"]] >= t) for t in grade) for r in amostra)
    out = {"amostra": {"regra": f"candidatos.jsonl[::{a.passo}]", "candidatos": len(amostra), "estados": len(est),
                       "por_conjunto": dict(collections.Counter(r["conjunto"] for r in amostra))},
           "cpu": {"precisao": "fp32", "threads": THREADS, "temperatura": pont.t, "truncados": trunc,
                   "segundos": round(seg, 2), "ms_por_estado": round(1000 * seg / max(1, len(est)), 1)},
           "gpu": {"modelo": meta_s.get("modelo"), "dispositivo": meta_s.get("dispositivo")},
           "dif_max": max(difs), "dif_media": sum(difs) / len(difs),
           "candidatos_que_mudam_de_lado_em_algum_tau_da_grade": lados}
    print(json.dumps(out, ensure_ascii=False), flush=True)
    _escrever(SAIDA / "paridade.json", out)
    if out["dif_max"] > 1e-3 or lados:
        raise SystemExit("cópia mínima não reproduz os scores do HF Jobs")


def cmd_dev(a) -> None:
    """Dev: 26 documentos da organização, só aqui, em CPU."""
    ensaio = a.modelo is not None          # ensaio do caminho: outro checkpoint e taus dados, nada gravado em dev.json
    from gama.extratores import carregar
    from gama.indice import construir
    from bench import conjuntos, pontuar

    pasta = SAIDA / ("dev_ensaio" if ensaio else "dev")
    pasta.mkdir(parents=True, exist_ok=True)
    textos, gold = conjuntos.carregar("dev")
    arqs = {}
    for nome, ext in (("gama", carregar("neural-cru", "/models")), ("regua", carregar("regua"))):
        arq = arqs[nome] = pasta / f"spans_{nome}.jsonl"
        with arq.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"meta": {"extrator": nome, "modelo": "/models (vinimlo/gama@5f924ca)" if nome == "gama"
                                          else "regras", "dispositivo": "cpu (container, 4 vCPUs)"}}) + "\n")
            for d, t in textos.items():
                t0 = time.perf_counter()
                ss = ext.extrair(t)
                fh.write(json.dumps({"conjunto": "dev", "id": d, "segundos": round(time.perf_counter() - t0, 4),
                                     "spans": [[s.inicio, s.fim, s.tipo, s.forma, s.digitos, s.confianca] for s in ss]},
                                    ensure_ascii=False) + "\n")
    lido = {n: pontuar._ler_spans(arq)[1] for n, arq in arqs.items()}
    # conferência com a gravação antiga do dev (saidas/bench/gama_dev.jsonl), se existir
    antigo = pontuar.SAIDA / "gama_dev.jsonl"
    conf_antigo = None
    if antigo.exists():
        _, la = pontuar._ler_spans(antigo)
        chave = lambda ss: [tuple(s[:5]) for s in ss]  # noqa: E731
        conf_antigo = {"arquivo": vn._rel(antigo), "docs_diferentes": sum(
            chave(la[("dev", d)]["spans"]) != chave(lido["gama"][("dev", d)]["spans"]) for d in textos if ("dev", d) in la)}
    docs, estados = {}, {}
    for d, t in textos.items():
        G = avaliar._aparados([avaliar._span(t, s) for s in lido["gama"][("dev", d)]["spans"]], t)
        R = avaliar._aparados([avaliar._span(t, s) for s in lido["regua"][("dev", d)]["spans"]], t)
        docs[d] = ([vn.Cand(f"dev/{d}/gama/{s.inicio}-{s.fim}", "gama", s) for s in G],
                   [vn.Cand(f"dev/{d}/regua/{s.inicio}-{s.fim}", "regua", s) for s in R])
        for c in docs[d][0] + docs[d][1]:
            estados[c.cid] = vn.estado(t, c.span.inicio, c.span.fim)
    distintos = list(dict.fromkeys(estados.values()))
    pont = _pontuador(pathlib.Path(a.modelo) if ensaio else MODELO)
    t0 = time.perf_counter()
    sc, trunc = pont(distintos)
    seg = time.perf_counter() - t0
    por_estado = dict(zip(distintos, sc))
    score = {cid: por_estado[e] for cid, e in estados.items()}
    with (pasta / "scores.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": {"verificador": NOME, "modelo": str(pont.dir), "precisao": "fp32 cpu"}}) + "\n")
        for cid, s in score.items():
            fh.write(json.dumps({"cid": cid, "score": s}) + "\n")

    prod = vn.producao(docs)
    ref = avaliar.com_guarda({d: [c.span for c in G] for d, (G, R) in docs.items()},
                             {d: [c.span for c in R] for d, (G, R) in docs.items()}, vn.FORTE)
    idx = construir(pontuar.DB)

    def nota(rotulo: str, finais: dict) -> dict:
        antes = pontuar.SAIDA
        pontuar.SAIDA = pasta
        try:
            return pontuar.oficial("dev", rotulo, textos, finais, idx)
        finally:
            pontuar.SAIDA = antes

    def extr(finais: dict) -> dict:
        r = pontuar.extracao(gold, {d: [(s.inicio, s.fim, pontuar.rotulo(s)) for s in ss] for d, ss in finais.items()},
                             juntar_vaga=False)
        return {"f1": r["f1"], "por_tipo": r["por_tipo"]}

    pares = [(score[c.cid], int(vn.casa(c.span, gold.get(d, []), False))) for d, (G, R) in docs.items() for c in G + R]
    if ensaio:
        rel = {"politicas": {p: {"tau": float(t)} for p, t in (x.split("=") for x in a.taus.split(","))}}
    else:
        rel = json.loads((SAIDA / f"{NOME}.json").read_text(encoding="utf-8"))
    out = {"documentos": len(textos), "candidatos": {"gama": sum(len(G) for G, _ in docs.values()),
                                                     "regua": sum(len(R) for _, R in docs.values()),
                                                     "estados_distintos": len(distintos), "truncados": trunc},
           "gama_local_x_gravacao_antiga": conf_antigo,
           "producao_igual_guarda_do_harness": sum(vn._chave(prod[d]) != vn._chave(ref[d]) for d in docs) == 0,
           "cpu": {"segundos": round(seg, 2), "ms_por_estado": round(1000 * seg / max(1, len(distintos)), 1),
                   "ms_por_candidato": round(1000 * seg / max(1, len(score)), 1)},
           "auroc": {"candidatos": len(pares), "positivos": sum(y for _, y in pares), "auroc": vn.auroc(pares),
                     "rotulo": "casa com o gabarito do dev (mesmo tipo, IoU >= 0,5, VAGA separada)"},
           "producao": {"extracao": extr(prod), "oficial": nota("producao_guarda@0.95", prod)},
           "por_politica": {}}   # não "politicas": o refazer do CLI relê todo *.json que tem essa chave
    for pol, res in rel["politicas"].items():
        tau = res["tau"]
        finais = vn.aplicar(pol, docs, score, tau)
        mudados = [d for d in docs if vn._chave(finais[d]) != vn._chave(prod[d])]
        por_tau = {f"{t:.2f}": sum(vn._chave(ss) != vn._chave(prod[d]) for d, ss in vn.aplicar(pol, docs, score, t).items())
                   for t in vn.GRADE}
        out["por_politica"][pol] = {"tau": tau, "docs_mudados_vs_producao": len(mudados), "docs_mudados": mudados,
                                 "docs_mudados_por_tau": por_tau, "extracao": extr(finais),
                                 "oficial": nota(f"{NOME}__{pol}@{tau:.2f}", finais)}
        print(pol, tau, "mudados", len(mudados), "oficial", out["por_politica"][pol]["oficial"].get("final"), flush=True)
    print("produção oficial", out["producao"]["oficial"].get("final"), "AUROC", out["auroc"], flush=True)
    _escrever(pasta / "dev.json" if ensaio else SAIDA / "dev.json", out)


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.controles.verificador_treinado.laya_ft",
                                 description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="acao", required=True)
    for nome in ("politicas", "refazer", "auroc"):
        sub.add_parser(nome)
    p = sub.add_parser("dev")
    p.add_argument("--modelo", help="ensaio: outro checkpoint (ex.: o Laya base) em vez de _modelo")
    p.add_argument("--taus", default="A=0.5,B=0.5,C=0.5", help="ensaio: taus por política")
    p = sub.add_parser("paridade")
    p.add_argument("--passo", type=int, default=50, help="um candidato a cada N de candidatos.jsonl")
    a = ap.parse_args()
    SAIDA.mkdir(parents=True, exist_ok=True)
    {"politicas": cmd_politicas, "refazer": cmd_refazer, "auroc": cmd_auroc, "paridade": cmd_paridade,
     "dev": cmd_dev}[a.acao](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
