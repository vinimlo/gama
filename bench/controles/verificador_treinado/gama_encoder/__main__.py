# -*- coding: utf-8 -*-
"""Avaliação local do verificador "gama_encoder" (container, CPU). O treino e a pontuação dos
13.664 candidatos rodam no HF Jobs (`job.py`); aqui só entram os arquivos gravados.

    docker compose run --rm gama python -m bench.controles.verificador_treinado.gama_encoder <acao> [...]

    rodar --scores S        políticas A, B e C pelo CLI de `bench.controles.verificador` (`cmd_rodar`, sem
                            alteração), com tau escolhido nas 305 e medido nas 172 e no estresse. A única
                            diferença é a pasta de saída: `nucleo.SAIDA` aponta para a deste candidato, então
                            relatório, spans finais e JSON da métrica oficial ficam em
                            saidas/bench/controles/verificador_treinado/gama_encoder/.
    refazer                 `cmd_refazer` do mesmo CLI, na mesma pasta: refaz o relatório do zero a partir dos
                            arquivos gravados e repassa os spans finais pelo harness (`avaliar.avaliar`).
    tabela                  `cmd_tabela` -> resumo.json.
    dev --modelo P          checagem do DEV, só local, em CPU: Gama v1.3 (neural-cru, /models) e régua nos 26
                            documentos, candidatos no formato do verificador, scores deste modelo (pasta local
                            P), e quantos documentos cada política muda em relação à produção (a guarda de
                            produção com 0,95), no tau de cada política escolhido nas 305 e na grade inteira;
                            F1 de extração (VAGA separada) e nota oficial do dev. -> dev/dev.json
    cpu --modelo P          os mesmos pesos em CPU contra os scores do job (GPU) numa amostra sorteada
                            (semente 0) de candidatos de cada conjunto: maior diferença e se algum candidato
                            troca de lado em algum tau escolhido. -> cpu/cpu.json
    auroc --validacao V     tabela de AUROC: validação interna (scores do job, `V`), 305, 172 e estresse
                            (do relatório) -> auroc.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

from bench.controles import avaliar
from bench.controles.verificador import __main__ as vcli
from bench.controles.verificador import nucleo as vn

SAIDA = avaliar.SAIDA / "verificador_treinado" / "gama_encoder"
NOME = "gama_encoder"
RELATORIO = SAIDA / f"{NOME}.json"
PASTA_JOB = pathlib.Path(__file__).resolve().parent


def _redirecionar() -> None:
    """Toda escrita do CLI do verificador (relatório, spans, JSON oficiais) vai para esta pasta."""
    SAIDA.mkdir(parents=True, exist_ok=True)
    vn.SAIDA = SAIDA


def _escrever(destino: pathlib.Path, obj) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", destino, flush=True)


def cmd_rodar(a) -> None:
    _redirecionar()
    ns = argparse.Namespace(candidatos=str(vn.CANDIDATOS), nome=NOME, scores=a.scores, campo="score", v0=False,
                            politicas="A,B,C", sem_oficial=False, saida=None)
    vcli.cmd_rodar(ns)


def cmd_refazer(a) -> None:
    _redirecionar()
    vcli.cmd_refazer(argparse.Namespace())


def cmd_tabela(a) -> None:
    _redirecionar()
    vcli.cmd_tabela(argparse.Namespace())


# ---------------------------------------------------------------- modelo local (CPU)

def _job():
    sys.path.insert(0, str(PASTA_JOB))
    import job  # noqa: E402  (o script UV; só as funções puras e o torch/transformers do container)
    return job


def _carregar(pasta: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(pasta)
    model = AutoModelForSequenceClassification.from_pretrained(pasta).eval()
    return tok, model


def _pontuar(tok, model, estados: list, lote: int = 16) -> tuple[list, float]:
    import torch
    job = _job()
    ids, trunc = job.tokenizar(tok, estados)
    if trunc:
        raise SystemExit(f"{trunc} estados truncados")
    t0 = time.perf_counter()
    p = job.prever(model, ids, tok.pad_token_id, torch.device("cpu"), lote)
    return p, time.perf_counter() - t0


def _taus() -> dict:
    rel = json.loads(RELATORIO.read_text(encoding="utf-8"))
    return {pol: r["tau"] for pol, r in rel["politicas"].items()}


# ---------------------------------------------------------------- dev

def cmd_dev(a) -> None:
    """Só local. Nada do dev sai da máquina: textos, spans e scores ficam em dev/."""
    from gama.extratores import carregar
    from gama.indice import construir
    from gama.pipeline import aparar
    from bench import conjuntos, pontuar
    from bench.controles.verificador_treinado.dados import nucleo as dn

    import gc
    from gama.extratores.guarda import ExtratorGuardado

    taus = _taus()
    textos, gold = conjuntos.carregar("dev")
    neural, regua = carregar("neural-cru", "/models"), carregar("regua")
    # o extrator de produção com as MESMAS instâncias (um Gama só na memória do container, 4 GB)
    prod_ext = ExtratorGuardado.__new__(ExtratorGuardado)
    prod_ext.neural, prod_ext.regua = neural, regua
    docs, regs, prod, iguais_producao = {}, [], {}, 0
    t_ext = 0.0
    for d, t in textos.items():
        t0 = time.perf_counter()
        g_raw, r_raw = neural.extrair(t), regua.extrair(t)
        t_ext += time.perf_counter() - t0
        lista = lambda ss: [[s.inicio, s.fim, s.tipo, s.forma, s.digitos, s.confianca] for s in ss]  # noqa: E731
        G = avaliar._aparados([avaliar._span(t, s) for s in lista(g_raw)], t)      # como o harness lê
        R = avaliar._aparados([avaliar._span(t, s) for s in lista(r_raw)], t)
        rs = dn.candidatos_doc("dev", d, t, G, R)
        regs += rs
        docs[d] = ([vn.Cand(r["cid"], "gama", r["_span"]) for r in rs if r["origem"] == "gama"],
                   [vn.Cand(r["cid"], "regua", r["_span"]) for r in rs if r["origem"] == "regua"])
        prod[d] = vn.politica_v0(*docs[d], vn.FORTE)
        # a saída do extrator de produção (ExtratorGuardado, aparado) = guarda@0,95 sobre os candidatos
        ext = [s for s in (aparar(x, t) for x in prod_ext.extrair(t)) if s]
        iguais_producao += vn._chave(ext) == vn._chave(prod[d])
    del neural, prod_ext
    gc.collect()
    tok, model = _carregar(a.modelo)
    sc_lista, seg = _pontuar(tok, model, [r["estado"] for r in regs])
    sc = {r["cid"]: s for r, s in zip(regs, sc_lista)}
    rot = {r["cid"]: int(vn.casa(r["_span"], gold[r["doc"]], False)) for r in regs}   # VAGA separada, como no estresse
    idx = construir(pontuar.DB)
    antes = pontuar.SAIDA
    pontuar.SAIDA = SAIDA / "dev"
    try:
        def oficial(rotulo, finais):
            return pontuar.oficial("dev", rotulo, textos, finais, idx)
        trip = lambda sp: {d: [(s.inicio, s.fim, pontuar.rotulo(s)) for s in ss] for d, ss in sp.items()}  # noqa: E731
        out = {"documentos": len(textos), "candidatos": len(regs),
               "candidatos_por_origem": {o: sum(r["origem"] == o for r in regs) for o in ("gama", "regua")},
               "elegiveis": {p: sum(r["elegivel"][p] for r in regs) for p in ("A", "B")},
               "positivos": sum(rot.values()), "auroc_dev": vn.auroc([(sc[c], rot[c]) for c in sc]),
               "extrator_producao_igual_guarda_095": {"docs": len(textos), "iguais": iguais_producao},
               "modelo": a.modelo, "segundos_extracao_cpu": round(t_ext, 1),
               "ms_por_candidato_cpu": round(1000 * seg / max(1, len(regs)), 1),
               "producao": {"extracao_f1": pontuar.extracao(gold, trip(prod), False)["f1"],
                            "oficial": oficial("producao_guarda@0.95", prod)["final"]},
               "politicas": {}}
        for pol, tau in taus.items():
            finais = vn.aplicar(pol, docs, sc, tau)
            mud = [d for d in finais if vn._chave(finais[d]) != vn._chave(prod[d])]
            r = {"tau_escolhido_nas_305": tau, "docs_mudados_vs_producao": len(mud),
                 "extracao_f1": pontuar.extracao(gold, trip(finais), False)["f1"],
                 "docs_mudados_por_tau": {f"{t:.2f}": sum(vn._chave(ss) != vn._chave(prod[d])
                                                          for d, ss in vn.aplicar(pol, docs, sc, t).items())
                                          for t in vn.GRADE},
                 "mudancas": {d: {"sai": [list(x) for x in vn._chave(prod[d]) if x not in vn._chave(finais[d])],
                                  "entra": [list(x) for x in vn._chave(finais[d]) if x not in vn._chave(prod[d])]}
                              for d in mud}}
            r["oficial"] = oficial(f"{NOME}__{pol}@{tau:.2f}", finais)["final"] if mud else out["producao"]["oficial"]
            r["candidata_a_uso_no_dev"] = not mud
            out["politicas"][pol] = r
            print(pol, "tau", tau, "docs mudados", len(mud), "F1", r["extracao_f1"], "oficial", r["oficial"], flush=True)
        eleg_c = [(sc[c], rot[c]) for c in sc]
        out["score_minimo_positivos_dev"] = min((s for s, y in eleg_c if y), default=None)
        out["score_maximo_negativos_dev"] = max((s for s, y in eleg_c if not y), default=None)
    finally:
        pontuar.SAIDA = antes
    _escrever(SAIDA / "dev" / "dev.json", out)
    with (SAIDA / "dev" / "scores_dev.jsonl").open("w", encoding="utf-8") as fh:
        for r in regs:
            fh.write(json.dumps({"cid": r["cid"], "score": sc[r["cid"]], "rotulo": rot[r["cid"]],
                                 "forte": r["forte"], "elegivel": r["elegivel"]}) + "\n")


# ---------------------------------------------------------------- CPU x GPU

def cmd_cpu(a) -> None:
    meta, sc_gpu = vn.ler_scores(a.scores)
    taus = _taus()
    linhas = [json.loads(x) for x in vn.CANDIDATOS.read_text(encoding="utf-8").splitlines()]
    regs = [r for r in linhas if "meta" not in r]
    rng = random.Random(0)
    amostra = []
    for c in vn.CONJUNTOS:
        cs = [r for r in regs if r["conjunto"] == c]
        amostra += rng.sample(cs, min(a.n, len(cs)))
    tok, model = _carregar(a.modelo)
    p, seg = _pontuar(tok, model, [r["estado"] for r in amostra])
    difs = [abs(x - sc_gpu[r["cid"]]) for r, x in zip(amostra, p)]
    troca = {pol: sum((x >= t) != (sc_gpu[r["cid"]] >= t) for r, x in zip(amostra, p)) for pol, t in taus.items()}
    out = {"amostra": len(amostra), "por_conjunto": a.n, "semente": 0, "modelo_local": a.modelo,
           "modelo_job": meta.get("modelo"), "max_dif_score": max(difs), "media_dif_score": sum(difs) / len(difs),
           "trocas_de_lado_no_tau": troca, "ms_por_candidato_cpu": round(1000 * seg / len(amostra), 1),
           "dispositivo": "cpu (container, 4 vCPUs), fp32, lote 16"}
    print(out, flush=True)
    _escrever(SAIDA / "cpu" / "cpu.json", out)


# ---------------------------------------------------------------- AUROC

def cmd_auroc(a) -> None:
    rel = json.loads(RELATORIO.read_text(encoding="utf-8"))
    linhas = [json.loads(x) for x in pathlib.Path(a.validacao).read_text(encoding="utf-8").splitlines()]
    val = [r for r in linhas if "meta" not in r]
    out = {"validacao_interna": {"candidatos": len(val), "positivos": sum(r["rotulo"] for r in val),
                                 "auroc": vn.auroc([(r["score"], r["rotulo"]) for r in val])}}
    for c, nome in (("reais", "305"), ("novas", "172"), ("estresse", "estresse")):
        out[nome] = rel["auroc"][c]
    meta_val = next(r["meta"] for r in linhas if "meta" in r)
    out["validacao_interna"]["por_subconjunto_job"] = meta_val.get("metricas")
    print(json.dumps({k: (v if k != "validacao_interna" else {kk: vv for kk, vv in v.items()
                                                                if kk != "por_subconjunto_job"})
                      for k, v in out.items()}, ensure_ascii=False), flush=True)
    _escrever(SAIDA / "auroc.json", out)


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.controles.verificador_treinado.gama_encoder",
                                 description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="acao", required=True)
    p = sub.add_parser("rodar")
    p.add_argument("--scores", required=True)
    sub.add_parser("refazer")
    sub.add_parser("tabela")
    p = sub.add_parser("dev")
    p.add_argument("--modelo", required=True)
    p = sub.add_parser("cpu")
    p.add_argument("--modelo", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--n", type=int, default=100)
    p = sub.add_parser("auroc")
    p.add_argument("--validacao", required=True)
    a = ap.parse_args()
    {"rodar": cmd_rodar, "refazer": cmd_refazer, "tabela": cmd_tabela, "dev": cmd_dev, "cpu": cmd_cpu,
     "auroc": cmd_auroc}[a.acao](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
