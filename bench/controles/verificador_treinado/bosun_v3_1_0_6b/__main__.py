# -*- coding: utf-8 -*-
"""Pontuação local do candidato bosun_v3_1_0_6b. Tudo no container:

    docker compose run --rm gama python -m bench.controles.verificador_treinado.bosun_v3_1_0_6b <acao> [...]

    rodar --scores S      políticas A, B e C pelo CLI do verificador (`verificador.__main__.cmd_rodar`, sem
                          alteração): tau escolhido nas 305, medido nas 172 e no estresse (docs mudados e
                          nota oficial), AUROC nos três conjuntos. A única diferença é a pasta de saída: o
                          `SAIDA` do núcleo do verificador aponta para a pasta deste candidato, então spans
                          finais e JSON oficiais ficam em saidas/bench/controles/verificador_treinado/<nome>/
    validacao --scores S  AUROC na validação interna (scores do job de inferência, rótulos de validacao.jsonl)
    dev --modelo P        checagem no dev (26 docs da organização), em CPU, aqui: Gama v1.3 (neural-cru,
                          /models) e régua, candidatos pelo construtor dos dados, score pelo modelo fundido
                          (bf16), políticas com o tau escolhido nas 305; docs mudados contra a produção
                          (guarda@0,95) e nota oficial no dev. Só contagens saem do dev.
    refazer               refaz tudo a partir dos arquivos gravados (scores, spans do dev, relatórios) em
                          refeito/ e compara; repassa os spans finais pelo harness (`avaliar.avaliar`, cru)
    resumo                junta os relatórios em resumo.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import time

from bench import conjuntos, pontuar
from bench.controles import avaliar
from bench.controles.verificador import __main__ as vcli
from bench.controles.verificador import nucleo as vn
from bench.controles.verificador_treinado.dados import nucleo as dn

from . import job as jb

NOME = jb.NOME
SAIDA = avaliar.SAIDA / "verificador_treinado" / NOME     # /app/saidas/bench/controles/verificador_treinado/<nome>
HUB = SAIDA / "_hub"
SCORES = HUB / "bench/saida/controles/verificador_treinado" / f"{NOME}.jsonl"
SCORES_VALIDACAO = HUB / "scores" / "validacao.jsonl"
MODELO = SAIDA / "modelo"
POLITICAS = SAIDA / "politicas.json"
VALIDACAO = SAIDA / "validacao.json"
DEV = SAIDA / "dev.json"
DEV_SPANS = SAIDA / "dev_spans.jsonl"
DEV_SCORES = SAIDA / "dev_scores.jsonl"
REFEITO = SAIDA / "refeito"


def _escrever(destino: pathlib.Path, obj) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", destino, flush=True)


@contextlib.contextmanager
def _trocar(modulo, atributo: str, valor):
    antes = getattr(modulo, atributo)
    setattr(modulo, atributo, valor)
    try:
        yield
    finally:
        setattr(modulo, atributo, antes)


# ---------------------------------------------------------------- políticas (CLI do verificador)

def rodar(scores: pathlib.Path, saida: pathlib.Path) -> dict:
    ns = argparse.Namespace(candidatos=str(vn.CANDIDATOS), nome=NOME, scores=str(scores), campo="score", v0=False,
                            politicas="A,B,C", sem_oficial=False, saida=str(saida))
    with _trocar(vn, "SAIDA", SAIDA):
        vcli.cmd_rodar(ns)
    return json.loads(saida.read_text(encoding="utf-8"))


def cmd_rodar(a) -> None:
    rodar(pathlib.Path(a.scores), POLITICAS)


# ---------------------------------------------------------------- validação interna

def _validacao(scores: pathlib.Path) -> dict:
    meta, sc = vn.ler_scores(scores)
    regs = dn.jsonl(dn.SAIDA / "validacao.jsonl")
    falta = [r["cid"] for r in regs if r["cid"] not in sc]
    if falta:
        raise SystemExit(f"{len(falta)} exemplos da validação sem score")

    def au(f):
        pares = [(sc[r["cid"]], r["rotulo"]) for r in regs if f(r)]
        return {"n": len(pares), "positivos": sum(y for _, y in pares), "auroc": vn.auroc(pares)}
    out = {"scores": {"arquivo": dn.rel(scores), "sha256": vn.sha256(scores), "modelo": meta.get("modelo")},
           "rotulos": {"arquivo": dn.rel(dn.SAIDA / "validacao.jsonl"), "sha256": vn.sha256(dn.SAIDA / "validacao.jsonl")},
           "todos": au(lambda r: True)}
    for c in ("destilacao", "final_v3"):
        out[c] = au(lambda r, c=c: r["conjunto"] == c)
        for o in ("gama", "regua", "borda"):
            if any(r["conjunto"] == c and r["origem"] == o for r in regs):
                out[f"{c}/{o}"] = au(lambda r, c=c, o=o: r["conjunto"] == c and r["origem"] == o)
    for e in ("A", "B"):
        out[f"destilacao/elegivel_{e}"] = au(lambda r, e=e: r["conjunto"] == "destilacao" and r["elegivel"][e])
    out["macro"] = round((out["destilacao"]["auroc"] + out["final_v3"]["auroc"]) / 2, 4)
    return out


def cmd_validacao(a) -> None:
    out = _validacao(pathlib.Path(a.scores))
    print(json.dumps({k: v for k, v in out.items() if k not in ("scores", "rotulos")}, ensure_ascii=False), flush=True)
    _escrever(VALIDACAO, out)


# ---------------------------------------------------------------- dev (CPU, local)

def _dev_spans() -> None:
    """Gama v1.3 (neural-cru, /models) e régua no dev, gravados no formato do harness (só offsets)."""
    from gama.extratores import CatalogoDeExtratores
    textos, _ = conjuntos.carregar("dev")
    with DEV_SPANS.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": {"gama": dn.MODELO_GAMA, "regua": "regras", "dispositivo": "cpu (container, 4 vCPUs)",
                                      "nota": "spans crus do dev; só offsets (o texto fica em dados/)"}}) + "\n")
        catalogo = CatalogoDeExtratores("/models")
        for nome, ext in (("gama", catalogo.carregar("neural-cru")), ("regua", catalogo.carregar("regua"))):
            for d, t in textos.items():
                fh.write(json.dumps({"conjunto": f"dev_{nome}", "id": d, "spans": [
                    [s.inicio, s.fim, s.tipo, s.forma, s.digitos, s.confianca] for s in ext.extrair(t)]},
                    ensure_ascii=False) + "\n")
            del ext


def _dev_candidatos() -> tuple[dict, dict, list]:
    """(textos, {doc: (G, R) de Cand}, registros) a partir de DEV_SPANS, pelo construtor dos dados."""
    textos, _ = conjuntos.carregar("dev")
    _, linhas = dn.ler_spans(DEV_SPANS)
    docs, regs = {}, []
    for d, t in textos.items():
        G = dn.aparados(t, linhas[("dev_gama", d)]["spans"])
        R = dn.aparados(t, linhas[("dev_regua", d)]["spans"])
        rs = dn.candidatos_doc("dev", d, t, G, R)
        regs += rs
        docs[d] = ([vn.Cand(r["cid"], "gama", r["_span"]) for r in rs if r["origem"] == "gama"],
                   [vn.Cand(r["cid"], "regua", r["_span"]) for r in rs if r["origem"] == "regua"])
    return textos, docs, regs


def _dev_scores(regs: list, modelo: pathlib.Path, dtype: str, threads: int) -> dict:
    """Pesos bf16 do Hub; em CPU a conta vai em fp32 por padrão (bf16 no CPU do container é ~3x mais lento)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(jb.SEMENTE)
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(modelo)
    m = AutoModelForSequenceClassification.from_pretrained(modelo, dtype=getattr(torch, dtype)).eval()
    estados = list(dict.fromkeys(r["estado"] for r in regs))
    seqs = [jb.codificar(tok, e)[0] for e in estados]
    with torch.inference_mode():
        jb.logits(m, seqs[:4], m.config.pad_token_id, 4, "cpu", False)       # aquecimento
        t0 = time.perf_counter()
        z = jb.logits(m, seqs, m.config.pad_token_id, 8, "cpu", False)
        dt = time.perf_counter() - t0
    por_estado = dict(zip(estados, z))
    tempo = {"candidatos": len(regs), "estados_distintos": len(estados), "segundos": round(dt, 2),
             "ms_por_candidato": round(1000 * dt / len(regs), 1), "ms_por_estado": round(1000 * dt / len(estados), 1),
             "lote": 8, "dispositivo": "cpu (container, 4 vCPUs)", "dtype": dtype,
             "threads": torch.get_num_threads()}
    import math
    sc = {r["cid"]: 1.0 / (1.0 + math.exp(-por_estado[r["estado"]])) for r in regs}
    vn.gravar_scores(DEV_SCORES, {"verificador": NOME, "modelo": str(modelo), "tempo": tempo,
                                  "nota": "scores dos candidatos do dev (cid = dev/<doc>/<origem>/<inicio>-<fim>)"}, sc)
    return tempo


def _dev_avaliar(textos: dict, docs: dict, sc: dict, taus: dict, idx) -> dict:
    _, gold = conjuntos.carregar("dev")
    prod = vn.producao(docs)
    out = {"docs": len(textos), "candidatos": sum(len(G) + len(R) for G, R in docs.values())}
    pares = [(sc[x.cid], int(vn.casa(x.span, gold[d], False))) for d, (G, R) in docs.items() for x in G + R]
    out["auroc"] = {"candidatos": len(pares), "positivos": sum(y for _, y in pares), "auroc": vn.auroc(pares)}
    with _trocar(pontuar, "SAIDA", SAIDA):
        out["producao"] = {"oficial": pontuar.oficial("dev", f"{NOME}__producao", textos, prod, idx)["final"],
                           "f1": pontuar.extracao(gold, avaliar.triplas(prod), False)["f1"]}
        out["politicas"] = {}
        for pol, tau in taus.items():
            finais = vn.aplicar(pol, docs, sc, tau)
            mud = [d for d in finais if vn._chave(finais[d]) != vn._chave(prod[d])]
            r = {"tau": tau, "docs_mudados_vs_producao": len(mud),
                 "spans_a_mais": sum(max(0, len(finais[d]) - len(prod[d])) for d in mud),
                 "spans_a_menos": sum(max(0, len(prod[d]) - len(finais[d])) for d in mud),
                 "f1": pontuar.extracao(gold, avaliar.triplas(finais), False)["f1"],
                 "oficial": pontuar.oficial("dev", f"{NOME}__{pol}@{tau:.2f}", textos, finais, idx)["final"],
                 "docs_mudados_por_tau": {f"{t:.2f}": sum(vn._chave(ss) != vn._chave(prod[d]) for d, ss in
                                                          vn.aplicar(pol, docs, sc, t).items()) for t in vn.GRADE}}
            out["politicas"][pol] = r
            print("dev", pol, {k: v for k, v in r.items() if k != "docs_mudados_por_tau"}, flush=True)
    return out


def _taus(relatorio: pathlib.Path) -> dict:
    r = json.loads(relatorio.read_text(encoding="utf-8"))
    return {p: v["tau"] for p, v in r["politicas"].items()}


def cmd_dev(a) -> None:
    from gama.indice import Indice
    if not DEV_SPANS.exists() or a.reextrair:
        _dev_spans()
    textos, docs, regs = _dev_candidatos()
    if a.so_scores or not DEV_SCORES.exists() or a.repontuar:
        _dev_scores(regs, pathlib.Path(a.modelo), a.dtype, a.threads)
        if a.so_scores:
            return
    meta_sc, sc = vn.ler_scores(DEV_SCORES)
    tempo = meta_sc["tempo"]
    out = {"modelo": str(a.modelo), "taus_de": dn.rel(POLITICAS), "tempo_cpu": tempo,
           "spans": {"arquivo": dn.rel(DEV_SPANS), "sha256": vn.sha256(DEV_SPANS)},
           "scores": {"arquivo": dn.rel(DEV_SCORES), "sha256": vn.sha256(DEV_SCORES)},
           **_dev_avaliar(textos, docs, sc, _taus(POLITICAS), Indice.do_banco(pontuar.DB))}
    _escrever(DEV, out)


# ---------------------------------------------------------------- refazer e resumo

def cmd_refazer(a) -> None:
    from gama.indice import Indice
    antigo = json.loads(POLITICAS.read_text(encoding="utf-8"))
    comp = {}
    sha = vn.sha256(SCORES)
    comp["scores_sha256_igual"] = sha == antigo["scores"]["sha256"]
    novo = rodar(SCORES, REFEITO / "politicas.json")
    difs = [k for k in ("politicas", "auroc", "conferencia_harness", "producao_estresse_oficial")
            if antigo.get(k) != novo.get(k)]
    comp["politicas"] = {"relatorio_refeito_igual": not difs, "campos_diferentes": difs}
    harness = {}
    for pol, res in novo["politicas"].items():
        rel = avaliar.avaliar(f"verificador_{NOME}__{pol}", [pathlib.Path("/app") / res["spans"]["arquivo"]], com_oficial=False)
        cru = rel["variantes"]["cru"]
        harness[pol] = {c: {"f1_harness": cru[c]["extracao"]["f1"], "f1_relatorio": res["conjuntos"][c]["extracao"]["f1"],
                            "ic95_harness": cru[c].get("vs_gama_v13_guarda", {}).get("ic95"),
                            "ic95_relatorio": res["conjuntos"][c].get("vs_gama_v13_guarda", {}).get("ic95")}
                        for c in vn.CONJUNTOS}
        harness[pol]["iguais"] = all(v["f1_harness"] == v["f1_relatorio"] and v["ic95_harness"] == v["ic95_relatorio"]
                                     for v in harness[pol].values() if isinstance(v, dict))
    comp["harness"] = harness
    comp["harness_cru_igual"] = all(h["iguais"] for h in harness.values())
    val_antigo = json.loads(VALIDACAO.read_text(encoding="utf-8"))
    val_novo = _validacao(SCORES_VALIDACAO)
    comp["validacao_igual"] = val_antigo == val_novo
    dev_antigo = json.loads(DEV.read_text(encoding="utf-8"))
    textos, docs, _ = _dev_candidatos()
    _, sc = vn.ler_scores(DEV_SCORES)
    dev_novo = _dev_avaliar(textos, docs, sc, _taus(REFEITO / "politicas.json"), Indice.do_banco(pontuar.DB))
    comp["dev_igual"] = all(dev_antigo[k] == dev_novo[k] for k in ("docs", "candidatos", "auroc", "producao", "politicas"))
    comp["tudo_igual"] = (comp["scores_sha256_igual"] and not difs and comp["harness_cru_igual"]
                          and comp["validacao_igual"] and comp["dev_igual"])
    print(json.dumps({k: v for k, v in comp.items() if k != "harness"}, ensure_ascii=False), flush=True)
    _escrever(REFEITO / "comparacao.json", comp)


def _auroc_elegiveis(scores: pathlib.Path) -> dict:
    """AUROC nos candidatos que cada política decide (elegíveis A e B), nas 305, nas 172 e no estresse."""
    _, docs = vn.ler_candidatos(vn.CANDIDATOS)
    _, sc = vn.ler_scores(scores)
    regs = {r["cid"]: r for r in dn.jsonl(vn.CANDIDATOS) if "cid" in r}
    out = {}
    for c in vn.CONJUNTOS:
        _, gold = avaliar.textos_e_ouro(c)
        for e in ("A", "B"):
            pares = [(sc[x.cid], int(vn.casa(x.span, gold[d], c != "estresse"))) for d, (G, R) in docs[c].items()
                     for x in G + R if regs[x.cid]["elegivel"][e]]
            out[f"{c}/elegivel_{e}"] = {"n": len(pares), "positivos": sum(y for _, y in pares), "auroc": vn.auroc(pares)}
    return out


def cmd_resumo(a) -> None:
    pol = json.loads(POLITICAS.read_text(encoding="utf-8"))
    val = json.loads(VALIDACAO.read_text(encoding="utf-8"))
    dev = json.loads(DEV.read_text(encoding="utf-8"))
    ref = json.loads((REFEITO / "comparacao.json").read_text(encoding="utf-8")) if (REFEITO / "comparacao.json").exists() else {}
    linhas = []
    for p, r in pol["politicas"].items():
        x = dict(r["resumo"])
        x["estresse_oficial_producao"] = pol.get("producao_estresse_oficial", {}).get("final")
        x["dev_docs_mudados"] = dev["politicas"][p]["docs_mudados_vs_producao"]
        x["dev_oficial"] = dev["politicas"][p]["oficial"]
        linhas.append(x)
    out = {"candidato": NOME, "scores": pol["scores"], "auroc": {
               "validacao_interna": {k: val[k] for k in ("todos", "destilacao", "final_v3", "macro",
                                                         "destilacao/elegivel_A", "destilacao/elegivel_B")},
               **{c: pol["auroc"][c] for c in vn.CONJUNTOS}, "dev": dev["auroc"],
               "elegiveis": _auroc_elegiveis(pathlib.Path("/app") / pol["scores"]["arquivo"])},
           "politicas": linhas, "dev": {"producao": dev["producao"], "tempo_cpu": dev["tempo_cpu"]},
           "tempo_gpu": pol["scores"]["meta"].get("tempo"), "refeito": {k: v for k, v in ref.items() if k != "harness"}}
    for x in linhas:
        print(x, flush=True)
    _escrever(SAIDA / "resumo.json", out)


def main() -> int:
    ap = argparse.ArgumentParser(prog=f"python -m bench.controles.verificador_treinado.{NOME}", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="acao", required=True)
    p = sub.add_parser("rodar")
    p.add_argument("--scores", default=str(SCORES))
    p = sub.add_parser("validacao")
    p.add_argument("--scores", default=str(SCORES_VALIDACAO))
    p = sub.add_parser("dev")
    p.add_argument("--modelo", default=str(MODELO))
    p.add_argument("--reextrair", action="store_true")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--so-scores", action="store_true", help="só pontua os candidatos do dev (sem políticas)")
    p.add_argument("--repontuar", action="store_true", help="refaz os scores mesmo se já existirem")
    sub.add_parser("refazer")
    sub.add_parser("resumo")
    a = ap.parse_args()
    SAIDA.mkdir(parents=True, exist_ok=True)
    {"rodar": cmd_rodar, "validacao": cmd_validacao, "dev": cmd_dev, "refazer": cmd_refazer, "resumo": cmd_resumo}[a.acao](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
