# -*- coding: utf-8 -*-
"""Candidato `gliner_ft_verif`: o GLiNER fine-tunado do controle H4 como verificador, SEM treino novo.

O QUE É
    vinimlo/gama-exp-gliner-ft@12f4e18 (urchade/gliner_multi-v2.1 fine-tunado no final_v3 pelo outro
    workflow, job 6abc04f5) já extraiu os três conjuntos do harness (job 6abc17d3, L4). Score de cada
    um dos 13.664 candidatos (saidas/bench/controles/verificador/candidatos.jsonl) = maior score de um
    span previsto pelo GLiNER ft do mesmo tipo (VAGA junto de JURIS) com IoU >= 0,5, senão 0. Quem
    calcula é `verificador.nucleo.scores_de_spans` (o `de-spans` do CLI do verificador), sem alteração.

    Entrada: `modelo_gliner_ft_piso.jsonl` (todos os spans com score >= 0,10; dataset
    vinimlo/gama-goldenset, revisão 19d5323a, oid git conferido abaixo). Escolhido ANTES de ver
    resultado porque guarda o score do GLiNER até 0,10. A outra saída (`modelo_gliner_ft.jsonl`,
    limiar 0,50) é o mesmo arquivo filtrado em 0,50 (o outro workflow conferiu em 20 documentos, 0
    diferentes) e só zeraria quem ficou entre 0,10 e 0,50: com a grade de tau 0,05-0,95, qualquer
    política que ela permite o piso também permite (com tau = 0,50 dá o mesmo). Por isso não entra
    como segunda variante (seria só mais um caminho de escolha).

COMO REUSA O CLI (sem alterá-lo)
    Importa `bench.controles.verificador.__main__` e troca `nucleo.SAIDA` por
    saidas/bench/controles/verificador_treinado/gliner_ft_verif antes de cada chamada: scores,
    relatório, spans finais e JSON oficiais vão para lá, nada para a pasta do experimento anterior.
    Os candidatos continuam lidos de `nucleo.CANDIDATOS` (fixado na importação).

    docker compose run --rm gama python -m bench.controles.verificador_treinado.gliner_ft_verif scores
    docker compose run --rm gama python -m bench.controles.verificador_treinado.gliner_ft_verif rodar
    docker compose run --rm gama python -m bench.controles.verificador_treinado.gliner_ft_verif auroc
    docker compose run --rm gama python -m bench.controles.verificador_treinado.gliner_ft_verif dev
    docker compose run --rm gama python -m bench.controles.verificador_treinado.gliner_ft_verif refazer \\
        --scores-hub saidas/bench/controles/verificador_treinado/gliner_ft_verif/_hub/<caminho no dataset>

    scores    confere o oid do arquivo do GLiNER, grava scores/gliner_ft_verif.jsonl pelo `de-spans` e
              mede o tempo (GLiNER na L4, dos `segundos` gravados, + casamento em CPU) -> tempo.json
    rodar     políticas A, B e C pelo `rodar` (tau nas 305, medido nas 172 e no estresse) -> gliner_ft_verif.json
    auroc     AUROC por conjunto e por grupo de candidatos (origem, forte, elegíveis A/B) -> auroc.json
    dev       só local, CPU: limite de documentos do dev que A/B/C PODEM mudar (o GLiNER ft não tem
              previsão no dev, então a contagem exata não é medida) -> dev.json
    refazer   scores de novo a partir dos spans do GLiNER, `refazer` do CLI (relatório do zero + harness
              cru) e, com --scores-hub, `rodar` a partir do arquivo baixado do Hub -> conferencia_final.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import statistics
import time

from bench.controles import avaliar
from gama.span import aparar_todos
from bench.controles.verificador import __main__ as cli
from bench.controles.verificador import nucleo as nu

NOME = "gliner_ft_verif"
DESTINO = avaliar.SAIDA / "verificador_treinado" / NOME
PISO = avaliar.SAIDA / "_hub" / "bench" / "saida" / "controles" / "gliner_ft" / "modelo_gliner_ft_piso.jsonl"
PISO_HUB = {"repo": "vinimlo/gama-goldenset", "caminho": "bench/saida/controles/gliner_ft/modelo_gliner_ft_piso.jsonl",
            "revisao": "19d5323a17ebc5657833c74fe747c7a717d65e5d", "oid_git": "b6739cea24c151c62053684c27c53e9b94f83a7f"}
SCORES = DESTINO / "scores" / f"{NOME}.jsonl"
DADOS_VERIF = avaliar.SAIDA / "verificador_treinado" / "dados" / "documentos.jsonl"
FINAL_V3_GLINER = pathlib.Path("/app/corpus/goldenset/v3/meta.jsonl")   # 5.410 treino + 590 reserva


def _redirecionar() -> None:
    nu.SAIDA = DESTINO
    DESTINO.mkdir(parents=True, exist_ok=True)


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(candidatos=str(nu.CANDIDATOS), **kw)


def _escrever(nome: str, obj) -> None:
    destino = DESTINO / nome
    destino.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", destino, flush=True)


def oid_git(arq: pathlib.Path) -> str:
    b = pathlib.Path(arq).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def _conferir_entrada() -> dict:
    oid = oid_git(PISO)
    if oid != PISO_HUB["oid_git"]:
        raise SystemExit(f"{PISO} não é o arquivo da revisão {PISO_HUB['revisao']}: oid {oid}")
    return {**PISO_HUB, "local": str(PISO), "oid_local": oid, "sha256": nu.sha256(PISO)}


# ---------------------------------------------------------------- scores e tempo

def cmd_scores(a) -> None:
    _redirecionar()
    entrada = _conferir_entrada()
    cli.cmd_de_spans(_ns(nome=NOME, spans=[str(PISO)]))
    _, docs = nu.ler_candidatos(nu.CANDIDATOS)
    _, salvos = nu.ler_scores(SCORES)
    tempos = []
    for _ in range(3):
        t0 = time.perf_counter()
        _, sc = nu.scores_de_spans(docs, [PISO])
        tempos.append(time.perf_counter() - t0)
        if sc != salvos:
            raise SystemExit("scores recomputados diferentes dos gravados")
    meta_g, linhas = avaliar.ler([PISO])
    por = {}
    for c in nu.CONJUNTOS:
        textos, _ = avaliar.textos_e_ouro(c)
        seg = sum(linhas[(c, d)]["segundos"] for d in textos)
        n = sum(len(G) + len(R) for G, R in docs[c].values())
        por[c] = {"documentos": len(textos), "candidatos": n, "segundos_gliner": round(seg, 3),
                  "ms_gliner_por_candidato": round(1000 * seg / n, 3), "ms_gliner_por_documento": round(1000 * seg / len(textos), 2)}
    n_tot = sum(v["candidatos"] for v in por.values())
    s_tot = sum(v["segundos_gliner"] for v in por.values())
    casar = statistics.median(tempos)
    out = {"entrada": entrada, "gliner": {k: meta_g[PISO.name].get(k) for k in
                                          ("modelo", "dispositivo", "torch", "gliner", "janelas", "limiar_decisao")},
           "por_conjunto": por, "candidatos": n_tot,
           "gliner_segundos_total": round(s_tot, 3), "gliner_ms_por_candidato": round(1000 * s_tot / n_tot, 3),
           "casamento_segundos_mediana_3": round(casar, 3), "casamento_segundos": [round(t, 3) for t in tempos],
           "casamento_ms_por_candidato": round(1000 * casar / n_tot, 4),
           "total_ms_por_candidato": round(1000 * (s_tot + casar) / n_tot, 3),
           "nota": "GLiNER: uma passada por documento (janelas 200/150, lote por documento) na NVIDIA L4 do job "
                   "6abc17d3 do outro workflow, `segundos` gravados no arquivo (não medido aqui); o custo é por "
                   "documento, dividido pelos candidatos. Casamento (aparar os spans previstos + IoU com cada "
                   "candidato): CPU do container (4 núcleos), leitura do arquivo incluída, mediana de 3."}
    _escrever("tempo.json", out)


# ---------------------------------------------------------------- políticas

def cmd_rodar(a) -> None:
    _redirecionar()
    cli.cmd_rodar(_ns(nome=NOME, scores=str(SCORES), campo="score", v0=False, politicas="A,B,C",
                      sem_oficial=False, saida=None))
    cli.cmd_tabela(_ns())


# ---------------------------------------------------------------- AUROC

def _validacao_interna() -> dict:
    docs = [json.loads(x) for x in DADOS_VERIF.read_text(encoding="utf-8").splitlines()]
    split_gliner = {}
    for x in FINAL_V3_GLINER.read_text(encoding="utf-8").splitlines():
        m = json.loads(x)
        split_gliner[m["documento_id"]] = m.get("split", "treino")
    k = collections.Counter((d["conjunto"], d["split"], split_gliner.get(d["doc"], "fora") if d["conjunto"] == "final_v3" else "-")
                            for d in docs)
    return {"medida": False,
            "motivo": "o GLiNER ft não tem previsões nos documentos da validação interna (151 ementas + 50 do "
                      "final_v3); gerá-las exige rodar o GLiNER, o que só pode ser feito no HF Jobs (GLiNER nunca "
                      "em container local; dependência nova nunca no host) e o teto deste candidato é US$ 0",
            "contaminacao_final_v3": "os documentos do final_v3 dos dados do verificador saíram do split treino do "
                                     "final_v3, o mesmo em que o GLiNER ft foi treinado (conferido por id contra "
                                     "corpus/goldenset/v3/meta.jsonl, 5.410 treino + 590 reserva, as contagens do "
                                     "metricas_treino.json do outro workflow): ali o AUROC não seria fora da amostra",
            "documentos_por_conjunto_split_e_split_no_gliner": {"|".join(c): n for c, n in sorted(k.items())}}


def cmd_auroc(a) -> None:
    _redirecionar()
    meta_s, sc = nu.ler_scores(SCORES)
    _, docs = nu.ler_candidatos(nu.CANDIDATOS)
    reg = {}
    for x in nu.CANDIDATOS.read_text(encoding="utf-8").splitlines():
        r = json.loads(x)
        if "meta" not in r:
            reg[r["cid"]] = r
    out = {"scores": {"arquivo": nu._rel(SCORES), "sha256": nu.sha256(SCORES)},
           "rotulo": "casa com o ouro (mesmo tipo, IoU >= 0,5; VAGA junto de JURIS nas ementas, separada no "
                     "estresse), o mesmo de `rodar`", "conjuntos": {}}
    for c in nu.CONJUNTOS:
        _, gold = avaliar.textos_e_ouro(c)
        pares = {}
        for d, (G, R) in docs[c].items():
            for x in G + R:
                pares[x.cid] = (sc[x.cid], int(nu.casa(x.span, gold[d], c != "estresse")))
        grupos = {
            "todos (C)": list(pares),
            "gama": [k for k in pares if reg[k]["origem"] == "gama"],
            "regua": [k for k in pares if reg[k]["origem"] == "regua"],
            "gama_forte (fixos em A e B)": [k for k in pares if reg[k]["forte"]],
            "elegiveis_A": [k for k in pares if reg[k]["elegivel"]["A"]],
            "elegiveis_B": [k for k in pares if reg[k]["elegivel"]["B"]],
            "so_B (regua sem Gama)": [k for k in pares if reg[k]["elegivel"]["B"] and not reg[k]["elegivel"]["A"]],
        }
        res = {}
        for g, ks in grupos.items():
            ps = [pares[k] for k in ks]
            pos = [s for s, y in ps if y]
            neg = [s for s, y in ps if not y]
            res[g] = {"candidatos": len(ps), "positivos": len(pos), "auroc": nu.auroc(ps) if ps else None,
                      "frac_score_zero_pos": round(sum(s == 0 for s in pos) / len(pos), 4) if pos else None,
                      "frac_score_zero_neg": round(sum(s == 0 for s in neg) / len(neg), 4) if neg else None,
                      "frac_score_ge_050_neg": round(sum(s >= 0.5 for s in neg) / len(neg), 4) if neg else None}
        out["conjuntos"][c] = res
        print(c, {g: (v["candidatos"], v["auroc"]) for g, v in res.items()}, flush=True)
    out["validacao_interna"] = _validacao_interna()
    _escrever("auroc.json", out)


# ---------------------------------------------------------------- dev (limite, só local)

def cmd_dev(a) -> None:
    """O GLiNER ft não tem previsão no dev (26 documentos da organização: não sai da máquina e o GLiNER
    não roda em container local), então os documentos mudados no dev NÃO são medidos. Aqui, só com os
    spans do Gama e da régua já gravados no dev (CPU, container), o limite: um documento só pode mudar
    em A se tiver candidato elegível em A (W + X) e em B se tiver elegível em B (W + X + régua sem
    Gama), qualquer que seja o score. E quantos desses elegíveis casam com o gabarito do dev (VAGA
    separada, como no estresse)."""
    _redirecionar()
    from bench import conjuntos, pontuar
    textos, gold = conjuntos.carregar("dev")
    fontes = {n: pontuar.SAIDA / f"{n}_dev.jsonl" for n in ("gama", "modelo_base12", "regua")}
    lido = {n: pontuar._ler_spans(p)[1] for n, p in fontes.items()}

    def sp(n, d):
        t = textos[d]
        return aparar_todos([avaliar._span(t, s) for s in lido[n][("dev", d)]["spans"]], t)

    chave = lambda ss: [(s.inicio, s.fim, s.tipo, s.forma) for s in ss]  # noqa: E731
    out = {"fontes": {n: {"arquivo": nu._rel(p), "sha256": nu.sha256(p)} for n, p in fontes.items()},
           "gama_dev_igual_base12_dev": sum(chave(sp("gama", d)) != chave(sp("modelo_base12", d)) for d in textos) == 0,
           "docs": len(textos), "por_doc": {}}
    k = collections.Counter()
    for d in textos:
        G, R = sp("gama", d), sp("regua", d)
        _, f, w, x = nu.grupos(G, R)
        ids = {id(s) for s in w + x}
        ea = [s for s in G + R if id(s) in ids]
        sob = [s for s in R if not any(nu._cruza(s, g) for g in G)]
        pos = lambda ss: sum(nu.casa(s, gold[d], False) for s in ss)  # noqa: E731
        out["por_doc"][d] = {"elegiveis_A": len(ea), "elegiveis_A_casam_ouro": pos(ea),
                             "so_B": len(sob), "so_B_casam_ouro": pos(sob), "candidatos": len(G) + len(R)}
        k["docs_podem_mudar_A"] += bool(ea)
        k["docs_podem_mudar_B"] += bool(ea or sob)
        k["docs_podem_mudar_C"] += bool(G or R)
        k["elegiveis_A"] += len(ea)
        k["elegiveis_A_casam_ouro"] += pos(ea)
        k["so_B"] += len(sob)
        k["so_B_casam_ouro"] += pos(sob)
        # as funções de política do verificador com scores extremos (todos 0 e todos 1) contra a produção
        Gc = [nu.Cand(f"dev/{d}/gama/{s.inicio}-{s.fim}", "gama", s) for s in G]
        Rc = [nu.Cand(f"dev/{d}/regua/{s.inicio}-{s.fim}", "regua", s) for s in R]
        prod = chave(nu.politica_v0(Gc, Rc, nu.FORTE))
        for pol in ("A", "B", "C"):
            for v in (0.0, 1.0):
                sc = {c.cid: v for c in Gc + Rc}
                k[f"docs_mudados_{pol}_score_{v:.0f}"] += chave(nu.politica(pol, Gc, Rc, sc, 0.5)) != prod
    out["limite"] = dict(k)
    out["nota"] = ("sem candidato elegível, a política é a produção com qualquer score (confirmado com scores "
                   "extremos pelas funções de política do verificador); com elegível, os documentos mudados de verdade "
                   "dependem do score do GLiNER ft, que não existe no dev (limite superior, não medida); "
                   "'casam_ouro' diz se aceitar o candidato ajudaria (1) ou atrapalharia (0)")
    print(out["gama_dev_igual_base12_dev"], out["limite"], flush=True)
    _escrever("dev.json", out)


# ---------------------------------------------------------------- refazer

def _spans_sha() -> dict:
    return {p.name: nu.sha256(p) for p in sorted((DESTINO / "spans").glob("*.jsonl"))}


def cmd_refazer(a) -> None:
    _redirecionar()
    res = {"entrada": _conferir_entrada()}
    _, docs = nu.ler_candidatos(nu.CANDIDATOS)
    _, sc = nu.scores_de_spans(docs, [PISO])
    _, salvos = nu.ler_scores(SCORES)
    res["scores_refeitos_dos_spans_iguais_aos_gravados"] = sc == salvos
    antes = _spans_sha()
    cli.cmd_refazer(_ns())
    res["spans_finais_iguais_apos_refazer"] = antes == _spans_sha()
    ref = json.loads((DESTINO / "refeito.json").read_text(encoding="utf-8"))
    res["refeito"] = {k: {"relatorio_refeito_igual": v["relatorio_refeito_igual"],
                          "campos_diferentes": v["campos_diferentes"], "harness_cru_igual": v["harness_cru_igual"]}
                      for k, v in ref.items()}
    if a.scores_hub:
        hub = pathlib.Path(a.scores_hub)
        _, sh = nu.ler_scores(hub)
        res["hub"] = {"arquivo": str(hub), "sha256": nu.sha256(hub), "sha256_local": nu.sha256(SCORES),
                      "bytes_iguais": hub.read_bytes() == SCORES.read_bytes(), "scores_iguais": sh == salvos}
        saida = DESTINO / "refeito_hub" / f"{NOME}.json"
        cli.cmd_rodar(_ns(nome=NOME, scores=str(hub), campo="score", v0=False, politicas="A,B,C",
                          sem_oficial=False, saida=str(saida)))
        orig = json.loads((DESTINO / f"{NOME}.json").read_text(encoding="utf-8"))
        novo = json.loads(saida.read_text(encoding="utf-8"))
        res["hub"]["rodar_do_hub_igual"] = {k: orig.get(k) == novo.get(k) for k in
                                            ("politicas", "auroc", "conferencia_harness", "producao_estresse_oficial")}
        res["hub"]["spans_finais_iguais"] = antes == _spans_sha()
    res["ok"] = (res["scores_refeitos_dos_spans_iguais_aos_gravados"] and res["spans_finais_iguais_apos_refazer"]
                 and all(v["relatorio_refeito_igual"] and v["harness_cru_igual"] for v in res["refeito"].values())
                 and (not a.scores_hub or (res["hub"]["bytes_iguais"] and all(res["hub"]["rodar_do_hub_igual"].values())
                                           and res["hub"]["spans_finais_iguais"])))
    print("ok:", res["ok"], flush=True)
    _escrever("conferencia_final.json", res)


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.controles.verificador_treinado.gliner_ft_verif",
                                 description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="acao", required=True)
    for nome in ("scores", "rodar", "auroc", "dev", "refazer"):
        p = sub.add_parser(nome)
        if nome == "refazer":
            p.add_argument("--scores-hub", help="o arquivo de scores baixado do Hub, para rodar a partir dele")
    a = ap.parse_args()
    {"scores": cmd_scores, "rodar": cmd_rodar, "auroc": cmd_auroc, "dev": cmd_dev, "refazer": cmd_refazer}[a.acao](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
