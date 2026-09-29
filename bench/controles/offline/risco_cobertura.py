# -*- coding: utf-8 -*-
"""H10: risco x cobertura do Gama v1.3, com as saídas já gravadas (nenhuma inferência nova).

Ordena as previsões pela confiança, da maior para a menor, e mede a taxa de erro entre as mantidas
em função da fração mantida (cobertura). As duas confianças do sistema são variáveis diferentes e
aqui nunca se misturam:

  confiança do extrator   média da probabilidade dos tokens rotulados (`Span.confianca`), a que a
                          guarda compara com 0,95. Usada no texto real (305 e 172 ementas).
  confiança calibrada     o campo `confianca` do JSON final, P(acerto) do balde via x classe x faixa
                          (`src/gama/calibracao.json`), a que entra no Brier da métrica oficial.
                          Usada no estresse, único conjunto com gabarito de classe e de link.

As definições de erro, cobertura, empate, faixas e bootstrap estão em DEFINICOES e vão para o JSON.
Foram escritas antes da primeira medida.

    docker compose run --rm gama python -m bench.controles.offline.risco_cobertura
    -> bench/controles/offline/risco_cobertura.json (e os SVG com grafico_risco.py)

Reaproveita o harness (`bench.controles.avaliar`: leitura, aparar, régua gravada, guarda de produção,
caminho oficial) e as funções de casamento da métrica oficial (`vendor.kaggle_metric._casar`,
`_contida`, `_parse_*`, importadas, não copiadas). O único trecho reescrito é o laço de casamento
de `bench.pontuar.extracao`, porque ela só devolve contagens e aqui cada previsão precisa do seu
rótulo de acerto; as contagens dele são conferidas contra as de `bench.pontuar.extracao`.
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import random
import re

from avaliacao.harness import avaliar as avaliar_oficial
from avaliacao.harness import montar_solution, montar_submission
from bench import conjuntos, pontuar
from bench.controles import avaliar as A
from gama.indice import construir
from vendor import kaggle_metric as K

APP = pathlib.Path("/app")
AQUI = APP / "bench" / "controles" / "offline"
BENCH = A.BENCH
ROTULO_OFICIAL = "offline_gama_v13__guarda@0.95"
LIMIAR = A.LIMIAR_PRODUCAO                       # 0,95
FONTES = {"estresse": BENCH / "gama.jsonl", "reais": BENCH / "gama.jsonl", "novas": BENCH / "modelo_base12.jsonl"}
FAIXAS_EXTRATOR = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 1.0000001]
REAMOSTRAS, SEMENTE = 2000, 0

DEFINICOES = {
    "escritas_em": "2026-09-29, antes da primeira medida",
    "ordem": "previsões ordenadas pela confiança, decrescente. Cobertura = mantidas / total de previsões da "
             "população; risco = erros entre as mantidas / mantidas. Recall = acertos mantidos / itens do gabarito.",
    "empates": "previsões com a mesma confiança entram juntas: a curva tem um ponto por valor distinto de "
               "confiança. AURC usa a expectativa sob ordem aleatória dentro do empate (erros espalhados por igual).",
    "estresse": {
        "confianca": "calibrada: campo 'confianca' do JSON final do caminho de produção (aparar -> guarda 0,95 -> "
                     "resolver -> classe -> calibracao.json), com 4 casas, como montar_submission a entrega à métrica",
        "populacao": "todas as citações dos JSON finais dos 600 documentos, menos as que a métrica ignora pela regra "
                     "EXTRA (sem par e contidas >= 90% numa citação do gabarito já casada), que ficam fora e são contadas",
        "erro": "o alvo do Brier da métrica oficial estendido às sem par: par casado (IoU >= 0,5, guloso por maior "
                "IoU, vendor/kaggle_metric._casar) com classe diferente, ou com classe real e id_canonico fora dos "
                "doc_ids do gabarito, conta erro (y = 0); predição sem par que não é EXTRA conta erro (FP espúrio); "
                "par casado com mesma classe e link certo (quando real) é acerto (y = 1)",
        "niveis": "N1 e N2 juntos (a curva não pondera nível); contagens por nível no JSON",
    },
    "texto_real": {
        "confianca": "do extrator: Span.confianca gravado por ExtratorNeural (média da prob. dos tokens rotulados)",
        "variantes": {
            "com_guarda": "saída da guarda de produção (limiar 0,95) sobre os spans gravados e a régua gravada. Span "
                          "do modelo mantido leva a própria confiança; span da régua que a guarda pôs no lugar leva a "
                          "maior confiança do extrator entre os spans fracos (< 0,95) do modelo que ele cruza, isto é, "
                          "a confiança do extrator naquele trecho (sempre < 0,95)",
            "cru": "os spans do modelo, só aparados, antes da guarda; cada um com a sua confiança",
        },
        "erro": "previsão sem par no casamento 1 para 1 de bench.pontuar.extracao (mesmo tipo com VAGA agrupada em "
                "JURIS, IoU >= 0,5, ouro percorrido em ordem e primeira previsão livre que casa), isto é, um FP",
        "populacao": "todas as previsões da variante nos 305 (reais, validação) ou nos 172 (novas, confirmação)",
    },
    "faixas_confianca_extrator": "[0; 0,5) [0,5; 0,8) [0,8; 0,9) [0,9; 0,95) [0,95; 0,98) [0,98; 0,99) "
                                 "[0,99; 0,999) [0,999; 1]",
    "resumos": "AURC (área sob risco x cobertura, média do risco nas N posições), AURC oráculo (erros por último), "
               "AURC aleatório (= risco total), E-AURC = AURC - oráculo, AUROC da confiança separando acerto de "
               "erro (empate = 1/2). IC95 por bootstrap pareado por documento: 2.000 reamostras, semente 0, "
               "reamostras ordenadas 50 e 1.949, como bench/alunos.py",
    "o_que_nao_e": "não muda a métrica oficial nem o schema da submissão; é diagnóstico do uso da confiança",
}


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- curva e resumos

def curva(itens: list[dict], total_ouro: int | None) -> list[dict]:
    """Um ponto por valor distinto de confiança (decrescente)."""
    grupos = collections.OrderedDict()
    for it in sorted(itens, key=lambda x: -x["conf"]):
        g = grupos.setdefault(it["conf"], [0, 0])
        g[0] += 1
        g[1] += it["erro"]
    pts, k, e, n = [], 0, 0, len(itens)
    for c, (m, err) in grupos.items():
        k, e = k + m, e + err
        p = {"limiar": c, "mantidas": k, "erros": e, "cobertura": round(k / n, 6), "risco": round(e / k, 6)}
        if total_ouro:
            p["recall"] = round((k - e) / total_ouro, 6)
        pts.append(p)
    return pts


def aurc(itens: list[dict]) -> float | None:
    n = len(itens)
    if not n:
        return None
    grupos = collections.OrderedDict()
    for it in sorted(itens, key=lambda x: -x["conf"]):
        g = grupos.setdefault(it["conf"], [0, 0])
        g[0] += 1
        g[1] += it["erro"]
    soma, k, e = 0.0, 0, 0
    for m, err in grupos.values():
        for j in range(1, m + 1):
            soma += (e + err * j / m) / (k + j)
        k, e = k + m, e + err
    return soma / n


def aurc_oraculo(itens: list[dict]) -> float | None:
    n, erros = len(itens), sum(it["erro"] for it in itens)
    if not n:
        return None
    return sum(max(0, k - (n - erros)) / k for k in range(1, n + 1)) / n


def auroc(itens: list[dict]) -> float | None:
    """P(confiança de um acerto > de um erro), empate vale 1/2 (Mann-Whitney por postos)."""
    pos = [it["conf"] for it in itens if not it["erro"]]
    neg = [it["conf"] for it in itens if it["erro"]]
    if not pos or not neg:
        return None
    todos = sorted([(c, 1) for c in pos] + [(c, 0) for c in neg])
    postos, i = {}, 0
    soma_pos = 0.0
    while i < len(todos):
        j = i
        while j < len(todos) and todos[j][0] == todos[i][0]:
            j += 1
        medio = (i + 1 + j) / 2
        soma_pos += medio * sum(1 for x in todos[i:j] if x[1])
        i = j
    return (soma_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def bootstrap(itens: list[dict]) -> dict:
    por_doc = collections.defaultdict(list)
    for it in itens:
        por_doc[it["doc"]].append(it)
    docs = sorted(por_doc)
    rng = random.Random(SEMENTE)
    au, ar = [], []
    for _ in range(REAMOSTRAS):
        amostra = [x for d in (rng.choice(docs) for _ in docs) for x in por_doc[d]]
        a = auroc(amostra)
        if a is not None:
            au.append(a)
        ar.append(aurc(amostra))

    def ic(v):
        v = sorted(v)
        if len(v) < REAMOSTRAS:
            return None
        return [round(v[int(0.025 * REAMOSTRAS)], 4), round(v[int(0.975 * REAMOSTRAS) - 1], 4)]
    return {"auroc_ic95": ic(au), "aurc_ic95": ic(ar), "reamostras": REAMOSTRAS, "semente": SEMENTE,
            "unidade": "documento"}


def faixas(itens: list[dict], bordas=FAIXAS_EXTRATOR) -> list[dict]:
    out = []
    for a, b in zip(bordas, bordas[1:]):
        xs = [it for it in itens if a <= it["conf"] < b]
        rot = f"[{a:g}; {min(b, 1.0):g}{']' if b > 1 else ')'}".replace(".", ",")
        out.append({"faixa": rot, "n": len(xs), "erros": sum(x["erro"] for x in xs),
                    "taxa_erro": round(sum(x["erro"] for x in xs) / len(xs), 4) if xs else None})
    return out


def resumir(itens: list[dict], total_ouro: int | None, com_faixas=True, com_boot=True) -> dict:
    n, e = len(itens), sum(it["erro"] for it in itens)
    r = {"previsoes": n, "erros": e, "risco_total": round(e / n, 4) if n else None,
         "aurc": _r(aurc(itens)), "aurc_oraculo": _r(aurc_oraculo(itens)),
         "aurc_aleatorio": round(e / n, 4) if n else None, "auroc": _r(auroc(itens))}
    r["e_aurc"] = round(r["aurc"] - r["aurc_oraculo"], 4) if r["aurc"] is not None else None
    if total_ouro is not None:
        r["itens_ouro"] = total_ouro
        r["recall_total"] = round((n - e) / total_ouro, 4)
    if com_boot and e:
        r.update(bootstrap(itens))
    if com_faixas:
        r["faixas"] = faixas(itens)
    pts = curva(itens, total_ouro)
    colunas = [k for k in ("limiar", "mantidas", "erros", "cobertura", "risco", "recall") if k in pts[0]]
    r["curva"] = {"colunas": colunas, "pontos": [[p[k] for k in colunas] for p in pts]}
    return r


def _r(x):
    return None if x is None else round(x, 4)


# ---------------------------------------------------------------- texto real: confiança do extrator

def casar_extracao(gold: list, pred: list) -> list[bool]:
    """O laço de `bench.pontuar.extracao` (juntar_vaga=True), devolvendo acerto por previsão."""
    def norm(t):
        return "JURIS" if t == "VAGA" else t
    ps = [(a, b, norm(t)) for a, b, t in pred]
    usados = set()
    for a, b, t in gold:
        t = norm(t)
        k = next((k for k, (pa, pb, pt) in enumerate(ps) if k not in usados and pt == t
                  and max(0, min(b, pb) - max(a, pa)) / (max(b, pb) - min(a, pa)) >= 0.5), None)
        if k is not None:
            usados.add(k)
    return [k in usados for k in range(len(ps))]


def texto_real(conjunto: str) -> dict:
    _, linhas = A.ler([FONTES[conjunto]])
    sp = A.spans(linhas, conjunto)
    reg = A.regua(conjunto)
    _, gold = A.textos_e_ouro(conjunto)
    guardados = A.com_guarda(sp, reg, LIMIAR)
    filtrados = A.com_guarda(sp, {d: [] for d in sp}, 0.0)   # só a regra da VAGA colada (limiar 0: todos fortes)
    total_ouro = sum(len(g) for g in gold.values())
    variantes = {}
    for nome, finais in (("com_guarda", guardados), ("cru", sp)):
        itens, triplas = [], {}
        for d, ss in finais.items():
            fracos = [w for w in filtrados[d] if w.confianca is not None and w.confianca < LIMIAR]
            acertos = casar_extracao(gold.get(d, []), [(s.inicio, s.fim, pontuar.rotulo(s)) for s in ss])
            triplas[d] = [(s.inicio, s.fim, pontuar.rotulo(s)) for s in ss]
            for s, ok in zip(ss, acertos):
                do_modelo = any(s is m for m in sp[d])
                if do_modelo:
                    conf = s.confianca
                else:
                    cruzados = [w.confianca for w in fracos if s.inicio < w.fim and w.inicio < s.fim]
                    conf = max(cruzados)
                itens.append({"doc": d, "conf": round(conf, 6), "erro": int(not ok),
                              "origem": "modelo" if do_modelo else "regua"})
        ref = pontuar.extracao(gold, triplas, juntar_vaga=True)
        tp = sum(v["tp"] for v in ref["por_tipo"].values())
        fp = sum(v["fp"] for v in ref["por_tipo"].values())
        assert tp == sum(1 - i["erro"] for i in itens) and fp == sum(i["erro"] for i in itens), (conjunto, nome)
        r = resumir(itens, total_ouro)
        r["f1_extracao"] = ref["f1"]
        r["conferencia"] = {"tp_fp_iguais_a_bench_pontuar_extracao": True, "tp": tp, "fp": fp}
        por_origem = {}
        for o in ("modelo", "regua"):
            xs = [i for i in itens if i["origem"] == o]
            if xs:
                por_origem[o] = {"n": len(xs), "erros": sum(i["erro"] for i in xs),
                                 "taxa_erro": round(sum(i["erro"] for i in xs) / len(xs), 4),
                                 "conf_min": min(i["conf"] for i in xs), "conf_max": max(i["conf"] for i in xs)}
        r["por_origem"] = por_origem
        if nome == "cru":
            acima = [i for i in itens if i["conf"] >= LIMIAR]
            r["no_limiar_da_guarda"] = {"limiar": LIMIAR, "mantidas": len(acima),
                                       "cobertura": round(len(acima) / len(itens), 4),
                                       "risco": round(sum(i["erro"] for i in acima) / len(acima), 4),
                                       "recall": round(sum(1 - i["erro"] for i in acima) / total_ouro, 4)}
        variantes[nome] = r
    return {"documentos": len(sp), "fonte": str(FONTES[conjunto].relative_to(APP)),
            "fonte_sha256": _sha(FONTES[conjunto]), "variantes": variantes}


# ---------------------------------------------------------------- estresse: confiança calibrada

def estresse(idx) -> dict:
    _, linhas = A.ler([FONTES["estresse"]])
    sp = A.spans(linhas, "estresse")
    finais = A.com_guarda(sp, A.regua("estresse"), LIMIAR)
    oficial = A.oficial(ROTULO_OFICIAL, finais, idx, True)          # regrava os JSON do zero
    pasta = A.SAIDA / "json" / "estresse" / ROTULO_OFICIAL
    sol = montar_solution(conjuntos.gabarito_oficial("estresse"))
    sub = montar_submission(pasta, sol["documento_id"].tolist())
    res_metrica = avaliar_oficial(sol, sub)
    sub_idx = sub.set_index("documento_id")
    itens, extras, niveis = [], 0, collections.Counter()
    brier_pares = []
    acc = {}
    for _, linha in sol.iterrows():
        doc, nivel = linha["documento_id"], int(linha["nivel"])
        golds = K._parse_solution_cell(linha["citacoes"], doc)
        preds = K._parse_submission_cell(sub_idx.loc[doc, "citacoes"], doc)
        K._acumular_documento(acc.setdefault(nivel, K._novo_acumulador()), golds, preds)
        pares, _, sem_par = K._casar(golds, preds)
        for gi, pi in pares:
            g, p = golds[gi], preds[pi]
            y = int(g["classe"] == p["classe"] and (g["classe"] != "real" or p["id_canonico"] in g["doc_ids"]))
            itens.append({"doc": doc, "conf": p["confianca"], "erro": 1 - y, "nivel": nivel, "casada": True})
            if p["confianca"] is not None:
                brier_pares.append((p["confianca"] - y) ** 2)
        casados = [golds[gi] for gi, _ in pares]
        for pi in sem_par:
            if any(K._contida(preds[pi], g) for g in casados):
                extras += 1
                continue
            itens.append({"doc": doc, "conf": preds[pi]["confianca"], "erro": 1, "nivel": nivel, "casada": False})
        niveis[nivel] += len(golds)
    sem_conf = sum(i["conf"] is None for i in itens)
    itens = [i for i in itens if i["conf"] is not None]
    # conferência: os rótulos por previsão reproduzem a métrica (tp/fp/fn por nível e Brier)
    confere = {}
    for nivel, a in acc.items():
        m = res_metrica["niveis"][nivel]
        brier_nivel = sum(a["brier_termos"]) / len(a["brier_termos"])
        confere[f"n{nivel}"] = {"tp": a["tp"], "fp": a["fp"], "fn": a["fn"], "brier": round(brier_nivel, 6),
                                "score": round(m["score"], 6)}
    total_ouro = sum(niveis.values())
    r = resumir(itens, total_ouro, com_faixas=False)
    confiab = []
    for c in sorted({i["conf"] for i in itens}, reverse=True):
        xs = [i for i in itens if i["conf"] == c]
        confiab.append({"confianca_calibrada": c, "n": len(xs), "acertos": sum(1 - i["erro"] for i in xs),
                        "taxa_acerto": round(sum(1 - i["erro"] for i in xs) / len(xs), 4)})
    r.update({"documentos": len(sol), "itens_ouro_por_nivel": {f"n{k}": v for k, v in sorted(niveis.items())},
              "extras_ignoradas": extras, "sem_confianca": sem_conf,
              "brier_pares_casados": round(sum(brier_pares) / len(brier_pares), 6),
              "confiabilidade": confiab,
              "metrica_oficial": {"final_recalculado": round(res_metrica["score_final"], 5),
                                  "final_harness": oficial["final"], "por_nivel": confere},
              "json_finais": str(pasta.relative_to(APP)), "fonte": str(FONTES["estresse"].relative_to(APP)),
              "fonte_sha256": _sha(FONTES["estresse"])})
    return r


def main() -> int:
    idx = construir(pontuar.DB)
    out = {"experimento": "offline/H10", "definicoes": DEFINICOES,
           "sistema": {"modelo": "Gama v1.3 = vinimlo/gama@5f924ca2fa77c2aae6afe6d770c78ca4438f52f3",
                       "spans": {c: str(p.relative_to(APP)) for c, p in FONTES.items()},
                       "nota_172": "modelo_base12.jsonl são os mesmos pesos antes de publicados (spans e confianças "
                                   "idênticos a gama.jsonl nos 905 documentos em comum; conferido em "
                                   "saidas/bench/controles/validacao.json)",
                       "guarda": "src/gama/extratores/guarda.py, limiar 0,95; régua gravada (saidas/bench/regua.jsonl "
                                 "e saidas/bench/controles/regua_novas.jsonl)"},
           "codigo": A.impressao()}
    out["estresse_confianca_calibrada"] = estresse(idx)
    print("estresse", {k: out["estresse_confianca_calibrada"][k] for k in ("previsoes", "erros", "extras_ignoradas")},
          flush=True)
    for c, chave in (("reais", "reais_305_confianca_extrator"), ("novas", "novas_172_confianca_extrator")):
        out[chave] = texto_real(c)
        for v, r in out[chave]["variantes"].items():
            print(c, v, {k: r[k] for k in ("previsoes", "erros", "risco_total", "aurc", "aurc_oraculo", "auroc")},
                  flush=True)
    destino = AQUI / "risco_cobertura.json"
    texto = json.dumps(out, ensure_ascii=False, indent=1)
    # uma linha por ponto da curva (o arquivo fica legível e pequeno)
    texto = re.sub(r"\[\n\s+([-\d.e]+),\n\s+(\d+),\n\s+(\d+),\n\s+([-\d.e]+),\n\s+([-\d.e]+)(?:,\n\s+([-\d.e]+))?\n\s+\]",
                   lambda m: "[" + ", ".join(g for g in m.groups() if g is not None) + "]", texto)
    destino.write_text(texto, encoding="utf-8")
    print("->", destino)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
