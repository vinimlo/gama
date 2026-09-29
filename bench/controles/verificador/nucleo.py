# -*- coding: utf-8 -*-
"""Candidatos, políticas e pontuação do verificador de candidatos.

CANDIDATOS (por documento, nos três conjuntos do harness)
    Os spans do Gama v1.3 (os arquivos que o harness usa como referência: `saidas/bench/gama.jsonl`
    no estresse e nas 305, `saidas/bench/modelo_base12.jsonl` nas 172) e os da régua (a gravação do
    harness), lidos e aparados pelo próprio harness (`avaliar.spans`, `avaliar.regua`). Cada candidato
    leva `cid` = "<conjunto>/<doc>/<origem>/<inicio>-<fim>", a origem (gama|regua), a confiança do
    Gama, se é forte (Gama, confiança >= 0,95, depois da regra 1), se cruza um span forte e o
    `estado` para o modelo de decisão (janela de 300 caracteres de cada lado, candidato entre [[ ]]).

GRUPOS DA GUARDA (sempre pela função de produção `guarda.guardar`, nunca reimplementados)
    G1 = sobreviventes da regra 1 (VAGA colada sai)   guardar(G, [])   com limiar 0
    F  = fortes, confiança >= 0,95 depois da regra 1   guardar(G, [])   com limiar 0,95
    W  = fracos = G1 - F
    X  = spans da régua que a guarda põe onde o Gama hesita (cruzam um fraco e nenhum forte)
         = guardar(G, R) - F  com limiar 0,95

POLÍTICAS (score >= tau; sobreposição entre elegíveis resolvida pelo maior score; empate: Gama
antes da régua, depois o início)
    V0-A  a guarda de produção com o limiar varrido: `guardar(G, R)` com CONFIANCA_MINIMA = tau.
          O score é a confiança do Gama; a régua não tem sinal (entra onde o Gama cai abaixo de tau).
    A     F fica; elegíveis W + X.
    B     como A, e também os spans da régua que não cruzam nenhum span do Gama (antes da regra 1).
    C     todo candidato do Gama e da régua é elegível (sem regra 1, sem fixos), como o teto (b) de
          `saidas/analises/teto_do_decisor.py`.

ESCOLHA DE TAU: grade 0,05 a 0,95 (passo 0,05), maior F1 exato nas 305; empate -> o maior tau.
O tau escolhido é medido nas 172 sem reescolha. No estresse: documentos cuja lista final de spans
(início, fim, tipo, forma) difere da saída de produção (guarda@0,95) e a nota oficial pelo harness.
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import pathlib

from gama.extratores import guarda
from gama.span import Span

from .. import avaliar
from ... import pontuar

SAIDA = avaliar.SAIDA / "verificador"          # /app/saidas/bench/controles/verificador
CANDIDATOS = SAIDA / "candidatos.jsonl"
CONJUNTOS = avaliar.CONJUNTOS
FORTE = avaliar.LIMIAR_PRODUCAO                 # 0,95
JANELA = 300
MARCAS = ("[[", "]]")
GRADE = [round(0.05 * i, 2) for i in range(1, 20)]
POLITICAS = ("A", "B", "C")


def sha256(caminho) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _rel(caminho) -> str:
    return str(pathlib.Path(caminho)).replace("/app/", "", 1)


def _cruza(a: Span, b: Span) -> bool:
    return a.inicio < b.fim and b.inicio < a.fim


def _chave(ss) -> list:
    return [(s.inicio, s.fim, s.tipo, s.forma) for s in ss]


# ---------------------------------------------------------------- grupos da guarda

def grupos(G: list, R: list) -> tuple[list, list, list, list]:
    """(G1, F, W, X) pela guarda de produção; ver o docstring do módulo."""
    with avaliar.limiar(0.0):
        g1 = guarda.guardar(G, [])
    with avaliar.limiar(FORTE):
        f = guarda.guardar(G, [])
        prod = guarda.guardar(G, R)
    ids_f = {id(s) for s in f}
    return g1, f, [s for s in g1 if id(s) not in ids_f], [s for s in prod if id(s) not in ids_f]


# ---------------------------------------------------------------- candidatos

def estado(texto: str, a: int, b: int) -> str:
    return texto[max(0, a - JANELA):a] + MARCAS[0] + texto[a:b] + MARCAS[1] + texto[b:b + JANELA]


def _cid(c: str, d: str, origem: str, s: Span) -> str:
    return f"{c}/{d}/{origem}/{s.inicio}-{s.fim}"


def fontes() -> dict:
    return {c: {"gama": _rel(avaliar.REFERENCIA[c]), "regua": _rel(avaliar.REGUA[c])} for c in CONJUNTOS}


def construir_candidatos(destino: pathlib.Path = CANDIDATOS) -> dict:
    """Grava o JSONL de candidatos (1ª linha meta) e devolve contagens por conjunto."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    arquivos = sorted({avaliar.REFERENCIA[c] for c in CONJUNTOS} | {avaliar.REGUA[c] for c in CONJUNTOS})
    meta = {"janela": JANELA, "marcas": MARCAS, "limiar_forte": FORTE, "fontes": fontes(),
            "sha256_fontes": {_rel(a): sha256(a) for a in arquivos}, "codigo": avaliar.impressao(),
            "campos": "cid, conjunto, doc, origem, inicio, fim, tipo, forma, digitos, confianca, forte, "
                      "vaga_colada, cruza_forte, elegivel{A,B,C}, estado"}
    contagem = {}
    vistos = set()
    with destino.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for c in CONJUNTOS:
            textos, _ = avaliar.textos_e_ouro(c)
            gama = avaliar.spans(pontuar._ler_spans(avaliar.REFERENCIA[c])[1], c)
            regua = avaliar.regua(c)
            k = collections.Counter()
            for d, t in textos.items():
                G, R = gama[d], regua[d]
                g1, f, w, x = grupos(G, R)
                ids_g1, ids_f, ids_a = {id(s) for s in g1}, {id(s) for s in f}, {id(s) for s in w + x}
                for origem, ss in (("gama", G), ("regua", R)):
                    for s in ss:
                        cid = _cid(c, d, origem, s)
                        if cid in vistos:
                            raise SystemExit(f"cid repetido: {cid}")
                        vistos.add(cid)
                        a_ = id(s) in ids_a
                        b_ = a_ or (origem == "regua" and not any(_cruza(s, g) for g in G))
                        rec = {"cid": cid, "conjunto": c, "doc": d, "origem": origem, "inicio": s.inicio,
                               "fim": s.fim, "tipo": s.tipo, "forma": s.forma, "digitos": s.digitos,
                               "confianca": s.confianca, "forte": id(s) in ids_f,
                               "vaga_colada": origem == "gama" and id(s) not in ids_g1,
                               "cruza_forte": any(_cruza(s, o) for o in f if o is not s),
                               "elegivel": {"A": a_, "B": b_, "C": True},
                               "estado": estado(t, s.inicio, s.fim)}
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        k[origem] += 1
                        k[f"{origem}_elegivel_A"] += a_
                        k[f"{origem}_elegivel_B"] += b_
                        k[f"{origem}_forte"] += rec["forte"]
                        k[f"{origem}_vaga_colada"] += rec["vaga_colada"]
            contagem[c] = {"documentos": len(textos), **dict(sorted(k.items()))}
            print(c, contagem[c], flush=True)
    return {"arquivo": _rel(destino), "sha256": sha256(destino), "contagem": contagem}


@dataclasses.dataclass(frozen=True)
class Cand:
    cid: str
    origem: str
    span: Span


def ler_candidatos(arq=CANDIDATOS) -> tuple[dict, dict]:
    """(meta, {conjunto: {doc: (G, R)}}) com os Span reconstruídos exatamente como o harness os lê
    (trecho = texto[inicio:fim], como `pontuar.para_span` e `aparar` fazem)."""
    meta, docs = {}, {}
    textos = {c: avaliar.textos_e_ouro(c)[0] for c in CONJUNTOS}
    for c in CONJUNTOS:
        docs[c] = {d: ([], []) for d in textos[c]}
    for linha in pathlib.Path(arq).read_text(encoding="utf-8").splitlines():
        r = json.loads(linha)
        if "meta" in r:
            meta = r["meta"]
            continue
        t = textos[r["conjunto"]][r["doc"]]
        s = Span(r["inicio"], r["fim"], t[r["inicio"]:r["fim"]], r["tipo"], r["forma"], r["digitos"], r["confianca"])
        docs[r["conjunto"]][r["doc"]][0 if r["origem"] == "gama" else 1].append(Cand(r["cid"], r["origem"], s))
    return meta, docs


def conferir(docs: dict) -> dict:
    """Candidatos relidos = spans do harness; V0 com tau = 0,95 = a guarda de produção do harness."""
    out = {}
    for c in CONJUNTOS:
        gama = avaliar.spans(pontuar._ler_spans(avaliar.REFERENCIA[c])[1], c)
        regua = avaliar.regua(c)
        prod = avaliar.com_guarda(gama, regua, FORTE)
        v0 = {d: politica_v0(G, R, FORTE) for d, (G, R) in docs[c].items()}
        out[c] = {"docs": len(docs[c]),
                  "gama_diferentes": sum([x.span for x in docs[c][d][0]] != gama[d] for d in gama),
                  "regua_diferentes": sum([x.span for x in docs[c][d][1]] != regua[d] for d in regua),
                  "v0_095_diferente_da_producao": sum(v0[d] != prod[d] for d in prod)}
    out["ok"] = all(v["gama_diferentes"] == v["regua_diferentes"] == v["v0_095_diferente_da_producao"] == 0
                    for v in out.values())
    return out


# ---------------------------------------------------------------- scores

def ler_scores(arq, campo: str = "score") -> tuple[dict, dict]:
    """(meta, {cid: score}). Linha 1 opcional {"meta": ...}; depois {"cid": ..., <campo>: float}.
    `campo` aceita caminho com ponto (ex.: "p.f2" no arquivo bruto do Laya)."""
    meta, sc = {}, {}
    for linha in pathlib.Path(arq).read_text(encoding="utf-8").splitlines():
        r = json.loads(linha)
        if "meta" in r:
            meta = r["meta"]
            continue
        v = r
        for k in campo.split("."):
            v = v[k]
        sc[r["cid"]] = float(v)
    return meta, sc


def gravar_scores(destino: pathlib.Path, meta: dict, scores: dict) -> dict:
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for cid, s in scores.items():
            fh.write(json.dumps({"cid": cid, "score": s}) + "\n")
    return {"arquivo": _rel(destino), "sha256": sha256(destino), "candidatos": len(scores)}


def _norm(rot: str, juntar_vaga: bool) -> str:
    return "JURIS" if juntar_vaga and rot == "VAGA" else rot


def _iou(a0, a1, b0, b1) -> float:
    return max(0, min(a1, b1) - max(a0, b0)) / (max(a1, b1) - min(a0, b0))


def casa(s: Span, gs: list, juntar_vaga: bool) -> bool:
    """Existe span do ouro do mesmo tipo com IoU >= 0,5 (o critério de `bench.pontuar.extracao`)."""
    t = _norm(pontuar.rotulo(s), juntar_vaga)
    return any(_norm(gt, juntar_vaga) == t and _iou(s.inicio, s.fim, a, b) >= 0.5 for a, b, gt in gs)


def scores_oraculo(docs: dict) -> dict:
    """Teto: score 1 se o candidato casa com o ouro (VAGA junto de JURIS fora do estresse), senão 0."""
    sc = {}
    for c in CONJUNTOS:
        _, gold = avaliar.textos_e_ouro(c)
        for d, (G, R) in docs[c].items():
            for x in G + R:
                sc[x.cid] = 1.0 if casa(x.span, gold[d], c != "estresse") else 0.0
    return sc


def scores_de_spans(docs: dict, arquivos: list) -> tuple[dict, dict]:
    """Score de um verificador que devolve spans (ex.: GLiNER fine-tunado), no formato do harness:
    score do candidato = maior score de um span previsto do mesmo tipo (VAGA junto de JURIS) com
    IoU >= 0,5, senão 0. Spans previstos passam por `avaliar.spans` (aparados; o score do formato
    curto vira a confiança). Span sem score conta 1."""
    metas, linhas = avaliar.ler(arquivos)
    sc, faltam = {}, {}
    for c in CONJUNTOS:
        pred = avaliar.spans(linhas, c)
        if pred is None:
            faltam[c] = "conjunto incompleto no arquivo: score 0 para todos os candidatos"
            pred = {d: [] for d in docs[c]}
        for d, (G, R) in docs[c].items():
            for x in G + R:
                t = _norm(pontuar.rotulo(x.span), True)
                v = [1.0 if p.confianca is None else float(p.confianca) for p in pred[d]
                     if _norm(pontuar.rotulo(p), True) == t
                     and _iou(x.span.inicio, x.span.fim, p.inicio, p.fim) >= 0.5]
                sc[x.cid] = max(v, default=0.0)
    return {"origem": [str(a) for a in arquivos], "metas": metas, "conjuntos_faltando": faltam}, sc


def auroc(pares: list) -> float | None:
    """AUROC de (score, rótulo 0/1) por postos médios (Mann-Whitney)."""
    pos = sum(y for _, y in pares)
    neg = len(pares) - pos
    if not pos or not neg:
        return None
    ordem = sorted(pares, key=lambda p: p[0])
    postos, i = [0.0] * len(ordem), 0
    while i < len(ordem):
        j = i
        while j + 1 < len(ordem) and ordem[j + 1][0] == ordem[i][0]:
            j += 1
        for k in range(i, j + 1):
            postos[k] = (i + j) / 2 + 1
        i = j + 1
    soma = sum(r for r, (_, y) in zip(postos, ordem) if y)
    return round((soma - pos * (pos + 1) / 2) / (pos * neg), 4)


# ---------------------------------------------------------------- políticas

def politica_v0(G: list, R: list, tau: float) -> list:
    """A guarda de produção com o limiar varrido (score = confiança do Gama)."""
    with avaliar.limiar(tau):
        return guarda.guardar([x.span for x in G], [x.span for x in R])


def politica(pol: str, G: list, R: list, score: dict, tau: float) -> list:
    gs, rs = [x.span for x in G], [x.span for x in R]
    if pol == "C":
        fixos, eleg = [], G + R
    else:
        _, f, w, x = grupos(gs, rs)
        ids = {id(s) for s in w + x}
        fixos, eleg = f, [c for c in G + R if id(c.span) in ids]
        if pol == "B":
            eleg += [c for c in R if not any(_cruza(c.span, g) for g in gs)]
    ok = sorted((c for c in eleg if score[c.cid] >= tau),
                key=lambda c: (-score[c.cid], c.origem != "gama", c.span.inicio, c.span.fim))
    escolhidos = []
    for c in ok:
        if not any(_cruza(c.span, o) for o in fixos + escolhidos):
            escolhidos.append(c.span)
    return sorted(fixos + escolhidos, key=lambda s: s.inicio)


def aplicar(pol: str, docs_c: dict, score: dict | None, tau: float) -> dict:
    if pol == "V0-A":
        return {d: politica_v0(G, R, tau) for d, (G, R) in docs_c.items()}
    return {d: politica(pol, G, R, score, tau) for d, (G, R) in docs_c.items()}


def producao(docs_c: dict) -> dict:
    return {d: politica_v0(G, R, FORTE) for d, (G, R) in docs_c.items()}


# ---------------------------------------------------------------- avaliação

def _faltando(docs: dict, score: dict) -> int:
    return sum(x.cid not in score for c in CONJUNTOS for G, R in docs[c].values() for x in G + R)


def escolher_tau(pol: str, docs: dict, score: dict | None) -> tuple[float, dict]:
    tabela = {t: avaliar.extracao("reais", aplicar(pol, docs["reais"], score, t))["f1_exato"] for t in GRADE}
    melhor = max(tabela.values())
    return max(t for t, f in tabela.items() if f == melhor), {f"{t:.2f}": round(f, 4) for t, f in tabela.items()}


def gravar_spans(destino: pathlib.Path, meta: dict, finais: dict) -> dict:
    """Spans finais no formato do harness (Span completo), para `avaliar pontuar` reler do zero."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for c in CONJUNTOS:
            for d, ss in finais[c].items():
                fh.write(json.dumps({"conjunto": c, "id": d, "spans": [[s.inicio, s.fim, s.tipo, s.forma, s.digitos,
                                                                         s.confianca] for s in ss]},
                                    ensure_ascii=False) + "\n")
    return {"arquivo": _rel(destino), "sha256": sha256(destino)}


class Oficial:
    """Nota oficial no estresse pelo harness (`avaliar.oficial`), com os JSON por documento em
    SAIDA/json/estresse/<rótulo>/; resultados em cache pela lista final de spans."""

    def __init__(self, idx):
        self.idx, self.cache = idx, {}

    def __call__(self, rotulo: str, finais: dict) -> dict:
        chave = hashlib.sha256(json.dumps({d: _chave(ss) + [(s.digitos, s.confianca) for s in ss]
                                           for d, ss in sorted(finais.items())}).encode()).hexdigest()
        if chave not in self.cache:
            antes = avaliar.SAIDA
            avaliar.SAIDA = SAIDA                 # os JSON oficiais vão para a pasta do verificador
            try:
                r = avaliar.oficial(rotulo, finais, self.idx, True)
            finally:
                avaliar.SAIDA = antes
            self.cache[chave] = {"rotulo": rotulo, **r}
        return self.cache[chave]


def avaliar_politica(nome: str, pol: str, docs: dict, score: dict | None, oficial: Oficial | None,
                     prod: dict, tau: float | None = None) -> dict:
    """Escolhe tau nas 305 (se não for dado), mede nas 172 e no estresse, grava os spans finais."""
    if tau is None:
        tau, tabela = escolher_tau(pol, docs, score)
        origem_tau = "escolhido nas 305 (maior F1 exato; empate -> maior tau)"
    else:
        tabela, origem_tau = {}, "fixo (dado)"
    finais = {c: aplicar(pol, docs[c], score, tau) for c in CONJUNTOS}
    res = {"politica": pol, "tau": tau, "origem_tau": origem_tau, "f1_305_por_tau": tabela, "conjuntos": {}}
    for c in CONJUNTOS:
        r = {"extracao": avaliar.extracao(c, finais[c]),
             "docs_mudados_vs_producao": sum(_chave(finais[c][d]) != _chave(prod[c][d]) for d in finais[c])}
        if c in ("reais", "novas"):
            r["vs_gama_v13_guarda"] = avaliar.bootstrap(avaliar.por_doc(c, finais[c]), avaliar.referencia(c))
        if c == "estresse":
            r["docs_mudados_por_tau"] = {f"{t:.2f}": sum(_chave(ss) != _chave(prod[c][d]) for d, ss in
                                                        aplicar(pol, docs[c], score, t).items()) for t in GRADE}
            if oficial is not None:
                r["oficial"] = oficial(f"{nome}__{pol}@{tau:.2f}", finais[c])
        res["conjuntos"][c] = r
    est = res["conjuntos"]["estresse"]
    res["candidata_a_uso"] = est["docs_mudados_vs_producao"] == 0
    res["spans"] = gravar_spans(SAIDA / "spans" / f"{nome}__{pol}.jsonl",
                                {"extrator": f"verificador:{nome}:{pol}", "tau": tau,
                                 "nota": "spans finais da política; `cru` no harness = esta política"}, finais)
    linha = resumo_linha(nome, res)
    print(" | ".join(f"{k} {v}" for k, v in linha.items()), flush=True)
    res["resumo"] = linha
    return res


def resumo_linha(nome: str, res: dict) -> dict:
    cj = res["conjuntos"]
    ic = lambda c: cj[c]["vs_gama_v13_guarda"]  # noqa: E731
    return {"verificador": nome, "politica": res["politica"], "tau": res["tau"],
            "f1_305": cj["reais"]["extracao"]["f1"], "f1_172": cj["novas"]["extracao"]["f1"],
            "dif_305": ic("reais")["dif"], "ic95_305": ic("reais")["ic95"],
            "dif_172": ic("novas")["dif"], "ic95_172": ic("novas")["ic95"],
            "estresse_docs_mudados": cj["estresse"]["docs_mudados_vs_producao"],
            "estresse_oficial": cj["estresse"].get("oficial", {}).get("final"),
            "estresse_f1": cj["estresse"]["extracao"]["f1"], "candidata_a_uso": res["candidata_a_uso"]}
