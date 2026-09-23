# -*- coding: utf-8 -*-
"""Mede as hipóteses levantadas pelo benchmark, com os spans já gravados (sem GPU).

    filtros    pós-filtros de fragmento sobre o Gama: quanto sobem no texto real e se tiram
               algum acerto no dev ou no estresse (critério de aceitação: zero spans a menos)
    vieses     o texto real mede o que parece medir? Estratos da amostragem, casamento
               frouxo (a convenção de borda do ouro pesa?) e bootstrap pareado por ementa
    selecao    o ganho do melhor filtro sobrevive a escolher numa metade e medir na outra?
    precisao   FP32 x BF16 x FP16 (spans do HF Jobs) e a cauda de confiança dos acertos no final_v1
    auditoria  amostra de erros no texto real, com contexto, para ler à mão

    docker compose run --rm gama python -m bench.hipoteses filtros
    docker compose run --rm gama python -m bench.hipoteses vieses
    docker compose run --rm gama python -m bench.hipoteses selecao     # depois de filtros
    docker compose run --rm gama python -m bench.hipoteses auditoria   # -> saidas/bench/auditoria.md

Resultados em `bench/hipoteses.json` (uma chave por ação).
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random

from gama.indice import construir
from gama.normalizar import nucleo_numerico
from gama.pipeline import aparar
from gama.resolver import _ART_NUM, _SUM_NUM, _chave_lei_da_citacao, _numero_ocr

from . import conjuntos
from .pontuar import DB, SAIDA, extracao, oficial, para_span, rotulo, spans_de

RESULTADOS = pathlib.Path("/app/bench/hipoteses.json")
PREDICOES_REAIS = pathlib.Path("/app/corpus/reais/predicoes.jsonl")
PRATA = pathlib.Path("/app/corpus/reais/prata.jsonl")


def carregar(extrator: str, nome: str) -> tuple[dict, dict, dict]:
    """(textos, gabarito de extração, spans aparados por documento)."""
    textos, gold = conjuntos.carregar(nome)
    _, linhas = spans_de(extrator)
    spans = {d: [s for s in (aparar(para_span(textos[d], x), textos[d]) for x in linhas[(nome, d)]["spans"]) if s]
             for d in textos}
    return textos, gold, spans


def triplas(spans: dict) -> dict:
    return {d: [(s.inicio, s.fim, rotulo(s)) for s in ss] for d, ss in spans.items()}


# ---------------------------------------------------------------- filtros

def tem_chave(s) -> bool:
    """O resolver tira do trecho uma chave para consultar o acervo? Sem chave (sem número de
    processo, súmula sem número, artigo sem diploma), a citação não é verificável."""
    t = s.trecho
    if s.forma == "vaga":
        return True
    if s.forma == "sumula":
        m = _SUM_NUM.search(t)
        return bool(m and _numero_ocr(m.group(2)))
    if s.forma == "artigo":
        return bool(_ART_NUM.search(t)) and _chave_lei_da_citacao(t) is not None
    return bool(nucleo_numerico(t))                       # processo, cnj, tema


def _com_digito(s) -> bool:
    return s.forma == "vaga" or any(c.isdigit() for c in s.trecho)


def _minimo(k):
    return lambda s: s.forma == "vaga" or len(s.trecho) >= k


def _conf(t):
    return lambda s: s.confianca is None or s.confianca >= t


def por_span(manter):
    """Filtro que decide span a span."""
    return lambda ss, regua: [s for s in ss if manter(s)]


def vaga_colada(k):
    """Descarta VAGA a até k caracteres de outro span do documento. Na redação da organização a
    referência vaga é uma frase própria ("julgado do STJ proferido em 2021 pela relatoria de
    ..."); em ementa real, "Rel. Min. Fulano, julgado em ..." é o rabo de um precedente numerado."""
    def f(ss, regua):
        def longe(s):
            return all(max(o.inicio - s.fim, s.inicio - o.fim) > k for o in ss if o is not s)
        return [s for s in ss if s.forma != "vaga" or longe(s)]
    return f


def troca_regua(t):
    """Span com confiança < t sai; no lugar entra o span da régua que o cruza, se houver e se não
    cruzar nenhum span confiante. No estilo da organização nenhum acerto fica abaixo de 0,99,
    então nada muda ali por construção; fora dele, a régua cobre onde o modelo hesita."""
    def f(ss, regua):
        fortes = [s for s in ss if s.confianca is None or s.confianca >= t]
        fracos = [s for s in ss if not (s.confianca is None or s.confianca >= t)]
        extra = [r for r in regua if any(r.inicio < w.fim and w.inicio < r.fim for w in fracos)
                 and not any(r.inicio < s.fim and s.inicio < r.fim for s in fortes)]
        return sorted(fortes + extra, key=lambda s: s.inicio)
    return f


def em_serie(*fs):
    def f(ss, regua):
        for g in fs:
            ss = g(ss, regua)
        return ss
    return f


FILTROS = {
    "nenhum": por_span(lambda s: True),
    "com_digito": por_span(_com_digito),
    "com_chave": por_span(tem_chave),
    **{f"min{k}": por_span(_minimo(k)) for k in (6, 8, 10, 12)},
    **{f"conf{t}": por_span(_conf(t)) for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99)},
    **{f"vaga_colada{k}": vaga_colada(k) for k in (0, 2, 3)},
    **{f"troca_regua{t}": troca_regua(t) for t in (0.9, 0.95)},
    "conf0.9+vaga_colada2": em_serie(por_span(_conf(0.9)), vaga_colada(2)),
    "conf0.95+vaga_colada2": em_serie(por_span(_conf(0.95)), vaga_colada(2)),
    "vaga_colada2+conf0.9": em_serie(vaga_colada(2), por_span(_conf(0.9))),
    "vaga_colada2+troca_regua0.9": em_serie(vaga_colada(2), troca_regua(0.9)),
    "vaga_colada2+troca_regua0.95": em_serie(vaga_colada(2), troca_regua(0.95)),
    "conf0.9+min10": em_serie(por_span(_conf(0.9)), por_span(_minimo(10))),
}


def _detalhe(r: dict) -> dict:
    out = {"f1": r["f1"]}
    for t, v in r["por_tipo"].items():
        p = v["tp"] / max(1, v["tp"] + v["fp"])
        rc = v["tp"] / max(1, v["tp"] + v["fn"])
        out[t] = {"f1": v["f1"], "p": round(p, 3), "r": round(rc, 3), "tp": v["tp"], "fp": v["fp"], "fn": v["fn"]}
    return out


def aplicar(nome_f: str, spans: dict, regua: dict) -> dict:
    return {d: FILTROS[nome_f](ss, regua.get(d, [])) for d, ss in spans.items()}


def filtros() -> dict:
    idx = construir(DB)
    dados = {n: carregar("gama", n) for n in conjuntos_nomes()}
    reguas = {n: carregar("regua", n)[2] for n in conjuntos_nomes()}
    saida = {}
    for nome_f in FILTROS:
        linha = {}
        for n, (textos, gold, spans) in dados.items():
            filtrados = aplicar(nome_f, spans, reguas[n])
            mudou = sum(set(map(id, spans[d])) != set(map(id, filtrados[d])) for d in spans)
            removidos = sum(len(spans[d]) - len(filtrados[d]) for d in spans)
            r = extracao(gold, triplas(filtrados), juntar_vaga=n == "reais")
            linha[n] = {"docs_mudados": mudou, "removidos": removidos, **_detalhe(r)}
            if n != "reais" and mudou:
                # Mesmos spans = mesma nota oficial; só recalcula quando algum documento mudou.
                linha[n]["oficial"] = oficial(n, f"gama_{nome_f}", textos, filtrados, idx)["final"]
        saida[nome_f] = linha
        print(f"{nome_f:28s}", " | ".join(
            f"{n} mud {v['docs_mudados']:3d} f1 {v['f1']:.4f}" + (f" of {v['oficial']:.5f}" if "oficial" in v else "")
            for n, v in linha.items()), flush=True)
    return saida


def conjuntos_nomes() -> list[str]:
    return ["dev", "estresse", "reais"]


# ---------------------------------------------------------------- vieses

def _iou(a, b, pa, pb) -> float:
    return max(0, min(b, pb) - max(a, pa)) / (max(b, pb) - min(a, pa))


def casar(gold: dict, pred: dict, criterio, com_tipo: bool = True) -> dict:
    """TP, FP, FN por documento (casamento guloso 1 para 1, VAGA conta como JURIS)."""
    def norm(t):
        return "JURIS" if t == "VAGA" else t
    out = {}
    for doc in gold:
        gs = [(a, b, norm(t)) for a, b, t in gold[doc]]
        ps = [(a, b, norm(t)) for a, b, t in pred.get(doc, [])]
        usados, tp = set(), 0
        for a, b, t in gs:
            k = next((k for k, (pa, pb, pt) in enumerate(ps) if k not in usados
                      and (pt == t or not com_tipo) and criterio(a, b, pa, pb)), None)
            if k is not None:
                usados.add(k)
                tp += 1
        out[doc] = (tp, len(ps) - tp, len(gs) - tp)
    return out


def f1_de(contagens, docs, pesos=None) -> float:
    tp = fp = fn = 0.0
    for d in docs:
        w = pesos[d] if pesos else 1.0
        a, b, c = contagens[d]
        tp, fp, fn = tp + w * a, fp + w * b, fn + w * c
    return round(2 * tp / max(1e-9, 2 * tp + fp + fn), 4)


CRITERIOS = {
    "iou50": lambda a, b, pa, pb: _iou(a, b, pa, pb) >= 0.5,
    "sobrepoe": lambda a, b, pa, pb: min(b, pb) > max(a, pa),
    "exato": lambda a, b, pa, pb: (a, b) == (pa, pb),
}


def estratos() -> tuple[dict, dict, dict, dict]:
    """(estrato de cada ementa anotada, peso = tamanho do estrato na população / na amostra).
    Estrato = o Gama v1 e a régua devolviam spans diferentes (a regra de `reais/anotar.py`)."""
    pop = collections.Counter()
    for r in map(json.loads, PREDICOES_REAIS.open(encoding="utf-8")):
        g = {tuple(s[:2]) for s in r["gama"]}
        rg = {tuple(s[:2]) for s in r["regua"]}
        pop["divergente" if g != rg else "concordante"] += 1
    estrato = {r["id"]: ("divergente" if r["divergente_gama_regua"] else "concordante")
               for r in map(json.loads, PRATA.open(encoding="utf-8"))}
    _, gold = conjuntos.carregar("reais")
    amostra = collections.Counter(estrato[d] for d in gold)
    return estrato, {d: pop[estrato[d]] / amostra[estrato[d]] for d in gold}, dict(pop), dict(amostra)


def vieses(melhor_filtro: str) -> dict:
    textos, gold = conjuntos.carregar("reais")
    preds = {}
    for ext in ("gama", "regua", "qwen", "gliner"):
        _, _, spans = carregar(ext, "reais")
        preds[ext] = triplas(spans)
        if ext == "gama" and melhor_filtro != "nenhum":
            preds[f"gama+{melhor_filtro}"] = triplas(aplicar(melhor_filtro, spans, carregar("regua", "reais")[2]))
    estrato, pesos, pop, amostra = estratos()
    docs = sorted(gold)
    por_estrato = {e: [d for d in docs if estrato[d] == e] for e in ("divergente", "concordante")}
    out = {"populacao": pop, "amostra": amostra, "ouro_por_estrato": {
        e: sum(len(gold[d]) for d in ds) for e, ds in por_estrato.items()}, "extratores": {}}
    rng = random.Random(0)
    reamostras = [[rng.choice(docs) for _ in docs] for _ in range(2000)]
    base = {}
    for ext, pred in preds.items():
        r = {}
        for nome_c, crit in CRITERIOS.items():
            cont = casar(gold, pred, crit)
            r[nome_c] = f1_de(cont, docs)
            if nome_c == "iou50":
                base[ext] = cont
                r["iou50_sem_tipo"] = f1_de(casar(gold, pred, crit, com_tipo=False), docs)
                r["iou50_ponderado"] = f1_de(cont, docs, pesos)
                for e, ds in por_estrato.items():
                    r[f"iou50_{e}"] = f1_de(cont, ds)
                boot = sorted(f1_de(cont, amostra_) for amostra_ in reamostras)
                r["iou50_ic95"] = [boot[50], boot[1949]]
        out["extratores"][ext] = r
        print(ext, json.dumps(r), flush=True)
    # Diferença pareada por ementa: a ordem entre dois extratores sobrevive à reamostragem?
    pares = {}
    ref = f"gama+{melhor_filtro}" if melhor_filtro != "nenhum" else "gama"
    for a, b in (("qwen", "gama"), ("regua", "gama"), ("qwen", ref), ("regua", ref)):
        if a == b or b not in base:
            continue
        difs = sorted(f1_de(base[a], s) - f1_de(base[b], s) for s in reamostras)
        pares[f"{a}-{b}"] = {"dif": round(f1_de(base[a], docs) - f1_de(base[b], docs), 4),
                             "ic95": [round(difs[50], 4), round(difs[1949], 4)],
                             "p_a_melhor": round(sum(x > 0 for x in difs) / len(difs), 3)}
    out["pares"] = pares
    print(json.dumps(pares), flush=True)
    return out


def selecao_honesta(inofensivos: list[str]) -> dict:
    """O ganho do melhor filtro sobrevive a escolher o filtro numa metade das ementas e medir
    na outra? Só concorrem os filtros que não mudam nada no dev nem no estresse."""
    textos, gold = conjuntos.carregar("reais")
    _, _, spans = carregar("gama", "reais")
    regua = carregar("regua", "reais")[2]
    cont = {f: casar(gold, triplas(aplicar(f, spans, regua)), CRITERIOS["iou50"]) for f in inofensivos}
    docs = sorted(gold)
    out = {}
    for semente in range(5):
        rng = random.Random(semente)
        embaralhados = docs[:]
        rng.shuffle(embaralhados)
        metades = (embaralhados[::2], embaralhados[1::2])
        for k, (escolha, teste) in enumerate((metades, metades[::-1])):
            melhor = max(inofensivos, key=lambda f: f1_de(cont[f], escolha))
            out[f"s{semente}m{k}"] = {"escolhido": melhor, "f1_teste": f1_de(cont[melhor], teste),
                                      "sem_filtro_teste": f1_de(cont["nenhum"], teste)}
    ganhos = [v["f1_teste"] - v["sem_filtro_teste"] for v in out.values()]
    resumo = {"ganho_medio_fora_da_escolha": round(sum(ganhos) / len(ganhos), 4),
              "ganho_minimo": round(min(ganhos), 4),
              "escolhidos": collections.Counter(v["escolhido"] for v in out.values())}
    print(json.dumps(resumo), flush=True)
    return {"rodadas": out, "resumo": resumo}


# ---------------------------------------------------------------- precisão numérica

def _ler_dtype(nome: str) -> tuple[dict, dict]:
    meta, linhas = {}, {}
    for x in (SAIDA / f"dtype_{nome}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(x)
        if "meta" in r:
            meta = r["meta"]
        else:
            linhas[(r["conjunto"], r["id"])] = r
    return meta, linhas


def precisao(nomes=("fp32", "bf16", "fp16")) -> dict:
    """FP32 x BF16 x FP16 na L4 (`bench/extrair_dtype.py`): spans iguais? quanto a confiança
    anda (faixa alta em 0,98, filtro em 0,95)? E a cauda de confiança dos acertos no final_v1,
    4.000 documentos no estilo da organização que o v1.2 não treinou."""
    dados = {n: _ler_dtype(n) for n in nomes if (SAIDA / f"dtype_{n}.jsonl").exists()}
    _, ref = dados["fp32"]
    out = {}
    for n, (meta, linhas) in dados.items():
        por_conj = collections.defaultdict(lambda: {"docs": 0, "docs_spans_diferentes": 0, "max_dconf": 0.0,
                                                    "cruza_098": 0, "cruza_095": 0, "segundos": 0.0})
        for chave, r in linhas.items():
            c = por_conj[chave[0]]
            c["docs"] += 1
            c["segundos"] += r["segundos"]
            a = {tuple(s[:4]): s[5] for s in r["spans"]}
            b = {tuple(s[:4]): s[5] for s in ref[chave]["spans"]}
            c["docs_spans_diferentes"] += a.keys() != b.keys()
            for k in a.keys() & b.keys():
                x, y = a[k] or 0, b[k] or 0
                c["max_dconf"] = max(c["max_dconf"], abs(x - y))
                c["cruza_098"] += (x >= 0.98) != (y >= 0.98)
                c["cruza_095"] += (x >= 0.95) != (y >= 0.95)
        for c in por_conj.values():
            c["s_por_doc"] = round(c.pop("segundos") / c["docs"], 4)
            c["max_dconf"] = round(c["max_dconf"], 5)
        out[n] = {"meta": meta, "conjuntos": dict(por_conj)}
        print(n, json.dumps(out[n]["conjuntos"]), flush=True)
    # Cauda de confiança dos acertos no final_v1 (FP32), contra o ouro local.
    textos, gold = conjuntos._com_gabarito_csv(pathlib.Path("/app/corpus/goldenset/v1"))
    confs, fps = [], 0
    for d in textos:
        ps = ref[("v1", d)]["spans"]
        rot = lambda s: "LEI" if s[2] == "lei" else ("VAGA" if s[3] == "vaga" else "JURIS")
        for s in ps:
            ok = any(t == rot(s) and _iou(a, b, s[0], s[1]) >= 0.5 for a, b, t in gold[d])
            if ok:
                confs.append(s[5])
            else:
                fps += 1
    confs.sort()
    out["cauda_v1"] = {"acertos": len(confs), "falsos_positivos": fps, "min": round(confs[0], 4),
                       "abaixo_098": sum(c < 0.98 for c in confs), "abaixo_095": sum(c < 0.95 for c in confs),
                       "abaixo_090": sum(c < 0.90 for c in confs), "menores": [round(c, 4) for c in confs[:10]]}
    print("cauda_v1", json.dumps(out["cauda_v1"]), flush=True)
    return out


# ---------------------------------------------------------------- auditoria

def auditoria(melhor_filtro: str, n: int = 20) -> dict:
    """Erros sorteados, com contexto, para ler à mão: o ouro está certo?"""
    textos, gold = conjuntos.carregar("reais")
    preds = {}
    for ext in ("gama", "qwen", "regua"):
        _, _, spans = carregar(ext, "reais")
        if ext == "gama":
            spans = aplicar(melhor_filtro, spans, carregar("regua", "reais")[2])
        preds[ext] = triplas(spans)
    rng = random.Random(1)
    blocos, contagem = [], {}
    for ext, pred in preds.items():
        fps, fns = [], []
        for d in sorted(gold):
            gs = [(a, b, "JURIS" if t == "VAGA" else t) for a, b, t in gold[d]]
            ps = [(a, b, "JURIS" if t == "VAGA" else t) for a, b, t in pred.get(d, [])]
            fps += [(d, p) for p in ps if not any(t == p[2] and _iou(a, b, p[0], p[1]) >= 0.5 for a, b, t in gs)]
            fns += [(d, g) for g in gs if not any(t == g[2] and _iou(g[0], g[1], a, b) >= 0.5 for a, b, t in ps)]
        contagem[ext] = {"fp": len(fps), "fn": len(fns)}
        for rotulo_, lista in (("FP", fps), ("FN", fns)):
            for d, (a, b, t) in sorted(rng.sample(lista, min(n, len(lista)))):
                tx = textos[d]
                ctx = (tx[max(0, a - 90):a] + "[[" + tx[a:b] + "]]" + tx[b:b + 90]).replace("\n", " ")
                perto = [tx[x:y] for x, y, _ in gold[d] if x < b + 40 and a - 40 < y]
                blocos.append(f"- {ext} {rotulo_} {t} `{d}` {a}-{b}: …{ctx}…  ouro perto: {perto}")
    arq = SAIDA / "auditoria.md"
    arq.write_text("\n".join(blocos) + "\n", encoding="utf-8")
    print(contagem, "->", arq)
    return contagem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=["filtros", "vieses", "selecao", "precisao", "auditoria"])
    ap.add_argument("--filtro", default="vaga_colada2+troca_regua0.95", help="filtro do Gama usado em vieses e auditoria")
    a = ap.parse_args()
    tudo = json.loads(RESULTADOS.read_text(encoding="utf-8")) if RESULTADOS.exists() else {}
    if a.acao == "filtros":
        tudo["filtros"] = filtros()
    elif a.acao == "vieses":
        tudo["vieses"] = vieses(a.filtro)
    elif a.acao == "precisao":
        tudo["precisao"] = precisao()
    elif a.acao == "selecao":
        inofensivos = [f for f, v in tudo["filtros"].items()
                       if v["dev"]["docs_mudados"] == 0 and v["estresse"]["docs_mudados"] == 0]
        tudo["selecao"] = selecao_honesta(inofensivos)
    else:
        tudo["auditoria"] = auditoria(a.filtro)
    RESULTADOS.write_text(json.dumps(tudo, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", RESULTADOS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
