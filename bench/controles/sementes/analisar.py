# -*- coding: utf-8 -*-
"""H9: estabilidade entre sementes do v1.2 (fine-tuning) e do v1.3 (destilação do v1.2 oficial).

Desenho fixado antes de treinar
    Sementes 13 (os pesos oficiais: v1.2 = vinimlo/gama@ad06ffd, v1.3 = vinimlo/gama@5f924ca), 7 e 21.
    Nenhuma semente é escolhida: as três entram em toda tabela. Os v1.3 das sementes 7 e 21 são
    destilados do MESMO professor oficial (v1.2 semente 13), então o braço v1.3 mede só a variação da
    destilação, não a do professor.
    Variantes: cru e guarda@0,95 (a de produção). O harness também grava guarda@t* (limiar escolhido
    nas 305); fica como extra, porque a pergunta é sobre a receita, não sobre o limiar.
    Regra de leitura, escrita antes dos números: a diferença v1.3 − v1.2 num conjunto "fica acima da
    dispersão entre sementes" se (a) todo v1.3 supera todo v1.2 nas 9 combinações de sementes e (b) a
    diferença das médias passa a maior amplitude (máx − mín) dentro de um braço. Só (a) = "separa, mas
    na escala do ruído de semente"; nem (a) = "não separa".
    Além disso: bootstrap pareado por ementa (2.000 reamostras, semente 0, como `bench/alunos.py`) das
    contagens somadas das três sementes de cada braço (v1.3 − v1.2), que mede a incerteza da amostra
    de ementas para a média das sementes; não mede a incerteza de semente (só três).

Entradas: spans crus de `extrair_sementes.py` (HF Jobs, L4) em
`saidas/bench/controles/_hub/bench/saida/controles/sementes/modelo_<braço>_s<semente>.jsonl`; o dev,
extraído aqui em CPU (`dev-extrair`), em `saidas/bench/controles/sementes/modelo_<braço>_s<semente>_dev.jsonl`.
Nada foi copiado do código existente: pontuação por `bench.controles.avaliar`; dev por
`bench.alunos.extrair_dev` / `solucao` / `_oficial` com `alunos.SAIDA` trocado para a pasta deste
experimento (o original grava em `saidas/bench/`, ao lado dos arquivos publicados).

    docker compose run --rm gama python -m bench.controles.sementes.analisar dev-extrair
    docker compose run --rm gama python -m bench.controles.sementes.analisar tudo    # -> resultado.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import statistics

from gama.extratores.regua import ExtratorRegua
from gama.indice import Indice

from ... import alunos, conjuntos, pontuar
from .. import avaliar

PASTA = avaliar.SAIDA / "sementes"                       # /app/saidas/bench/controles/sementes
HUB = avaliar.SAIDA / "_hub" / "bench" / "saida" / "controles" / "sementes"
MODELOS = PASTA / "_modelos"                             # pesos baixados no host, só para o dev
BRACOS = ("v12", "v13")
SEMENTES = (13, 7, 21)
PUBLICADO = {"v12": avaliar.BENCH / "modelo_professor.jsonl", "v13": avaliar.BENCH / "modelo_base12.jsonl"}
PUBLICADO_DEV = {"v12": avaliar.BENCH / "modelo_professor_dev.jsonl", "v13": avaliar.BENCH / "modelo_base12_dev.jsonl"}
VARIANTES = ("cru", "guarda@0.95")
METRICAS = (("estresse", "oficial"), ("estresse", "f1"), ("reais", "f1"), ("novas", "f1"))


def nome(braco: str, semente: int) -> str:
    return f"{braco}_s{semente}"


@contextlib.contextmanager
def trocar(modulo, atributo: str, valor):
    antes = getattr(modulo, atributo)
    setattr(modulo, atributo, valor)
    try:
        yield
    finally:
        setattr(modulo, atributo, antes)


def _chave(ss) -> list:
    return [(s.inicio, s.fim, s.tipo, s.forma) for s in ss]


# ---------------------------------------------------------------- dev (CPU, local)

def dev_extrair(so: list[str] | None = None) -> None:
    """Spans crus no dev (26 docs da organização), um modelo por vez, pelo `alunos.extrair_dev`."""
    PASTA.mkdir(parents=True, exist_ok=True)
    with trocar(alunos, "SAIDA", PASTA):
        for b in BRACOS:
            for s in SEMENTES:
                n = nome(b, s)
                if so and n not in so:
                    continue
                pasta = MODELOS / n
                if not (pasta / "config.json").exists():
                    print("sem pesos locais:", pasta, flush=True)
                    continue
                alunos.extrair_dev(n, str(pasta))


def _solucao(textos: dict, arq: pathlib.Path, conjunto: str, reg, idx) -> tuple[dict, dict]:
    _, linhas = pontuar._ler_spans(arq)
    return alunos.solucao(textos, linhas, conjunto, reg, idx)


def dev_comparar(idx) -> dict:
    """JSON final (guarda de produção -> resolver -> confiança calibrada) de cada semente contra o do
    modelo oficial do mesmo braço, documento a documento, e a métrica oficial no dev."""
    textos, _ = conjuntos.carregar("dev")
    reg = ExtratorRegua()
    out = {}
    for b in BRACOS:
        base = {}
        for rotulo, arq in (("publicado", PUBLICADO_DEV[b]), *((str(s), PASTA / f"modelo_{nome(b, s)}_dev.jsonl")
                                                              for s in SEMENTES)):
            if not arq.exists():
                continue
            spans, jsons = _solucao(textos, arq, "dev", reg, idx)
            with trocar(alunos, "SAIDA", PASTA):
                oficial = alunos._oficial("dev", f"sementes_{b}_{rotulo}", jsons)
            base[rotulo] = (spans, jsons)
            r = {"arquivo": str(arq), "oficial": oficial}
            for ref in ("publicado", "13"):
                if ref in base and ref != rotulo:
                    rs, rj = base[ref]
                    r[f"docs_json_diferente_de_{ref}"] = sum(jsons[d] != rj[d] for d in textos)
                    r[f"docs_spans_diferentes_de_{ref}"] = sum(_chave(spans[d]) != _chave(rs[d]) for d in textos)
            out.setdefault(b, {})[rotulo] = r
            print("dev", b, rotulo, json.dumps(r, ensure_ascii=False), flush=True)
    out["docs"] = len(textos)
    return out


# ---------------------------------------------------------------- estresse, 305, 172

def arquivo(braco: str, semente: int) -> pathlib.Path:
    return HUB / f"modelo_{nome(braco, semente)}.jsonl"


def reproducao_s13() -> dict:
    """Os spans da semente 13 reextraídos neste job são os publicados? (mesmos pesos, mesma L4)"""
    out = {}
    for b in BRACOS:
        _, novo = pontuar._ler_spans(arquivo(b, 13))
        _, pub = pontuar._ler_spans(PUBLICADO[b])
        for c in avaliar.CONJUNTOS:
            textos, _ = avaliar.textos_e_ouro(c)
            comuns = [d for d in textos if (c, d) in novo and (c, d) in pub]
            dif = sum([x[:5] for x in novo[(c, d)]["spans"]] != [x[:5] for x in pub[(c, d)]["spans"]] for d in comuns)
            conf = max((abs(x[5] - y[5]) for d in comuns for x, y in zip(novo[(c, d)]["spans"], pub[(c, d)]["spans"])),
                       default=0.0)
            out[f"{b}_{c}"] = {"docs": len(comuns), "docs_spans_diferentes": dif, "max_dif_confianca": conf}
    return out


def pontuar_todos(idx) -> dict:
    rel = {}
    for b in BRACOS:
        for s in SEMENTES:
            n = nome(b, s)
            r = avaliar.avaliar(f"sementes_{n}", [arquivo(b, s)], idx=idx)
            (PASTA / f"{n}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
            rel[n] = r
    return rel


def concordancia(rel: dict) -> dict:
    """Quanto cada semente muda a saída em relação à 13 do mesmo braço: documentos com spans crus
    diferentes (estresse, 305, 172) e com JSON final diferente no estresse (cru e com a guarda)."""
    out = {}
    for b in BRACOS:
        _, l13 = pontuar._ler_spans(arquivo(b, 13))
        for s in SEMENTES[1:]:
            _, ls = pontuar._ler_spans(arquivo(b, s))
            r = {}
            for c in avaliar.CONJUNTOS:
                sp13, sps = avaliar.spans(l13, c), avaliar.spans(ls, c)
                r[f"{c}_docs_spans_crus_diferentes"] = sum(_chave(sp13[d]) != _chave(sps[d]) for d in sp13)
                reg = avaliar.regua(c)
                g13, gs = avaliar.com_guarda(sp13, reg, 0.95), avaliar.com_guarda(sps, reg, 0.95)
                r[f"{c}_docs_spans_guarda_diferentes"] = sum(_chave(g13[d]) != _chave(gs[d]) for d in g13)
                r[f"{c}_docs"] = len(sp13)
            for v in VARIANTES:
                p13 = avaliar.SAIDA / "json" / "estresse" / f"sementes_{nome(b, 13)}__{v}"
                ps = avaliar.SAIDA / "json" / "estresse" / f"sementes_{nome(b, s)}__{v}"
                difs = 0
                for arq in sorted(p13.glob("*.json")):
                    a = json.loads(arq.read_text(encoding="utf-8"))["citacoes"]
                    z = json.loads((ps / arq.name).read_text(encoding="utf-8"))["citacoes"]
                    difs += a != z
                r[f"estresse_docs_json_final_diferente_{v}"] = difs
            out[nome(b, s)] = r
    return out


def _valor(r: dict, v: str, c: str, m: str) -> float:
    x = r["variantes"][v][c]
    return x["oficial"]["final"] if m == "oficial" else x["extracao"]["f1"]


def _estat(xs: list[float]) -> dict:
    return {"media": round(statistics.mean(xs), 5), "min": min(xs), "max": max(xs),
            "amplitude": round(max(xs) - min(xs), 5), "dp": round(statistics.stdev(xs), 5)}


def tabela(rel: dict) -> dict:
    out = {}
    for b in BRACOS:
        for v in VARIANTES:
            for c, m in METRICAS:
                xs = {s: _valor(rel[nome(b, s)], v, c, m) for s in SEMENTES}
                out[f"{b}|{v}|{c}_{m}"] = {"por_semente": {str(s): x for s, x in xs.items()}, **_estat(list(xs.values()))}
        out[f"{b}|limiar_escolhido_nas_305"] = {str(s): rel[nome(b, s)]["varredura_reais"]["escolhido"] for s in SEMENTES}
    return out


def _somar(conts: list[dict]) -> dict:
    """Contagens por documento de várias sementes -> um só dicionário por documento (F1 da soma)."""
    return {d: {f"{i}_{t}": v for i, c in enumerate(conts) for t, v in c[d].items()} for d in conts[0]}


def _contagens(braco: str, semente: int, conjunto: str, v: str) -> dict:
    _, linhas = avaliar.ler([arquivo(braco, semente)])
    sp = avaliar.spans(linhas, conjunto)
    fin = sp if v == "cru" else avaliar.com_guarda(sp, avaliar.regua(conjunto), 0.95)
    return avaliar.por_doc(conjunto, fin)


def vs_professor() -> dict:
    """Cada v1.3 (semente da destilação) contra o próprio professor, o v1.2 semente 13: bootstrap
    pareado por ementa. Responde se o ganho do aluno sobre ESTE professor resiste à semente da
    destilação, pergunta diferente de v1.3 contra a receita do v1.2."""
    out = {}
    for v in VARIANTES:
        for c in ("reais", "novas"):
            ref = _contagens("v12", 13, c, v)
            for s in SEMENTES:
                out[f"{v}|{c}|v13_s{s}"] = avaliar.bootstrap(_contagens("v13", s, c, v), ref)
    return out


def diferencas(rel: dict) -> dict:
    out = {}
    for v in VARIANTES:
        for c in ("reais", "novas", "estresse"):
            ms = ("f1", "oficial") if c == "estresse" else ("f1",)
            for m in ms:
                x = {b: {s: _valor(rel[nome(b, s)], v, c, m) for s in SEMENTES} for b in BRACOS}
                cruzadas = [x["v13"][i] - x["v12"][j] for i in SEMENTES for j in SEMENTES]
                media = statistics.mean(x["v13"].values()) - statistics.mean(x["v12"].values())
                amp = max(max(x[b].values()) - min(x[b].values()) for b in BRACOS)
                a, bb = min(cruzadas) > 0, media > amp
                r = {"oficial_s13_menos_s13": round(x["v13"][13] - x["v12"][13], 5),
                     "media_v13_menos_media_v12": round(media, 5),
                     "cruzadas_9": {"min": round(min(cruzadas), 5), "max": round(max(cruzadas), 5),
                                    "positivas": sum(d > 0 for d in cruzadas)},
                     "maior_amplitude_intra_braco": round(amp, 5),
                     "a_todo_v13_supera_todo_v12": a, "b_media_passa_amplitude": bb,
                     "veredito": ("acima da dispersão entre sementes" if a and bb else
                                  "separa, mas na escala do ruído de semente" if a else "não separa")}
                if c != "estresse":
                    conts = {b: _somar([_contagens(b, s, c, v) for s in SEMENTES]) for b in BRACOS}
                    r["bootstrap_tres_sementes_somadas"] = avaliar.bootstrap(conts["v13"], conts["v12"])
                out[f"{v}|{c}_{m}"] = r
    return out


def tudo() -> dict:
    PASTA.mkdir(parents=True, exist_ok=True)
    idx = Indice.do_banco(pontuar.DB)
    rel = pontuar_todos(idx)
    res = {"reproducao_s13": reproducao_s13(), "tabela": tabela(rel), "diferencas": diferencas(rel),
           "v13_contra_o_professor": vs_professor(), "concordancia_com_s13": concordancia(rel),
           "resumo_harness": {n: r["resumo"] for n, r in rel.items()},
           "s_por_doc_l4": {n: r["s_por_doc"] for n, r in rel.items()},
           "meta": {n: r["meta"] for n, r in rel.items()},
           "codigo": avaliar.impressao()}
    res["dev"] = dev_comparar(idx)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("acao", choices=["dev-extrair", "tudo"])
    ap.add_argument("--so", default="", help="dev-extrair: só estes modelos (ex.: v12_s7,v13_s7)")
    ap.add_argument("--saida", default=str(PASTA / "resultado.json"))
    a = ap.parse_args()
    if a.acao == "dev-extrair":
        dev_extrair([x for x in a.so.split(",") if x])
        return 0
    res = tudo()
    pathlib.Path(a.saida).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", a.saida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
