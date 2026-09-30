# -*- coding: utf-8 -*-
"""Harness comum dos controles: pontua um JSONL de spans, cru e com a guarda, nos três conjuntos.

Não reimplementa nada do benchmark: importa `bench.pontuar` (leitura dos spans, `para_span`,
`rotulo`, `extracao`, `oficial`), `bench.alunos` (textos e ouro das 172, F1 por documento) e a
guarda de produção (`gama.extratores.guarda.guardar`).

Conjuntos (valor do campo `conjunto` no JSONL)
    estresse  600 docs sintéticos, redação nunca treinada (N1 + N2). Métrica OFICIAL pelo caminho
              de produção: aparar -> processar (resolver, classe, confiança calibrada) ->
              vendor/kaggle_metric.py, via `bench.pontuar.oficial`. Também o F1 de extração com
              VAGA separada, como `bench/pontuar.py` faz no estresse.
    reais     305 ementas: VALIDAÇÃO. F1 de extração (mesmo tipo, IoU >= 0,5, casamento 1 para 1,
              VAGA agrupada em JURIS), `bench.pontuar.extracao`. Toda escolha (limiar) sai daqui.
    novas     172 ementas (o JSONL de entrada tem 200; valem as 172 do ouro `corpus/reais/novo`):
              CONFIRMAÇÃO. Mesma métrica das 305. Nunca informa escolha.

Variantes de cada candidato
    cru          os spans do arquivo, aparados (`gama.span.aparar`)
    guarda@0.95  a guarda de produção com o limiar de produção
    guarda@<t*>  a mesma guarda com o limiar escolhido nas 305 numa grade de 0,50 a 0,99 (passo
                 0,01). Critério: maior F1 exato nas 305; empate -> mais perto de 0,95; depois o
                 maior. Só existe se o modelo tem confiança; o t* é medido nas 172 sem reescolha.
O limiar entra trocando `guarda.CONFIANCA_MINIMA` durante a chamada: a função de produção roda
intacta. A régua da guarda são os spans gravados em `saidas/bench/regua.jsonl` (estresse, reais;
os mesmos que o benchmark publicado usou) e `saidas/bench/controles/regua_novas.jsonl` (novas;
gerado aqui na primeira vez com `ExtratorRegua`).

Bootstrap: diferença de F1 para o Gama v1.3 com a guarda de produção, pareado por ementa, 2.000
reamostras, semente 0 (o procedimento de `bench/alunos.py`: IC95 = reamostras 50 e 1.949). A
referência são os spans de `saidas/bench/gama.jsonl` (vinimlo/gama@5f924ca; estresse e reais) e
`saidas/bench/modelo_base12.jsonl` (os mesmos pesos antes de publicados; as 172).

FORMATO DO JSONL DE SPANS (o de `bench/extrair_alunos.py`; `bench/pontuar.py` lê igual)
    1ª linha: {"meta": {"extrator": <nome>, "modelo": "<repo>@<rev7>", "dispositivo": ..., ...}}
    depois, uma linha por documento:
      {"conjunto": "estresse"|"reais"|"novas", "id": "<id>", "segundos": <float>, "spans": [...]}
    cada span é um de
      [inicio, fim, tipo, forma, digitos, confianca]  Span completo, como `ExtratorNeural.extrair`
                                                      devolve (tipo jurisprudencia|lei; forma
                                                      cnj|processo|sumula|tema|artigo|vaga)
      [inicio, fim, "JURIS"|"LEI"|"VAGA"]             extrator de fora (Qwen)
      [inicio, fim, "JURIS"|"LEI"|"VAGA", score]      extrator de fora com score (GLiNER)
    Offsets em codepoints Python do campo `texto` de `bench/entrada_remota.jsonl` (estresse +
    reais) e `bench/entrada_novas.jsonl` (novas) do dataset vinimlo/gama-goldenset. Um conjunto
    com documento faltando é pulado (e contado em `presentes`); ids a mais são ignorados.
    Na métrica oficial, o Span completo entra com a confiança do modelo (faixa alta/baixa da
    calibração), como o Gama; o formato curto entra sem confiança (faixa "regra"), como
    `bench/pontuar.py` faz com GLiNER e Qwen. O score do formato curto só serve à guarda.

COMO UM EXPERIMENTO GRAVA OS SPANS (HF Jobs; o dev não sai da máquina e não entra aqui)
    1. No job (cabeçalho PEP 723 de `bench/extrair_alunos.py`, torch cu126):
       raiz = snapshot_download("vinimlo/gama-goldenset", repo_type="dataset",
           revision="2fff5f670e13f77f339937cfe3da52ed0af9572d",
           allow_patterns=["bench/entrada_remota.jsonl", "bench/entrada_novas.jsonl", "bench/codigo/**"])
       sys.path.insert(0, str(raiz / "bench" / "codigo"))
       from gama.extratores.neural import ExtratorNeural   # idêntico a src/ (neural, bio, formas, span)
       Modelo de classificação de tokens BIO salvo com `save_pretrained` (+ tokenizer) roda por
       `ExtratorNeural(pasta)` e sai no formato completo. O snapshot `bench/codigo` NÃO tem a guarda
       e tem um resolver antigo: o job só extrai; guarda, resolver e métrica rodam aqui.
    2. Uma linha por documento das duas entradas (905 + 200), `segundos` medido com
       `torch.cuda.synchronize()` antes e depois, e a linha `meta` primeiro (modelo@revisão,
       dispositivo, torch, parâmetros), exatamente como `bench/extrair_alunos.py`.
    3. Sobe com HfApi().upload_file(path_or_fileobj=..., repo_id="vinimlo/gama-goldenset",
       repo_type="dataset", path_in_repo=f"bench/saida/controles/<experimento>/modelo_<nome>.jsonl")
       e imprime `REVISAO <oid>`. Nada fora de `bench/saida/controles/`.
    4. Aqui: hf download vinimlo/gama-goldenset bench/saida/controles/<experimento>/modelo_<nome>.jsonl
       --repo-type dataset --revision <oid> --local-dir saidas/bench/controles/_hub
       e pontua com a linha abaixo, apontando `--spans` para o arquivo baixado.

    docker compose run --rm gama python -m bench.controles.avaliar pontuar \\
        --nome probe --spans saidas/bench/controles/probe/modelo_probe.jsonl
    docker compose run --rm gama python -m bench.controles.avaliar validar

Saída: `saidas/bench/controles/<nome>.json` (`validacao.json` na validação); os JSON por
documento da métrica oficial em `saidas/bench/controles/json/estresse/<nome>__<variante>/`.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import pathlib
import random
import time

from gama.extratores import guarda
from gama.extratores.regua import ExtratorRegua
from gama.formas import span_de
from gama.indice import construir
from gama.span import aparar_todos

from .. import alunos, pontuar

APP = pathlib.Path("/app")
BENCH = pontuar.SAIDA                                 # /app/saidas/bench
SAIDA = BENCH / "controles"
CONJUNTOS = ("estresse", "reais", "novas")
GRADE = [round(0.50 + 0.01 * i, 2) for i in range(50)]  # 0,50 a 0,99
LIMIAR_PRODUCAO = guarda.CONFIANCA_MINIMA             # 0,95, lido antes de qualquer troca
REAMOSTRAS, SEMENTE = 2000, 0
REFERENCIA = {"estresse": BENCH / "gama.jsonl", "reais": BENCH / "gama.jsonl",
              "novas": BENCH / "modelo_base12.jsonl"}
REGUA = {"estresse": BENCH / "regua.jsonl", "reais": BENCH / "regua.jsonl",
         "novas": SAIDA / "regua_novas.jsonl"}
CODIGO = ["src/gama/extratores/guarda.py", "src/gama/extratores/regua.py", "src/gama/pipeline.py",
          "src/gama/classificar.py", "src/gama/calibracao.json", "src/gama/resolver.py",
          "src/gama/formas.py", "src/gama/indice.py", "bench/pontuar.py", "bench/alunos.py",
          "bench/conjuntos.py", "vendor/kaggle_metric.py", "avaliacao/harness.py"]


# ---------------------------------------------------------------- leitura

_TEXTOS: dict = {}


def textos_e_ouro(conjunto: str) -> tuple[dict, dict]:
    """(textos por id, ouro de extração por id). estresse/reais via `bench.conjuntos`, novas
    via `bench.alunos` (ouro de `corpus/reais/novo`, 172 ids)."""
    if conjunto not in _TEXTOS:
        _TEXTOS[conjunto] = alunos._textos_e_ouro(conjunto)
    return _TEXTOS[conjunto]


def ler(arquivos) -> tuple[dict, dict]:
    """(meta por arquivo, {(conjunto, id): linha}); arquivos posteriores sobrescrevem."""
    metas, linhas = {}, {}
    for arq in map(pathlib.Path, arquivos):
        meta, ls = pontuar._ler_spans(arq)
        metas[arq.name] = meta
        linhas.update(ls)
    return metas, linhas


def formato(linhas: dict) -> tuple[bool, bool]:
    """(Span completo?, tem confiança/score?)."""
    completo = score = False
    for r in linhas.values():
        for s in r["spans"]:
            completo |= len(s) == 6
            score |= (len(s) == 6 and s[5] is not None) or (len(s) == 4 and s[3] is not None)
    return completo, score


def _span(texto: str, s: list):
    """Linha gravada -> Span. `bench.pontuar.para_span` descarta o score do formato curto (a
    métrica oficial não o usa); a guarda precisa dele, então aqui ele vira a confiança."""
    if len(s) == 4 and s[3] is not None:
        return span_de(texto, s[0], s[1], s[2], s[3])
    return pontuar.para_span(texto, s)


def spans(linhas: dict, conjunto: str) -> dict | None:
    """Spans aparados por documento do conjunto, ou None se falta algum documento."""
    textos, _ = textos_e_ouro(conjunto)
    if any((conjunto, d) not in linhas for d in textos):
        return None
    return {d: aparar_todos([_span(t, s) for s in linhas[(conjunto, d)]["spans"]], t) for d, t in textos.items()}


_REGUA: dict = {}


def _gravar_regua_novas(arq: pathlib.Path) -> None:
    textos, _ = textos_e_ouro("novas")
    ext = ExtratorRegua()
    arq.parent.mkdir(parents=True, exist_ok=True)
    with arq.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": {"extrator": "regua", "modelo": "regras", "codigo": impressao()}}) + "\n")
        for d, t in textos.items():
            t0 = time.perf_counter()
            ss = ext.extrair(t)
            fh.write(json.dumps({"conjunto": "novas", "id": d, "segundos": round(time.perf_counter() - t0, 4),
                                 "spans": [[s.inicio, s.fim, s.tipo, s.forma, s.digitos, s.confianca] for s in ss]},
                                ensure_ascii=False) + "\n")


def regua(conjunto: str) -> dict:
    """Spans da régua aparados, da gravação usada no benchmark publicado."""
    if conjunto not in _REGUA:
        if conjunto == "novas" and not REGUA["novas"].exists():
            _gravar_regua_novas(REGUA["novas"])
        _REGUA[conjunto] = spans(pontuar._ler_spans(REGUA[conjunto])[1], conjunto)
    return _REGUA[conjunto]


# ---------------------------------------------------------------- guarda e métricas

@contextlib.contextmanager
def limiar(t: float):
    """A guarda de produção com outro limiar: `guardar` lê o global a cada chamada."""
    antes = guarda.CONFIANCA_MINIMA
    guarda.CONFIANCA_MINIMA = t
    try:
        yield
    finally:
        guarda.CONFIANCA_MINIMA = antes


def com_guarda(sp: dict, reg: dict, t: float) -> dict:
    with limiar(t):
        return {d: guarda.guardar(ss, reg[d]) for d, ss in sp.items()}


def triplas(sp: dict) -> dict:
    return {d: [(s.inicio, s.fim, pontuar.rotulo(s)) for s in ss] for d, ss in sp.items()}


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / max(1, 2 * tp + fp + fn)


def extracao(conjunto: str, sp: dict) -> dict:
    """`bench.pontuar.extracao` (VAGA em JURIS fora do estresse) + P/R e F1 exato."""
    _, gold = textos_e_ouro(conjunto)
    r = pontuar.extracao(gold, triplas(sp), juntar_vaga=conjunto != "estresse")
    tot = {k: sum(v[k] for v in r["por_tipo"].values()) for k in ("tp", "fp", "fn")}
    r["f1_exato"] = _f1(tot["tp"], tot["fp"], tot["fn"])
    r["p"] = round(tot["tp"] / max(1, tot["tp"] + tot["fp"]), 4)
    r["r"] = round(tot["tp"] / max(1, tot["tp"] + tot["fn"]), 4)
    r.update(tot)
    for v in r["por_tipo"].values():
        v["p"] = round(v["tp"] / max(1, v["tp"] + v["fp"]), 4)
        v["r"] = round(v["tp"] / max(1, v["tp"] + v["fn"]), 4)
    return r


def oficial(rotulo: str, sp: dict, idx, completo: bool) -> dict:
    """Métrica oficial no estresse por `bench.pontuar.oficial`, com os JSON em SAIDA/json. O
    formato curto entra sem confiança do modelo, como no benchmark publicado."""
    if not completo:
        sp = {d: [dataclasses.replace(s, confianca=None) for s in ss] for d, ss in sp.items()}
    textos, _ = textos_e_ouro("estresse")
    antes = pontuar.SAIDA
    pontuar.SAIDA = SAIDA
    try:
        return pontuar.oficial("estresse", rotulo, textos, sp, idx)
    finally:
        pontuar.SAIDA = antes


def por_doc(conjunto: str, sp: dict) -> dict:
    _, gold = textos_e_ouro(conjunto)
    return alunos._por_doc(gold, triplas(sp))


def bootstrap(cont: dict, ref: dict) -> dict:
    """F1(candidato) - F1(referência), IC95 pareado por documento, como `bench/alunos.py`."""
    docs = sorted(cont)
    rng = random.Random(SEMENTE)
    difs = sorted(alunos._f1(cont, s) - alunos._f1(ref, s)
                  for s in ([rng.choice(docs) for _ in docs] for _ in range(REAMOSTRAS)))
    return {"dif": round(alunos._f1(cont, docs) - alunos._f1(ref, docs), 4),
            "ic95": [round(difs[int(0.025 * REAMOSTRAS)], 4), round(difs[int(0.975 * REAMOSTRAS) - 1], 4)],
            "frac_dif_menor_ou_igual_zero": round(sum(x <= 0 for x in difs) / REAMOSTRAS, 4),
            "reamostras": REAMOSTRAS, "semente": SEMENTE}


_REF: dict = {}


def referencia(conjunto: str) -> dict:
    """Contagens por documento do Gama v1.3 com a guarda de produção."""
    if conjunto not in _REF:
        sp = spans(pontuar._ler_spans(REFERENCIA[conjunto])[1], conjunto)
        _REF[conjunto] = por_doc(conjunto, com_guarda(sp, regua(conjunto), LIMIAR_PRODUCAO))
    return _REF[conjunto]


def varrer(sp_reais: dict, grade=GRADE) -> tuple[float, dict]:
    """F1 nas 305 com a guarda em cada limiar da grade -> (limiar escolhido, tabela)."""
    reg = regua("reais")
    exatos = {t: extracao("reais", com_guarda(sp_reais, reg, t))["f1_exato"] for t in grade}
    melhor = max(exatos.values())
    escolhido = min((t for t, f in exatos.items() if f == melhor), key=lambda t: (abs(t - LIMIAR_PRODUCAO), -t))
    return escolhido, {f"{t:.2f}": round(f, 4) for t, f in exatos.items()}


def impressao() -> dict:
    """sha256 (12) do código que decide o número: o Gabriel refatora em paralelo."""
    return {c: hashlib.sha256((APP / c).read_bytes()).hexdigest()[:12] for c in CODIGO if (APP / c).exists()}


# ---------------------------------------------------------------- avaliação de um candidato

def variante(t: float | None) -> str:
    """Nome da variante: "cru" ou "guarda@0.95" (duas casas; mais, se o limiar tiver)."""
    if t is None:
        return "cru"
    return f"guarda@{t:.2f}" if round(t, 2) == t else f"guarda@{t}"


def resumo(out: dict) -> dict:
    """Uma linha plana por variante, para tabelas: oficial e F1 no estresse, F1 e diferença
    para o Gama v1.3 com guarda (IC95) nas 305 e nas 172."""
    linhas = {}
    for v, res in out["variantes"].items():
        x = {}
        for c, r in res.items():
            x[f"{c}_f1"] = r["extracao"]["f1"]
            if "oficial" in r:
                x[f"{c}_oficial"] = r["oficial"]["final"]
            if "vs_gama_v13_guarda" in r:
                x[f"{c}_dif_v13"] = r["vs_gama_v13_guarda"]["dif"]
                x[f"{c}_ic95_v13"] = r["vs_gama_v13_guarda"]["ic95"]
        linhas[v] = x
    return linhas

def avaliar(nome: str, arquivos: list, limiares_extra=(), com_oficial: bool = True, grade=GRADE,
            idx=None) -> dict:
    metas, linhas = ler(arquivos)
    completo, com_score = formato(linhas)
    out = {"nome": nome, "arquivos": [str(a) for a in arquivos], "meta": metas,
           "formato": "completo" if completo else "curto", "tem_confianca": com_score,
           "codigo": impressao(), "presentes": {}, "s_por_doc": {}, "variantes": {}}
    sp = {}
    for c in CONJUNTOS:
        textos, _ = textos_e_ouro(c)
        faltam = sum((c, d) not in linhas for d in textos)
        out["presentes"][c] = {"documentos": len(textos) - faltam, "faltam": faltam}
        if faltam == 0:
            sp[c] = spans(linhas, c)
            out["s_por_doc"][c] = round(sum(linhas[(c, d)].get("segundos", 0.0) for d in textos) / len(textos), 4)

    variantes = {"cru": None, variante(LIMIAR_PRODUCAO): LIMIAR_PRODUCAO}
    if "reais" in sp and com_score:
        t, tabela = varrer(sp["reais"], grade)
        out["varredura_reais"] = {"escolhido": t, "variante": variante(t), "criterio": "maior F1 exato nas 305; "
                                  "empate: mais perto de 0,95, depois o maior", "f1_por_limiar": tabela}
        variantes[variante(t)] = t
    else:
        out["varredura_reais"] = {"escolhido": None, "nota": "sem as 305 ou sem confiança do modelo: o limiar "
                                  "não tem efeito (a guarda só tira VAGA colada)"}
    for t in limiares_extra:
        variantes[variante(t)] = t
    if com_oficial and "estresse" in sp and idx is None:
        idx = construir(pontuar.DB)

    for v, t in variantes.items():
        res = {}
        for c, s in sp.items():
            finais = s if t is None else com_guarda(s, regua(c), t)
            r = {"extracao": extracao(c, finais)}
            if c == "estresse" and com_oficial:
                r["oficial"] = oficial(f"{nome}__{v}", finais, idx, completo)
            if c in ("reais", "novas"):
                r["vs_gama_v13_guarda"] = bootstrap(por_doc(c, finais), referencia(c))
            res[c] = r
        out["variantes"][v] = res
        print(nome, v, " | ".join(
            f"{c} F1 {r['extracao']['f1']:.4f}".replace(".", ",")
            + (f" oficial {r['oficial']['final']:.5f}".replace(".", ",") if "oficial" in r else "")
            for c, r in res.items()), flush=True)
    out["resumo"] = resumo(out)
    return out


# ---------------------------------------------------------------- validação

# (candidato, conjunto, variante, métrica, valor publicado como escrito, fonte)
PUBLICADOS = [
    ("gama_v13", "estresse", "cru", "oficial", "1.09999", "bench/resultados.json gama"),
    ("gama_v13", "estresse", "guarda@0.95", "oficial", "1.09999", "bench/resultados.json gama_guarda"),
    ("gama_v13", "reais", "cru", "f1", "0.6201", "bench/resultados.json gama"),
    ("gama_v13", "reais", "guarda@0.95", "f1", "0.8180", "bench/resultados.json gama_guarda"),
    ("regua", "estresse", "cru", "oficial", "0.84880", "bench/resultados.json regua"),
    ("regua", "reais", "cru", "f1", "0.6627", "bench/resultados.json regua"),
    ("qwen", "estresse", "cru", "oficial", "0.81065", "bench/resultados.json qwen"),
    ("qwen", "reais", "cru", "f1", "0.7073", "bench/resultados.json qwen"),
    ("gliner", "estresse", "cru", "oficial", "0.32732", "bench/resultados.json gliner"),
    ("gliner", "reais", "cru", "f1", "0.4093", "bench/resultados.json gliner"),
    ("v12", "novas", "guarda@0.95", "f1", "0.8204", "bench/alunos.json professor novas (0,820)"),
    ("v12", "novas", "cru", "f1", "0.615", "D-008 (CPU local, novas_professor.json)"),
    ("v13_base12", "novas", "guarda@0.95", "f1", "0.8392", "bench/alunos.json base12 novas (0,839)"),
    # conferências extras, fora da lista obrigatória
    ("gama_v13", "estresse", "cru", "f1", "1.0", "bench/resultados.json gama (extração)"),
    ("regua", "estresse", "cru", "f1", "0.8582", "bench/resultados.json regua (extração)"),
    ("qwen", "estresse", "cru", "f1", "0.8246", "bench/resultados.json qwen (extração)"),
    ("gliner", "estresse", "cru", "f1", "0.4531", "bench/resultados.json gliner (extração)"),
    ("v12", "reais", "guarda@0.95", "f1", "0.8076", "bench/alunos.json professor reais"),
    ("v12", "reais", "cru", "f1", "0.605", "D-008 (v1.2 sozinho, 305)"),
    ("v12", "estresse", "guarda@0.95", "oficial", "1.09999", "bench/alunos.json professor estresse"),
    ("v13_base12", "reais", "guarda@0.95", "f1", "0.8180", "bench/alunos.json base12 reais"),
    ("v13_base12", "estresse", "guarda@0.95", "oficial", "1.09999", "bench/alunos.json base12 estresse"),
    ("regua", "novas", "cru", "f1", "0.679", "D-008 (régua nas 172)"),
]
CANDIDATOS_VALIDACAO = {
    "gama_v13": ["gama.jsonl"], "regua": ["regua.jsonl", "controles/regua_novas.jsonl"],
    "qwen": ["qwen.jsonl"], "gliner": ["gliner.jsonl"],
    "v12": ["modelo_professor.jsonl"], "v13_base12": ["modelo_base12.jsonl"],
}


def _confere(medido: float, publicado: str) -> dict:
    casas = len(publicado.split(".")[1]) if "." in publicado else 0
    pub = float(publicado)
    tol = max(0.5 * 10 ** -casas, 5e-5)
    return {"publicado": pub, "medido": medido, "dif": round(medido - pub, 6), "tolerancia": tol,
            "ok": abs(medido - pub) <= tol + 1e-12}


def _sanidade() -> dict:
    """Textos que os extratores viram = textos da pontuação; régua gravada = régua do código atual;
    referência do Gama v1.3 (gama.jsonl) = base12 onde os dois existem."""
    out = {}
    for arq, conjs in (("entrada_remota.jsonl", ("estresse", "reais")), ("entrada_novas.jsonl", ("novas",))):
        entrada = {(r["conjunto"], r["id"]): r["texto"] for r in map(json.loads, (BENCH / arq).open(encoding="utf-8"))}
        for c in conjs:
            textos, _ = textos_e_ouro(c)
            out[f"textos_iguais_{c}"] = {"docs": len(textos),
                                         "diferentes": sum(entrada.get((c, d)) != t for d, t in textos.items())}
    ext = ExtratorRegua()
    for c in CONJUNTOS:
        textos, _ = textos_e_ouro(c)
        gravada = regua(c)
        agora = {d: aparar_todos(ext.extrair(t), t) for d, t in textos.items()}
        out[f"regua_gravada_igual_codigo_atual_{c}"] = {
            "docs": len(textos),
            "diferentes": sum([(s.inicio, s.fim, s.tipo, s.forma) for s in gravada[d]]
                              != [(s.inicio, s.fim, s.tipo, s.forma) for s in agora[d]] for d in textos)}
    _, g = pontuar._ler_spans(BENCH / "gama.jsonl")
    _, b = pontuar._ler_spans(BENCH / "modelo_base12.jsonl")
    for c in ("estresse", "reais"):
        sg, sb = spans(g, c), spans(b, c)
        gg, gb = com_guarda(sg, regua(c), LIMIAR_PRODUCAO), com_guarda(sb, regua(c), LIMIAR_PRODUCAO)
        chave = lambda ss: [(s.inicio, s.fim, s.tipo, s.forma) for s in ss]  # noqa: E731
        out[f"gama_jsonl_igual_base12_{c}"] = {
            "docs": len(sg), "cru_diferentes": sum(chave(sg[d]) != chave(sb[d]) for d in sg),
            "guarda_diferentes": sum(chave(gg[d]) != chave(gb[d]) for d in sg),
            "max_dif_confianca": round(max((abs((x.confianca or 0) - (y.confianca or 0))
                                            for d in sg for x, y in zip(sg[d], sb[d])), default=0.0), 8)}
    # guarda com limiar trocado para 0,95 = guarda de produção sem troca
    sg = spans(g, "reais")
    sem_troca = {d: guarda.guardar(ss, regua("reais")[d]) for d, ss in sg.items()}
    out["guarda_limiar_095_igual_producao"] = sem_troca == com_guarda(sg, regua("reais"), 0.95)
    return out


def validar() -> dict:
    idx = construir(pontuar.DB)
    regua("novas")                                  # grava controles/regua_novas.jsonl se faltar
    relatorios = {n: avaliar(n, [BENCH / a for a in arqs], idx=idx) for n, arqs in CANDIDATOS_VALIDACAO.items()}
    for n, rel in relatorios.items():
        (SAIDA / f"validacao_{n}.json").write_text(json.dumps(rel, ensure_ascii=False, indent=1), encoding="utf-8")
    conferencias = []
    for cand, c, v, met, pub, fonte in PUBLICADOS:
        r = relatorios[cand]["variantes"][v][c]
        medido = r["oficial"]["final"] if met == "oficial" else r["extracao"]["f1"]
        conferencias.append({"candidato": cand, "conjunto": c, "variante": v, "metrica": met, "fonte": fonte,
                             **_confere(medido, pub)})
        print(f"{'ok ' if conferencias[-1]['ok'] else 'NÃO'} {cand:11s} {c:9s} {v:12s} {met:8s} "
              f"publicado {pub:>8s} medido {medido}", flush=True)
    # v1.2 cru nas 172 medido em CPU (o número publicado): previsões gravadas em novas_professor.json
    npj = json.loads((BENCH / "novas_professor.json").read_text(encoding="utf-8"))
    _, gold = textos_e_ouro("novas")
    cpu = {k: pontuar.extracao(gold, {d: [tuple(x) for x in ss] for d, ss in npj[k].items()}, True)["f1"]
           for k in ("neural-cru", "neural", "regua")}
    # bootstrap: v1.3 contra v1.2, ambos com a guarda (publicado em bench/alunos.json, base12)
    boot = {}
    for c, pub in (("reais", (0.0104, [-0.0049, 0.0258])), ("novas", (0.0188, [0.003, 0.0331]))):
        cont = {}
        for n in ("v12", "v13_base12"):
            _, linhas = ler([BENCH / a for a in CANDIDATOS_VALIDACAO[n]])
            cont[n] = por_doc(c, com_guarda(spans(linhas, c), regua(c), LIMIAR_PRODUCAO))
        b = bootstrap(cont["v13_base12"], cont["v12"])
        boot[c] = {"medido": b, "publicado": {"dif": pub[0], "ic95": pub[1]},
                   "ok": b["dif"] == pub[0] and b["ic95"] == pub[1]}
        print("bootstrap v1.3 - v1.2", c, b["dif"], b["ic95"], "publicado", pub, flush=True)
    out = {"conferencias": conferencias, "todas_ok": all(x["ok"] for x in conferencias),
           "novas_professor_cpu": cpu, "bootstrap_v13_menos_v12": boot, "sanidade": _sanidade(),
           "codigo": impressao(), "relatorios": {n: f"saidas/bench/controles/validacao_{n}.json" for n in relatorios}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("acao", choices=["pontuar", "validar"])
    ap.add_argument("--nome", help="nome do candidato (saída em saidas/bench/controles/<nome>.json)")
    ap.add_argument("--spans", action="append", default=[], help="JSONL de spans (repetível)")
    ap.add_argument("--limiares", default="", help="limiares extras da guarda, ex.: 0.9,0.97")
    ap.add_argument("--sem-oficial", action="store_true", help="pula a métrica oficial no estresse")
    ap.add_argument("--saida", help="arquivo de saída (padrão saidas/bench/controles/<nome>.json)")
    a = ap.parse_args()
    SAIDA.mkdir(parents=True, exist_ok=True)
    if a.acao == "validar":
        res = validar()
        destino = pathlib.Path(a.saida) if a.saida else SAIDA / "validacao.json"
    else:
        if not a.nome or not a.spans:
            ap.error("pontuar exige --nome e --spans")
        extras = [float(x) for x in a.limiares.split(",") if x]
        res = avaliar(a.nome, [pathlib.Path(s) for s in a.spans], extras, not a.sem_oficial)
        destino = pathlib.Path(a.saida) if a.saida else SAIDA / f"{a.nome}.json"
    destino.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", destino)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
