# -*- coding: utf-8 -*-
"""Anotação prata de ementas reais por dois LLMs abertos, com alinhamento exato.

Só ementas onde o Gama e a régua DIVERGEM (mais uma
amostra onde concordam, para medir a concordância de base). Cada LLM reescreve a
ementa com marcadores ⟦J⟧…⟦/J⟧ (jurisprudência identificada), ⟦L⟧…⟦/L⟧ (dispositivo de
lei), ⟦V⟧…⟦/V⟧ (referência vaga: tribunal + ano + relator, sem número). O texto sem
marcadores tem que ser IDÊNTICO ao original — senão a anotação é descartada (LLM não
sabe offset; ele devolve texto, e o alinhamento é verificado, não suposto).

Rótulo prata = os dois LLMs concordam (mesmo tipo, IoU >= 0,8). Divergência entre eles
vai para adjudicação por critério escrito; só o ambíguo vai para decisão humana.

    docker compose run --rm lab python -m reais.anotar
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import pathlib
import random
import re

from geracao.expandir import _json
from geracao.llm import chat

MARCA = re.compile(r"⟦(/?)([JLV])⟧")
TIPO = {"J": "JURIS", "L": "LEI", "V": "VAGA"}

PROMPT = """Marque as CITAÇÕES JURÍDICAS da ementa abaixo, reescrevendo-a INTEIRA e SEM ALTERAR
NENHUM caractere (nem espaço, nem quebra de linha, nem pontuação) — só inserindo marcadores:

- ⟦J⟧…⟦/J⟧ precedente identificado por número: recurso/processo com número ("REsp 1.234.567/SP",
  "AgInt no AREsp nº 1.576.933/SP", "RE 958.252"), súmula ("Súmula 7/STJ", "Súmula Vinculante 10"),
  tema ("Tema 725 da repercussão geral");
- ⟦L⟧…⟦/L⟧ dispositivo de lei: artigo com o diploma ("art. 5º, LV, da Constituição Federal",
  "art. 1.022 do CPC", "art. 312 do Código de Processo Penal");
- ⟦V⟧…⟦/V⟧ referência vaga a precedente SEM número, com tribunal e ano e relator ("julgado do STJ
  proferido em 2021 pela relatoria de Fulano").

Regras de borda:
- artigo ou preposição ANTES da citação fica FORA ("o ⟦J⟧REsp 1.234/SP⟦/J⟧", "no ⟦J⟧HC 123⟦/J⟧");
- pontuação logo DEPOIS fica FORA;
- no precedente, o span vai da classe até o número e a UF; "Rel. Min. ...", turma, "julgado em",
  "DJe ..." que vêm DEPOIS do número ficam FORA;
- lei citada sem artigo ("Lei 8.078/1990" sozinha) NÃO é marcada; referência genérica sem
  identificação ("a jurisprudência desta Corte", "o entendimento sumulado") NÃO é marcada.

Responda em JSON: {{"texto": "<a ementa inteira com os marcadores>"}}

EMENTA:
{texto}"""


def alinhar(original: str, marcado: str) -> list | None:
    """Remove os marcadores; se o texto bater com o original, devolve os spans."""
    spans, abertos, limpo = [], {}, []
    pos = 0
    ultimo = 0
    for m in MARCA.finditer(marcado):
        limpo.append(marcado[ultimo:m.start()])
        pos += m.start() - ultimo
        ultimo = m.end()
        fecha, t = m.group(1), m.group(2)
        if not fecha:
            abertos[t] = pos
        elif t in abertos:
            spans.append((abertos.pop(t), pos, TIPO[t]))
    limpo.append(marcado[ultimo:])
    return sorted(spans) if "".join(limpo) == original and not abertos else None


def _iou(a, b) -> float:
    i = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return i / (max(a[1], b[1]) - min(a[0], b[0])) if i else 0.0


def concordancia(x: list, y: list) -> tuple[list, list]:
    """-> (acordados, divergentes) entre dois conjuntos de spans (mesmo tipo, IoU >= 0,8)."""
    usados, ok = set(), []
    for s in x:
        k = next((k for k, t in enumerate(y) if k not in usados and t[2] == s[2] and _iou(s, t) >= 0.8), None)
        if k is not None:
            usados.add(k)
            ok.append(s if s[:2] == y[k][:2] else (min(s[0], y[k][0]), max(s[1], y[k][1]), s[2]))
    div = [s for s in x if not any(_iou(s, o) >= 0.8 and s[2] == o[2] for o in ok)] + \
          [t for k, t in enumerate(y) if k not in usados]
    return ok, div


def _cruza(a, lista) -> bool:
    return any(a[0] < b[1] and b[0] < a[1] for b in lista)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amostra", default="/app/corpus/reais/amostra.jsonl")
    ap.add_argument("--predicoes", default="/app/corpus/reais/predicoes.jsonl")
    ap.add_argument("--saida", default="/app/corpus/reais/prata.jsonl")
    ap.add_argument("--concordantes", type=int, default=60, help="amostra de ementas sem divergência")
    ap.add_argument("--divergentes", type=int, default=300, help="amostra de ementas com divergência")
    ap.add_argument("--paralelo", type=int, default=8)
    ap.add_argument("--ids", help="JSON com a lista de ementas a anotar (ex.: o teste de reais.sortear_teste); "
                                  "sem ele, a amostra enriquecida por divergência")
    a = ap.parse_args()
    textos = {r["id"]: r for r in map(json.loads, open(a.amostra, encoding="utf-8"))}
    preds = {r["id"]: r for r in map(json.loads, open(a.predicoes, encoding="utf-8"))}
    div, conc = [], []
    for i, p in preds.items():
        g = [tuple(s[:2]) for s in p["gama"]]
        r = [tuple(s[:2]) for s in p["regua"]]
        (div if set(g) != set(r) else conc).append(i)
    if a.ids:
        alvo = json.loads(pathlib.Path(a.ids).read_text())
        div = set(div)
    else:
        rng = random.Random(3)
        alvo = sorted(rng.sample(sorted(div), min(a.divergentes, len(div)))) + \
            sorted(rng.sample(sorted(conc), min(a.concordantes, len(conc))))
    print(f"{len(div)} ementas com divergência Gama x régua; {len(conc)} sem; anotando {len(alvo)}", flush=True)

    def anota(i):
        texto = textos[i]["texto"]
        out = {}
        for mod in ("deepseek", "kimi"):
            msgs = [{"role": "user", "content": PROMPT.format(texto=texto)}]
            try:
                marcado = _json(chat(mod, msgs, temperatura=0.0, semente=0)).get("texto", "")
            except RuntimeError:
                marcado = None               # falha de rede persistente: descarta só este item
            out[mod] = alinhar(texto, marcado) if isinstance(marcado, str) else None
        return i, out

    res = {}
    with cf.ThreadPoolExecutor(a.paralelo) as ex:
        for n, (i, out) in enumerate(ex.map(anota, alvo), 1):
            res[i] = out
            if n % 50 == 0:
                print(f"{n}/{len(alvo)}", flush=True)
    stats = collections.Counter()
    with open(a.saida, "w", encoding="utf-8") as fh:
        for i in alvo:
            d, k = res[i]["deepseek"], res[i]["kimi"]
            if d is None or k is None:
                stats["descartada_alinhamento"] += 1
                continue
            ok, dv = concordancia(d, k)
            stats["ementas_anotadas"] += 1
            stats["spans_prata"] += len(ok)
            stats["spans_divergentes_llm"] += len(dv)
            fh.write(json.dumps({"id": i, "divergente_gama_regua": i in div, "prata": ok,
                                 "divergente_llm": dv, "gama": preds[i]["gama"], "regua": preds[i]["regua"],
                                 "proveniencia": textos[i]["proveniencia"]}, ensure_ascii=False) + "\n")
    print(json.dumps(dict(stats), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
