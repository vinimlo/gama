# -*- coding: utf-8 -*-
"""Confere, no container, que as cópias em `extrair_gliner25.py` batem com `bench/extrair_gliner.py`.

O job do HF recebe só o script, então constantes, `janelas` e o laço de `extrair` foram copiados.
Sem GLiNER nem torch: os modelos são falsos e determinísticos. Aqui:
    1. constantes (rótulos, limiar, janelas, modelo e revisão da v2.1) iguais às do zero-shot;
    2. `janelas` igual nos 1.105 textos de entrada (estresse + 305 + 200 novas);
    3. `extrair_v21` igual a `bench.extrair_gliner.extrair` com um `predict_entities` falso;
    4. `Extrator25.extrair` (formato de saída da gliner2, lote e janela a janela) igual a
       `bench.extrair_gliner.extrair` quando o modelo falso devolve as mesmas entidades: confere a
       soma do início da janela, o mapeamento de rótulo e a fusão;
    5. as descrições cobrem exatamente os quatro rótulos, na mesma ordem.

    docker compose run --rm gama python -m bench.controles.gliner25.conferir
"""
from __future__ import annotations

import json
import pathlib
import random
import re

from bench import extrair_gliner as zs
from bench.controles.gliner25 import extrair_gliner25 as g

BENCH = pathlib.Path("/app/saidas/bench")
PALAVRA = re.compile(r"\w+(?:[-_]\w+)*|\S")


def _falsas(texto: str, rotulos, limiar: float) -> list[dict]:
    """Entidades pseudoaleatórias por (texto, rótulos), alinhadas a palavras."""
    rng = random.Random(hash((texto, tuple(rotulos))) % 2**32)
    ps = [(m.start(), m.end()) for m in PALAVRA.finditer(texto)]
    out = []
    for _ in range(rng.randint(0, 6)):
        if not ps:
            break
        i = rng.randrange(len(ps))
        j = min(len(ps) - 1, i + rng.randint(0, 5))
        sc = round(rng.random(), 6)
        if sc >= limiar:
            out.append({"start": ps[i][0], "end": ps[j][1], "label": rng.choice(list(rotulos)), "score": sc})
    return out


class Falso21:
    def predict_entities(self, texto, rotulos, threshold=0.5):
        return _falsas(texto, rotulos, threshold)


class Falso25:
    """A API da gliner2: {"entities": {rótulo: [{text, confidence, start, end}]}}."""

    def extract_entities(self, texto, rotulos, threshold=0.5, include_confidence=False, include_spans=False):
        nomes = list(rotulos)                        # lista ou dict nome -> descrição
        ents = {r: [] for r in nomes}
        for e in _falsas(texto, nomes, threshold):
            ents[e["label"]].append({"text": texto[e["start"]:e["end"]].strip(), "confidence": e["score"],
                                     "start": e["start"], "end": e["end"]})
        return {"entities": ents}

    def batch_extract_entities(self, textos, rotulos, batch_size=8, threshold=0.5, **kw):
        return [self.extract_entities(t, rotulos, threshold, **kw) for t in textos]


def main() -> int:
    res = {}
    res["constantes"] = all([g.MODELO_V21 == zs.MODELO, g.REVISAO_V21 == zs.REVISAO, g.ROTULOS == zs.ROTULOS,
                             g.LIMIAR == zs.LIMIAR, g.PALAVRAS == zs.PALAVRAS, g.PASSO == zs.PASSO])
    textos = []
    for arq in ("entrada_remota.jsonl", "entrada_novas.jsonl"):
        textos += [json.loads(x)["texto"] for x in (BENCH / arq).read_text(encoding="utf-8").splitlines()]
    res["janelas"] = {"textos": len(textos), "diferentes": sum(g.janelas(t) != zs.janelas(t) for t in textos)}
    f21 = Falso21()
    res["extrair_v21"] = {"textos": len(textos),
                          "diferentes": sum(g.extrair_v21(f21, t) != zs.extrair(f21, t) for t in textos)}
    f25 = Falso25()
    for lote in (True, False):
        ext = g.Extrator25(f25, False, zs.LIMIAR)
        dif = sum(ext.extrair(t, lote=lote) != zs.extrair(f21, t) for t in textos)
        res[f"extrator25_{'lote' if lote else 'janela'}"] = {"textos": len(textos), "diferentes": dif,
                                                             "texto_diferente": ext.texto_diferente}
    res["descricoes"] = list(g.DESCRICOES) == list(g.ROTULOS) and list(g.esquema(True)) == list(g.ROTULOS)
    ok = (res["constantes"] and res["janelas"]["diferentes"] == 0 and res["extrair_v21"]["diferentes"] == 0
          and all(res[k]["diferentes"] == 0 and res[k]["texto_diferente"] == 0
                  for k in ("extrator25_lote", "extrator25_janela")) and res["descricoes"])
    res["ok"] = ok
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
