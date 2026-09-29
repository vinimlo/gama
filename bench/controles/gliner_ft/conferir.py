# -*- coding: utf-8 -*-
"""Confere, no container, que as cópias em `treinar_gliner.py` batem com os originais.

O job do HF recebe só o script, então `janelas`, a fusão de `extrair` (de
`bench/extrair_gliner.py`) e `extracao` (de `bench/pontuar.py`) foram copiadas. Aqui:
    1. constantes (modelo, revisão, rótulos, janelas) iguais às do zero-shot;
    2. `janelas` igual nos 1.105 textos de entrada (estresse + 305 + 200 novas);
    3. `Extrator.extrair` igual a `bench.extrair_gliner.extrair` com um modelo falso que
       devolve entidades pseudoaleatórias (com sobreposição entre janelas);
    4. `extracao` igual a `bench.pontuar.extracao` em predições pseudoaleatórias;
    5. conversão do final_v3: toda citação cabe inteira em alguma janela, com borda alinhada.

    docker compose run --rm gama python -m bench.controles.gliner_ft.conferir
"""
from __future__ import annotations

import json
import pathlib
import random

from bench import extrair_gliner as zs
from bench import pontuar
from bench.controles.gliner_ft import treinar_gliner as ft

BENCH = pathlib.Path("/app/saidas/bench")


class Falso:
    """predict_entities determinístico por (texto, rótulos): spans aleatórios alinhados a palavras."""

    def predict_entities(self, texto, rotulos, threshold=0.5):
        rng = random.Random(hash(texto) % 2**32)
        ps = ft.palavras(texto)
        out = []
        for _ in range(rng.randint(0, 6)):
            if not ps:
                break
            i = rng.randrange(len(ps))
            j = min(len(ps) - 1, i + rng.randint(0, 5))
            sc = round(rng.random(), 6)
            if sc >= threshold:
                out.append({"start": ps[i][1], "end": ps[j][2], "label": rng.choice(list(rotulos)), "score": sc})
        return out


def main() -> int:
    res = {}
    res["constantes"] = all([ft.MODELO == zs.MODELO, ft.REVISAO == zs.REVISAO, ft.ROTULOS == zs.ROTULOS,
                             ft.PALAVRAS == zs.PALAVRAS, ft.PASSO == zs.PASSO,
                             ft.LIMIAR_ZERO_SHOT == zs.LIMIAR])
    textos = []
    for arq in ("entrada_remota.jsonl", "entrada_novas.jsonl"):
        textos += [json.loads(x)["texto"] for x in (BENCH / arq).read_text(encoding="utf-8").splitlines()]
    res["janelas"] = {"textos": len(textos), "diferentes": sum(ft.janelas(t) != zs.janelas(t) for t in textos)}
    falso = Falso()
    ext = ft.Extrator(falso)
    res["extrair"] = {"textos": len(textos),
                      "diferentes": sum(ext.extrair(t, zs.LIMIAR) != zs.extrair(falso, t) for t in textos)}
    rng = random.Random(0)
    difs = 0
    for _ in range(300):
        gold, pred = {}, {}
        for d in range(rng.randint(1, 8)):
            gold[d] = [(a, a + rng.randint(1, 30), rng.choice("JLV")) for a in sorted(rng.sample(range(500), 5))]
            pred[d] = [(a, a + rng.randint(1, 30), rng.choice("JLV")) for a in sorted(rng.sample(range(500), 6))]
        for jv in (False, True):
            difs += ft.extracao(gold, pred, jv) != pontuar.extracao(gold, pred, jv)
    res["extracao"] = {"casos": 600, "diferentes": difs}
    from gama.formas import forma
    docs, split = ft.carregar(pathlib.Path("/app/corpus/goldenset/v3"))
    for nome in ("treino", "estresse"):
        sub = {d: v for d, v in docs.items() if split.get(d, "treino") == nome}
        _, st = ft.exemplos(sub, forma)
        res[f"conversao_{nome}"] = {k: st.get(k, 0) for k in ("citacoes", "citacoes_fora_de_toda_janela",
                                                               "cortadas_pela_janela", "borda_desalinhada",
                                                               "sem_palavra", "sobrepostas", "janelas")}
        res[f"conversao_{nome}"]["largura"] = st["largura_citacao_palavras"]
    res["impressao_textos_final_v3"] = ft.impressao_textos(docs)
    ok = (res["constantes"] and res["janelas"]["diferentes"] == 0 and res["extrair"]["diferentes"] == 0
          and res["extracao"]["diferentes"] == 0
          and all(res[f"conversao_{n}"][k] == 0 for n in ("treino", "estresse")
                  for k in ("citacoes_fora_de_toda_janela", "borda_desalinhada", "sem_palavra", "sobrepostas")))
    res["ok"] = ok
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
