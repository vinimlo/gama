# -*- coding: utf-8 -*-
"""Confere, no container, as cópias de `ft_treinar.py` (o job do HF só recebe o script).

Sem gliner2 nem torch. Aqui:
    1. constantes iguais às do zero-shot (`bench/extrair_gliner.py`) e do fine-tune da v2.1
       (`bench/controles/gliner_ft/treinar_gliner.py`);
    2. funções copiadas com o MESMO código-fonte das originais (`inspect.getsource`);
    3. `janelas` igual nos 1.105 textos de entrada;
    4. `Extrator.direto` (formato de saída da gliner2, janelas em lote) igual a
       `bench.extrair_gliner.extrair` quando o modelo falso devolve as mesmas entidades: confere a
       soma do início da janela, o mapeamento de rótulo e a fusão;
    5. conversão do final_v3: toda citação inteira em alguma janela, borda alinhada ao divisor da
       gliner2, nenhum ouro deixado sem marca pela busca por texto da biblioteca.
O divisor de palavras copiado da gliner2 e a reconstrução "piso -> limiar" (que usa
`resolve_overlaps` da biblioteca) são conferidos dentro do job, contra o modelo.

    docker compose run --rm gama python -m bench.controles.gliner25.ft_conferir
"""
from __future__ import annotations

import inspect
import json
import pathlib
import random
import re

from bench import extrair_gliner as zs
from bench.controles.gliner25 import ft_treinar as ft
from bench.controles.gliner_ft import treinar_gliner as v21

BENCH = pathlib.Path("/app/saidas/bench")
PALAVRA = re.compile(r"\w+(?:[-_]\w+)*|\S")


def _falsas(texto: str, rotulos, limiar: float) -> list[dict]:
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
    """A API da gliner2: [{"entities": {rótulo: [{text, confidence, start, end}]}}] por texto."""

    def batch_extract_entities(self, textos, rotulos, batch_size=8, threshold=0.5, **kw):
        out = []
        for t in textos:
            ents = {r: [] for r in rotulos}
            for e in _falsas(t, list(rotulos), threshold):
                ents[e["label"]].append({"text": t[e["start"]:e["end"]].strip(), "confidence": e["score"],
                                         "start": e["start"], "end": e["end"]})
            out.append({"entities": ents})
        return out


def main() -> int:
    res = {}
    res["constantes"] = {
        "zero_shot": ft.ROTULOS == zs.ROTULOS and ft.PALAVRAS == zs.PALAVRAS and ft.PASSO == zs.PASSO,
        "v21_ft": all(getattr(ft, k) == getattr(v21, k) for k in (
            "LIMIAR_ZERO_SHOT", "DADOS", "REV_TREINO", "REV_BENCH", "ROT_OURO", "NOME", "PISO", "GRADE",
            "SEMENTE", "ROTULOS", "PALAVRAS", "PASSO")),
    }
    fontes = {"janelas": (ft.janelas, zs.janelas)}
    for nome in ("fundir", "carregar", "impressao_textos", "nome_rotulo", "janelas_treino", "extracao",
                 "f1_exato", "escolher", "ler_entradas"):
        fontes[nome] = (getattr(ft, nome), getattr(v21, nome))
    res["fontes_iguais"] = {n: inspect.getsource(a) == inspect.getsource(b) for n, (a, b) in fontes.items()}
    textos = []
    for arq in ("entrada_remota.jsonl", "entrada_novas.jsonl"):
        textos += [json.loads(x)["texto"] for x in (BENCH / arq).read_text(encoding="utf-8").splitlines()]
    res["janelas"] = {"textos": len(textos), "diferentes": sum(ft.janelas(t) != zs.janelas(t) for t in textos)}
    ext = ft.Extrator.__new__(ft.Extrator)             # sem gliner2: só o laço, com o modelo falso
    ext.modelo = Falso25()
    res["direto"] = {"textos": len(textos),
                     "diferentes": sum(ext.direto(t, zs.LIMIAR) != zs.extrair(Falso21(), t) for t in textos)}
    from gama.formas import DetectorDeForma
    docs, split = ft.carregar(pathlib.Path("/app/corpus/goldenset/v3"))
    res["impressao_textos_final_v3"] = ft.impressao_textos(docs)
    for nome in ("treino", "estresse"):
        sub = {d: v for d, v in docs.items() if split.get(d, "treino") == nome}
        _, st = ft.exemplos(sub, DetectorDeForma().forma)
        aud = st["auditoria_ouro_por_texto"]
        res[f"conversao_{nome}"] = {k: st.get(k, 0) for k in (
            "citacoes", "citacoes_fora_de_toda_janela", "cortadas_pela_janela", "borda_desalinhada",
            "sem_palavra", "sobrepostas", "janelas", "janelas_sem_citacao")}
        res[f"conversao_{nome}"]["ouro_nao_marcado"] = aud.get("ouro_nao_marcado", 0)
        res[f"conversao_{nome}"]["extra_total"] = aud.get("extra_total", 0)
        res[f"conversao_{nome}"]["posicoes_ouro"] = aud.get("posicoes_ouro", 0)
    ok = (all(res["constantes"].values()) and all(res["fontes_iguais"].values())
          and res["janelas"]["diferentes"] == 0 and res["direto"]["diferentes"] == 0
          and res["impressao_textos_final_v3"] == "d783eadaa04b7334"
          and all(res[f"conversao_{n}"][k] == 0 for n in ("treino", "estresse")
                  for k in ("citacoes_fora_de_toda_janela", "borda_desalinhada", "sem_palavra", "sobrepostas",
                            "ouro_nao_marcado")))
    res["ok"] = ok
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
