# -*- coding: utf-8 -*-
"""Comparações pareadas do GLiNER fine-tunado além das que `bench.controles.avaliar` já grava.

O relatório do harness traz, por variante, a diferença para o Gama v1.3 COM guarda. Aqui entram,
com o mesmo bootstrap (pareado por ementa, 2.000 reamostras, semente 0):
    - contra o Gama v1.3 CRU (mesmos pesos, spans de `saidas/bench/gama.jsonl` nas 305 e de
      `modelo_base12.jsonl` nas 172);
    - contra o GLiNER zero-shot do benchmark (`saidas/bench/gliner.jsonl`), cru e com a guarda
      no mesmo limiar; só nas 305 (o arquivo publicado não cobre as 172).
Nada é escolhido aqui: as variantes e o limiar vêm do relatório do harness.

    docker compose run --rm gama python -m bench.controles.gliner_ft.comparar \\
        --relatorio saidas/bench/controles/gliner_ft.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

from bench import pontuar
from bench.controles import avaliar as av


def limiar_de(variante: str) -> float | None:
    return None if variante == "cru" else float(variante.split("@")[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relatorio", required=True)
    ap.add_argument("--saida")
    a = ap.parse_args()
    rel = json.loads(pathlib.Path(a.relatorio).read_text(encoding="utf-8"))
    _, linhas = av.ler(rel["arquivos"])
    _, zs = pontuar._ler_spans(av.BENCH / "gliner.jsonl")
    out = {"relatorio": a.relatorio, "codigo": av.impressao(), "contrastes": {}, "referencias": {}}
    for c in ("reais", "novas"):
        sp = av.spans(linhas, c)
        v13 = av.spans(pontuar._ler_spans(av.REFERENCIA[c])[1], c)
        ref = {"gama_v13_cru": av.por_doc(c, v13), "gama_v13_guarda": av.referencia(c)}
        out["referencias"][c] = {"gama_v13_cru": av.extracao(c, v13)["f1"],
                                 "gama_v13_guarda": av.extracao(c, av.com_guarda(v13, av.regua(c), 0.95))["f1"]}
        zs_sp = av.spans(zs, c)                       # None nas novas (sem spans publicados)
        for v in rel["variantes"]:
            t = limiar_de(v)
            fin = sp if t is None else av.com_guarda(sp, av.regua(c), t)
            cont = av.por_doc(c, fin)
            x = {"f1": av.extracao(c, fin)["f1"],
                 "vs_gama_v13_cru": av.bootstrap(cont, ref["gama_v13_cru"]),
                 "vs_gama_v13_guarda": av.bootstrap(cont, ref["gama_v13_guarda"])}
            if zs_sp is not None:
                zfin = zs_sp if t is None else av.com_guarda(zs_sp, av.regua(c), t)
                x["gliner_zero_shot_mesma_variante_f1"] = av.extracao(c, zfin)["f1"]
                x["vs_gliner_zero_shot_mesma_variante"] = av.bootstrap(cont, av.por_doc(c, zfin))
            out["contrastes"].setdefault(v, {})[c] = x
            print(c, v, json.dumps({k: (y["dif"], y["ic95"]) if isinstance(y, dict) else y
                                    for k, y in x.items()}, ensure_ascii=False), flush=True)
    destino = pathlib.Path(a.saida) if a.saida else pathlib.Path(a.relatorio).with_name(
        pathlib.Path(a.relatorio).stem + "_comparacao.json")
    destino.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", destino)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
