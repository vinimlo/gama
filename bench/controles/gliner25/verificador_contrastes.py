# -*- coding: utf-8 -*-
"""Contrastes do verificador (experimento 3b): GLiNER 2.5 contra a v2.1, política a política.

Lê os spans finais que o CLI do verificador gravou em `saidas/bench/controles/gliner25/verificador/
spans/<nome>__<política>.jsonl` (via `verificador25.py`) e compara, pareado por ementa (bootstrap de
`avaliar.bootstrap`: 2.000 reamostras, semente 0), a 2.5 contra a v2.1 na mesma política e com o tau
que cada uma escolheu nas 305. Junta também, dos relatórios, tau, F1, IC95 contra o Gama v1.3 com
guarda, AUROC por conjunto, documentos mudados no estresse e a nota oficial. Por fim, a diferença de
AUROC 2.5 menos v2.1 sobre os mesmos candidatos das 305 e das 172, com IC95 por bootstrap de ementas.

    docker compose run --rm -T -e PYTHONDONTWRITEBYTECODE=1 gama \\
        python -m bench.controles.gliner25.verificador_contrastes
Saída: `saidas/bench/controles/gliner25/verificador/contrastes.json`.
"""
from __future__ import annotations

import json

from bench import pontuar
from bench.controles import avaliar as av

PASTA = av.SAIDA / "gliner25" / "verificador"
NOMES = ["gliner21_zs", "gliner25_zs", "gliner25_desc050", "gliner21_ft", "gliner25_ft",
         "gliner21_ft_piso", "gliner25_ft_piso", "aceita_tudo"]
PARES = [("gliner25_zs", "gliner21_zs"), ("gliner25_desc050", "gliner21_zs"), ("gliner25_ft", "gliner21_ft"),
         ("gliner25_ft_piso", "gliner21_ft_piso")]
POLITICAS = ("A", "B", "C")


def _por_doc(nome: str, pol: str, c: str) -> dict:
    _, linhas = pontuar._ler_spans(PASTA / "spans" / f"{nome}__{pol}.jsonl")
    return av.por_doc(c, av.spans(linhas, c))


def auroc_pareado(a: str, b: str, c: str) -> dict:
    """AUROC(a) - AUROC(b) sobre os mesmos candidatos (rótulo: casa com o ouro, VAGA em JURIS), com IC95
    por bootstrap de documentos (2.000 reamostras, semente 0, os mesmos postos de `avaliar.bootstrap`)."""
    import random
    from bench.controles.verificador import nucleo as nu
    _, docs = nu.ler_candidatos()
    _, gold = av.textos_e_ouro(c)
    sa, sb = (nu.ler_scores(PASTA / "scores" / f"{n}.jsonl")[1] for n in (a, b))
    por_doc = {d: [(sa[x.cid], sb[x.cid], int(nu.casa(x.span, gold[d], True))) for x in G + R]
               for d, (G, R) in docs[c].items()}
    ds = sorted(por_doc)

    def dif(amostra):
        linhas = [t for d in amostra for t in por_doc[d]]
        return nu.auroc([(x, y) for x, _, y in linhas]) - nu.auroc([(z, y) for _, z, y in linhas])
    rng = random.Random(av.SEMENTE)
    difs = sorted(dif([rng.choice(ds) for _ in ds]) for _ in range(av.REAMOSTRAS))
    todos = [t for d in ds for t in por_doc[d]]
    return {"auroc_a": nu.auroc([(x, y) for x, _, y in todos]), "auroc_b": nu.auroc([(z, y) for _, z, y in todos]),
            "dif": round(dif(ds), 4), "ic95": [round(difs[int(0.025 * av.REAMOSTRAS)], 4),
                                               round(difs[int(0.975 * av.REAMOSTRAS) - 1], 4)],
            "candidatos": len(todos), "positivos": sum(y for _, _, y in todos)}


def main() -> int:
    tabela = {}
    for n in NOMES:
        r = json.loads((PASTA / f"{n}.json").read_text(encoding="utf-8"))
        tabela[n] = {"auroc": r["auroc"], "scores_sha256": r["scores"]["sha256"], "politicas": {}}
        for pol, res in r["politicas"].items():
            f1_tau = res["f1_305_por_tau"]
            melhor = max(f1_tau.values())
            tabela[n]["politicas"][pol] = {
                **{k: v for k, v in res["resumo"].items() if k not in ("verificador", "politica")},
                "taus_empatados_no_maximo_305": [t for t, f in f1_tau.items() if f == melhor],
                "estresse_docs_mudados_por_tau": res["conjuntos"]["estresse"]["docs_mudados_por_tau"]}
    contrastes = {}
    for a, b in PARES:
        for pol in POLITICAS:
            k = f"{a} - {b} [{pol}]"
            contrastes[k] = {c: av.bootstrap(_por_doc(a, pol, c), _por_doc(b, pol, c)) for c in ("reais", "novas")}
            print(k, " | ".join(f"{c} {v['dif']:+.4f} {v['ic95']}" for c, v in contrastes[k].items()), flush=True)
    aurocs = {}
    for a, b in PARES:
        for c in ("reais", "novas"):
            k = f"{a} - {b} [{c}]"
            aurocs[k] = auroc_pareado(a, b, c)
            print("AUROC", k, aurocs[k], flush=True)
    out = {"nota": "pareado por ementa; cada lado com o tau que escolheu nas 305", "tabela": tabela,
           "contrastes_25_vs_21": contrastes, "auroc_25_vs_21": aurocs, "codigo": av.impressao()}
    destino = PASTA / "contrastes.json"
    destino.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", destino)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
