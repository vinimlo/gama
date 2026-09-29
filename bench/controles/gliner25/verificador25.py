# -*- coding: utf-8 -*-
"""Verificador de candidatos com o GLiNER 2.5 (experimento 3b): o CLI de `bench.controles.verificador`
sem mudança, só com a pasta de saída trocada para `saidas/bench/controles/gliner25/verificador/`.

`nucleo.SAIDA` é lido a cada chamada (scores/, spans/, relatórios e os JSON da métrica oficial em
json/estresse/), então trocar o global antes de chamar o CLI desvia toda escrita para cá. Os
candidatos continuam os de `saidas/bench/controles/verificador/candidatos.jsonl` (só leitura).

    docker compose run --rm -T -e PYTHONDONTWRITEBYTECODE=1 gama \\
        python -m bench.controles.gliner25.verificador25 <acao do CLI> [...]

Mesmas ações do CLI original (de-spans, rodar, tabela, refazer...); ver o docstring de
`bench/controles/verificador/__main__.py`.
"""
from __future__ import annotations

from bench.controles import avaliar
from bench.controles.verificador import __main__ as cli
from bench.controles.verificador import nucleo as nu

nu.SAIDA = avaliar.SAIDA / "gliner25" / "verificador"

if __name__ == "__main__":
    raise SystemExit(cli.main())
