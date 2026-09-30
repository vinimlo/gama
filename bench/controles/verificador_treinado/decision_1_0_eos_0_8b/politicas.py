# -*- coding: utf-8 -*-
"""O CLI de `bench/controles/verificador` (sem alteração) com as saídas nesta pasta:
saidas/bench/controles/verificador_treinado/decision_1_0_eos_0_8b/ em vez de .../verificador/.

Só troca `nucleo.SAIDA` antes de chamar o `main` de lá: relatório, spans finais das políticas, JSON da
métrica oficial do estresse e `refeito/` vão para cá; os candidatos continuam os de
saidas/bench/controles/verificador/candidatos.jsonl (o padrão do CLI, fixado na importação).

    docker compose run --rm gama python -m bench.controles.verificador_treinado.decision_1_0_eos_0_8b.politicas \\
        rodar --nome decision_1_0_eos_0_8b \\
        --scores saidas/bench/controles/verificador_treinado/decision_1_0_eos_0_8b/scores/decision_1_0_eos_0_8b.jsonl
    docker compose run --rm gama python -m bench.controles.verificador_treinado.decision_1_0_eos_0_8b.politicas refazer
"""
from __future__ import annotations

import sys

from bench.controles import avaliar
from bench.controles.verificador import __main__ as cli
from bench.controles.verificador import nucleo as nu

SAIDA = avaliar.SAIDA / "verificador_treinado" / "decision_1_0_eos_0_8b"


def main() -> int:
    nu.SAIDA = SAIDA
    sys.argv[0] = "python -m bench.controles.verificador_treinado.decision_1_0_eos_0_8b.politicas"
    return cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
