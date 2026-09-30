# -*- coding: utf-8 -*-
"""Entrypoint no contrato exigido pela organizacao:

    docker run <img> --input /data/in --output /data/out

Le um .txt por documento e escreve um .json por documento (schema 1.2).
Determinismo: modelo em eval(), sem amostragem, sem rede. O indice do acervo e
construido uma vez por execucao e reaproveitado; o extrator neural roda em janelas
de ate 2.048 tokens -- centenas de milissegundos por documento em GPU, muito
abaixo do teto de 60 s.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

from .classificar import classificar
from .extratores import carregar
from .indice import construir
from .resolver import resolver
from .span import aparar, aparar_todos  # noqa: F401  (aparar: scripts do treino importam daqui)

SCHEMA = "1.2"
DB_PADRAO = "/app/dados/desafio1_bracis.db"


def ler_texto(arq) -> str:
    """Texto exatamente como veio: newline="" preserva \r\n. read_text() trocaria por
    \n e deslocaria todos os offsets depois da primeira quebra (revisão independente, rodada 1, achado 2)."""
    with open(arq, encoding="utf-8", newline="") as fh:
        return fh.read()


def _iou(a, b) -> float:
    i = max(0, min(a.fim, b.fim) - max(a.inicio, b.inicio))
    return i / (max(a.fim, b.fim) - min(a.inicio, b.inicio)) if i else 0.0


def sem_sobreposicao(citacoes: list) -> list:
    """Última guarda antes do JSON: duas predições com IoU >= 0,5 entre si invalidam a
    submissão inteira (regra da métrica). Só esse critério — interseção pequena é
    permitida e pode ser dois acertos (revisão independente, rodada 2, achado 4)."""
    mantidas = []
    for c in sorted(citacoes, key=lambda c: (c.inicio, -(c.fim - c.inicio))):
        if not any(_iou(c, m) >= 0.5 for m in mantidas):
            mantidas.append(c)
    return mantidas


def processar(texto: str, idx, extrator=None) -> list:
    """Documento -> lista de citacoes ja classificadas.

    O extrator decide O QUE e citacao; o indice decide se ela existe no acervo.
    Sem extrator explicito, usa a regua (baseline).
    """
    extrator = extrator or carregar("regua")
    spans = aparar_todos(extrator.extrair(texto), texto)
    return sem_sobreposicao([classificar(span, resolver(span, idx)) for span in spans])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Gama: verificador de citacoes juridicas (BRACIS 2026 x Jusbrasil)")
    ap.add_argument("--input", required=True, help="pasta com os .txt")
    ap.add_argument("--output", required=True, help="pasta de saida dos .json")
    ap.add_argument("--db", default=os.environ.get("GAMA_DB", DB_PADRAO),
                    help="base canonica SQLite")
    ap.add_argument("--extrator", default=os.environ.get("GAMA_EXTRATOR", "regua"),
                    choices=["regua", "neural", "neural-cru", "uniao"], help="quem decide o que e citacao")
    ap.add_argument("--modelos", default=os.environ.get("GAMA_MODELOS", "/models"),
                    help="pasta com os pesos do extrator neural (montada por volume)")
    args = ap.parse_args(argv)

    entrada = pathlib.Path(args.input)
    saida = pathlib.Path(args.output)
    saida.mkdir(parents=True, exist_ok=True)

    arquivos = sorted(entrada.glob("*.txt"))
    if not arquivos:
        print(f"nenhum .txt em {entrada}", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    idx = construir(args.db)
    nome = args.extrator
    if nome != "regua" and not (pathlib.Path(args.modelos) / "config.json").exists():
        # Sem pesos montados: saída válida pela régua é melhor que submissão vazia.
        print(f"AVISO: sem pesos em {args.modelos} (monte com -v <pesos>:/models:ro); "
              "usando o extrator de regras", file=sys.stderr)
        nome = "regua"
    extrator = carregar(nome, args.modelos)
    print(f"extrator: {nome}", file=sys.stderr)
    t_indice = time.perf_counter() - t0

    total = 0
    for arq in arquivos:
        # Encoding e regra rigida: UTF-8 sem BOM, LF, NFC. Nao alterar.
        texto = ler_texto(arq)
        citacoes = processar(texto, idx, extrator)
        total += len(citacoes)
        doc = {
            "schema_version": SCHEMA,
            "documento_id": arq.stem,
            "citacoes": [c.para_json(i + 1) for i, c in enumerate(citacoes)],
        }
        (saida / f"{arq.stem}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    gasto = time.perf_counter() - t0
    print(f"{len(arquivos)} documentos, {total} citacoes -> {saida}")
    print(f"indice {t_indice:.2f}s | total {gasto:.2f}s | "
          f"{gasto / len(arquivos) * 1000:.0f} ms/documento")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
