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
from dataclasses import dataclass

from .classificar import Citacao, Classificador
from .extratores import CatalogoDeExtratores, Extrator, ExtratorRegua
from .indice import Indice
from .resolver import Resolvedor
from .span import Intervalo, Span, aparar  # noqa: F401  (aparar: fachada, scripts do treino importam daqui)

SCHEMA = "1.2"
DB_PADRAO = "/app/dados/desafio1_bracis.db"


@dataclass(frozen=True)
class Documento:
    nome: str          # documento_id: o nome do .txt sem extensão
    texto: str

    @classmethod
    def ler(cls, arq) -> Documento:
        """Texto exatamente como veio: newline="" preserva \r\n. read_text() trocaria por
        \n e deslocaria todos os offsets depois da primeira quebra (revisão independente, rodada 1, achado 2)."""
        with open(arq, encoding="utf-8", newline="") as fh:
            return cls(pathlib.Path(arq).stem, fh.read())


class Pipeline:
    """Documento -> lista de citações já classificadas.

    O extrator decide O QUE é citação; o índice decide se ela existe no acervo.
    Sem extrator explícito, usa a régua (baseline).
    """
    IOU_MAXIMO = 0.5

    def __init__(self, indice: Indice, extrator: Extrator | None = None,
                 classificador: Classificador | None = None):
        self.extrator = extrator or ExtratorRegua()
        self.resolvedor = Resolvedor(indice)
        self.classificador = classificador or Classificador()

    def processar(self, texto: str) -> list[Citacao]:
        spans = Span.aparar_todos(self.extrator.extrair(texto), texto)
        return self.sem_sobreposicao([self.classificador.classificar(s, self.resolvedor.resolver(s)) for s in spans])

    @classmethod
    def sem_sobreposicao(cls, citacoes: list) -> list:
        """Última guarda antes do JSON: duas predições com IoU >= 0,5 entre si invalidam a
        submissão inteira (regra da métrica). Só esse critério — interseção pequena é
        permitida e pode ser dois acertos (revisão independente, rodada 2, achado 4).
        Aceita qualquer objeto com `inicio` e `fim`."""
        mantidas = []
        for c in sorted(citacoes, key=lambda c: (c.inicio, -(c.fim - c.inicio))):
            if not any(Intervalo.iou(c, m) >= cls.IOU_MAXIMO for m in mantidas):
                mantidas.append(c)
        return mantidas


class SaidaJSON:
    """Um .json por documento, no schema da organização."""
    SCHEMA = SCHEMA

    def __init__(self, pasta):
        self.pasta = pathlib.Path(pasta)
        self.pasta.mkdir(parents=True, exist_ok=True)

    def documento(self, doc: Documento, citacoes: list) -> dict:
        return {
            "schema_version": self.SCHEMA,
            "documento_id": doc.nome,
            "citacoes": [c.para_json(i + 1) for i, c in enumerate(citacoes)],
        }

    def escrever(self, doc: Documento, citacoes: list) -> pathlib.Path:
        # Encoding e regra rigida: UTF-8 sem BOM, LF, NFC. Nao alterar.
        arq = self.pasta / f"{doc.nome}.json"
        arq.write_text(json.dumps(self.documento(doc, citacoes), ensure_ascii=False, indent=2), encoding="utf-8")
        return arq


class Aplicacao:
    """O entrypoint: lê os argumentos, monta índice e extrator uma vez e processa a pasta."""

    @staticmethod
    def argumentos() -> argparse.ArgumentParser:
        ap = argparse.ArgumentParser(description="Gama: verificador de citacoes juridicas (BRACIS 2026 x Jusbrasil)")
        ap.add_argument("--input", required=True, help="pasta com os .txt")
        ap.add_argument("--output", required=True, help="pasta de saida dos .json")
        ap.add_argument("--db", default=os.environ.get("GAMA_DB", DB_PADRAO),
                        help="base canonica SQLite")
        ap.add_argument("--extrator", default=os.environ.get("GAMA_EXTRATOR", "regua"),
                        choices=CatalogoDeExtratores.NOMES, help="quem decide o que e citacao")
        ap.add_argument("--modelos", default=os.environ.get("GAMA_MODELOS", "/models"),
                        help="pasta com os pesos do extrator neural (montada por volume)")
        return ap

    @staticmethod
    def extrator(nome: str, modelos: str) -> Extrator:
        if nome != "regua" and not (pathlib.Path(modelos) / "config.json").exists():
            # Sem pesos montados: saída válida pela régua é melhor que submissão vazia.
            print(f"AVISO: sem pesos em {modelos} (monte com -v <pesos>:/models:ro); "
                  "usando o extrator de regras", file=sys.stderr)
            nome = "regua"
        print(f"extrator: {nome}", file=sys.stderr)
        return CatalogoDeExtratores(modelos).carregar(nome)

    def executar(self, argv=None) -> int:
        args = self.argumentos().parse_args(argv)
        entrada = pathlib.Path(args.input)
        saida = SaidaJSON(args.output)

        arquivos = sorted(entrada.glob("*.txt"))
        if not arquivos:
            print(f"nenhum .txt em {entrada}", file=sys.stderr)
            return 1

        t0 = time.perf_counter()
        pipeline = Pipeline(Indice.do_banco(args.db), self.extrator(args.extrator, args.modelos))
        t_indice = time.perf_counter() - t0

        total = 0
        for arq in arquivos:
            doc = Documento.ler(arq)
            citacoes = pipeline.processar(doc.texto)
            total += len(citacoes)
            saida.escrever(doc, citacoes)

        gasto = time.perf_counter() - t0
        print(f"{len(arquivos)} documentos, {total} citacoes -> {saida.pasta}")
        print(f"indice {t_indice:.2f}s | total {gasto:.2f}s | "
              f"{gasto / len(arquivos) * 1000:.0f} ms/documento")
        return 0


# ------------------------------------------------------------------ fachadas (até a leva 8)

def ler_texto(arq) -> str:
    return Documento.ler(arq).texto


def _iou(a, b) -> float:
    return Intervalo.iou(a, b)


def sem_sobreposicao(citacoes: list) -> list:
    return Pipeline.sem_sobreposicao(citacoes)


def processar(texto: str, idx, extrator=None) -> list:
    return Pipeline(idx, extrator).processar(texto)


def main(argv=None) -> int:
    return Aplicacao().executar(argv)


if __name__ == "__main__":
    raise SystemExit(main())
