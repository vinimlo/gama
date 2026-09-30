# -*- coding: utf-8 -*-
"""Indice canonico construido UMA VEZ, offline, a partir do SQLite.

Por que indice e nao FTS por citacao:

O FTS devolve qualquer documento que MENCIONE o numero, e acordaos citam uns
aos outros o tempo todo -- buscar "1.276.977" devolve seis documentos do STF
e nenhum deles e o RE 1.276.977. O sinal que separa "e o processo" de
"apenas cita" e a POSICAO: no registro que e o feito, o numero esta no
cabecalho. Indexar so o numero proprio de cada ficha (cabecalho.py, por
tribunal) resolve a ambiguidade de uma vez e ainda
troca uma varredura FTS por citacao por um lookup O(1) -- o que mantem a
solucao folgada dentro do teto de 60 s/documento.
"""
from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator

from .cabecalho import CadeiaDeClasse, LeitorDeNumeroProprio, NumeroProprio
from .leis import APELIDOS_LEI, IdentificadorDeLei
from .normalizar import NumeroDeProcesso

__all__ = ["APELIDOS_LEI", "Acervo", "Ficha", "Indice", "IndiceDeDispositivos", "IndiceDeProcessos",
           "IndiceDeSumulas", "construir"]

# O cabecalho termina onde o texto do artigo comeca ("... Art. 276."). Parar
# na virgula falhava na Constituicao, cujo cabecalho nao tem virgula:
#   "Artigo 276 da Lei n 4.737, de 15 de julho de 1965 Art. 276. ..."
#   "Artigo 93 da Constituicao Federal de 1988 Art. 93. ..."
_CAB_DISPOSITIVO = re.compile(
    r"Artigo\s+(\d+)\s*[ºo°]?\s+d[oa]s?\s+(.{5,80}?)\s+Art\.?\s",
    re.I | re.S,
)
_CAB_SUMULA = re.compile(
    r"S[uú]mula\s+(Vinculante\s+)?n\.?\s*(\d+)\s+do\s+(\w+)", re.I)


def _sem_zeros(numero: str) -> str:
    return numero.lstrip("0") or numero


@dataclass(frozen=True)
class Ficha:
    """Um documento do acervo."""
    id: str
    natureza: str          # acordao | sumula | dispositivo
    tribunal: str | None
    ano: int | None
    relator: str | None
    texto: str


class Acervo:
    """O SQLite congelado da organização, só leitura."""

    def __init__(self, caminho_db):
        self.caminho_db = caminho_db

    def fichas(self) -> Iterator[Ficha]:
        con = sqlite3.connect(f"file:{self.caminho_db}?mode=ro", uri=True)
        try:
            linhas = con.execute(
                "SELECT id, natureza, tribunal, ano, relator, texto FROM documentos"
            ).fetchall()
        finally:
            con.close()
        return (Ficha(*linha) for linha in linhas)


class IndiceDeProcessos:
    """Chave do número próprio -> ids das fichas; e a cadeia de classe de cada ficha, que
    separa as que dividem o número (embargos, agravo e principal do mesmo processo)."""

    def __init__(self):
        self.por_chave: dict = defaultdict(list)
        self.cadeias: dict = {}

    def adicionar(self, doc_id: str, proprio: NumeroProprio) -> None:
        self.cadeias[doc_id] = proprio.cadeia
        for chave in proprio.chaves:
            if doc_id not in self.por_chave[chave]:
                self.por_chave[chave].append(doc_id)

    def candidatos(self, bruto: str) -> list:
        return self.por_chave.get(NumeroDeProcesso.do_bruto(bruto).chave, [])

    def cadeia(self, doc_id: str) -> CadeiaDeClasse:
        return CadeiaDeClasse(self.cadeias.get(doc_id, ()))


class IndiceDeSumulas(dict):
    """(número, tribunal, vinculante) -> id."""

    def adicionar(self, texto: str, doc_id: str) -> None:
        m = _CAB_SUMULA.search(texto[:120])
        if m:
            self[(_sem_zeros(m.group(2)), m.group(3).upper(), bool(m.group(1)))] = doc_id

    def resolver(self, numero: str, tribunal: str | None, vinculante: bool) -> list:
        chave = (_sem_zeros(numero), tribunal, vinculante)
        if chave in self:
            return [self[chave]]
        # Tribunal declarado e diferente: "Sumula 211 do TSE" nao e a 211 do STJ.
        # Resolver pelo numero aqui transformaria inventada em real (tau).
        if tribunal is not None:
            return []
        # Sem tribunal declarado: todas as de mesmo número e vínculo. Mais de uma é
        # empate, e o classificar decide (D-002).
        return [i for (n, _t, v), i in self.items() if n == chave[0] and v == vinculante]


class IndiceDeDispositivos(dict):
    """(artigo, chave da lei) -> id."""

    def adicionar(self, texto: str, doc_id: str, leis: IdentificadorDeLei) -> None:
        m = _CAB_DISPOSITIVO.search(texto[:200])
        if m:
            self[(_sem_zeros(m.group(1)), leis.do_cabecalho(m.group(2)))] = doc_id

    def resolver(self, artigo: str, chave_lei: str) -> list:
        chave = (_sem_zeros(artigo), chave_lei)
        return [self[chave]] if chave in self else []


@dataclass
class Indice:
    """Os índices do acervo, construídos UMA VEZ, offline."""
    processos: IndiceDeProcessos = field(default_factory=IndiceDeProcessos)
    sumulas: IndiceDeSumulas = field(default_factory=IndiceDeSumulas)
    dispositivos: IndiceDeDispositivos = field(default_factory=IndiceDeDispositivos)
    meta: dict = field(default_factory=dict)          # id -> (tribunal, ano, relator)

    def __post_init__(self):
        if not isinstance(self.sumulas, IndiceDeSumulas):
            self.sumulas = IndiceDeSumulas(self.sumulas)
        if not isinstance(self.dispositivos, IndiceDeDispositivos):
            self.dispositivos = IndiceDeDispositivos(self.dispositivos)

    @classmethod
    def construir(cls, acervo: Acervo, leitor: LeitorDeNumeroProprio | None = None,
                  leis: IdentificadorDeLei | None = None) -> Indice:
        """Varre o acervo uma vez e monta todos os índices."""
        leitor = leitor or LeitorDeNumeroProprio()
        leis = leis or IdentificadorDeLei()
        idx = cls()
        for f in acervo.fichas():
            idx.meta[f.id] = (f.tribunal, f.ano, f.relator)
            if f.natureza == "acordao":
                # So o NUMERO PROPRIO de cada ficha vira chave (cabecalho.py). O indice
                # antigo indexava todo numero dos primeiros 300 chars e datas, numero
                # de registro e a Lei 13.015/2014 viravam chave: 280 chaves ambiguas,
                # 176 fichas sem chave unica. Agora: 996/996 com numero proprio.
                idx.processos.adicionar(f.id, leitor.ler(f.texto, f.tribunal))
            elif f.natureza == "sumula":
                idx.sumulas.adicionar(f.texto, f.id)
            elif f.natureza == "dispositivo":
                idx.dispositivos.adicionar(f.texto, f.id, leis)
        return idx

    @classmethod
    def do_banco(cls, caminho_db) -> Indice:
        return cls.construir(Acervo(caminho_db))

    def tribunal(self, doc_id: str) -> str | None:
        return self.meta.get(doc_id, (None,))[0]

    def candidatos_processo(self, bruto: str) -> list:
        return self.processos.candidatos(bruto)

    # -------------------------------------------------------------- fachadas (até a leva 8)

    @property
    def por_processo(self) -> dict:
        return self.processos.por_chave

    @property
    def cadeia(self) -> dict:
        return self.processos.cadeias

    def resolve_sumula(self, numero: str, tribunal: str | None, vinculante: bool) -> list:
        return self.sumulas.resolver(numero, tribunal, vinculante)

    def resolve_dispositivo(self, artigo: str, chave_lei: str) -> list:
        return self.dispositivos.resolver(artigo, chave_lei)


def construir(caminho_db) -> Indice:
    """Fachada (até a leva 8): `Indice.do_banco(caminho_db)`."""
    return Indice.do_banco(caminho_db)


def _chave_lei(descricao: str) -> str:
    """Fachada (até a leva 8): `IdentificadorDeLei.do_cabecalho`."""
    return IdentificadorDeLei.do_cabecalho(descricao)
