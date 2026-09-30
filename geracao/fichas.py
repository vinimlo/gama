# -*- coding: utf-8 -*-
"""Fichas do acervo em forma estruturada — a matéria-prima das citações reais.

O gerador sintético cita fichas REAIS do acervo (as 1.014, não só as ~95 que o
dev set usa). Para renderizar "AgInt no AREsp 1.576.933/SP" a partir de uma ficha
é preciso saber a cadeia de classe, o número como o cabeçalho o escreve e a UF;
para uma referência vaga, tribunal + ano + relator. O rótulo sai por construção:
a citação de uma ficha X é `real` com id_canonico X.
"""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass

from gama.cabecalho import LeitorDeNumeroProprio

UFS = ["AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG",
       "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"]

_ESTADOS = {
    "ACRE": "AC", "ALAGOAS": "AL", "AMAPA": "AP", "AMAZONAS": "AM", "BAHIA": "BA",
    "CEARA": "CE", "DISTRITO FEDERAL": "DF", "ESPIRITO SANTO": "ES", "GOIAS": "GO",
    "MARANHAO": "MA", "MATO GROSSO DO SUL": "MS", "MATO GROSSO": "MT", "MINAS GERAIS": "MG",
    "PARA": "PA", "PARAIBA": "PB", "PARANA": "PR", "PERNAMBUCO": "PE", "PIAUI": "PI",
    "RIO DE JANEIRO": "RJ", "RIO GRANDE DO NORTE": "RN", "RIO GRANDE DO SUL": "RS",
    "RONDONIA": "RO", "RORAIMA": "RR", "SANTA CATARINA": "SC", "SAO PAULO": "SP",
    "SERGIPE": "SE", "TOCANTINS": "TO",
}


def _sem_acento(s: str) -> str:
    d = unicodedata.normalize("NFD", s)
    return "".join(c for c in d if unicodedata.category(c) != "Mn")


@dataclass(frozen=True)
class Ficha:
    id: str
    tribunal: str
    cadeia: tuple          # ('AgInt', 'AREsp')
    numero: str            # como o cabeçalho escreve: '1.576.933' ou '0600530-94.2020.6.26.0171'
    uf: str | None
    ano: int | None
    relator: str           # sem "Min."/"Ministro"; caixa original preservada


@dataclass(frozen=True)
class Sumula:
    id: str
    tribunal: str
    numero: str
    vinculante: bool


@dataclass(frozen=True)
class Dispositivo:
    id: str
    artigo: str            # '373', '5º', '1º'
    lei: str               # chave do índice: '13105', 'CF', 'LC64', ...
    incisos: tuple         # ('I', 'II', ...) presentes no texto
    paragrafos: tuple      # ('1º-A', ...)
    alineas: tuple         # ('a', 'b', ...)


_PREFIXO_RELATOR = re.compile(
    r"^\s*(?:Min(?:istr[oa])?\.?|MIN(?:ISTR[OA])?\.?|Desembargador[a]?(?:\s+Federal)?)\s+", re.I)


def limpar_relator(r: str | None) -> str:
    r = (r or "").strip()
    # "LÁZARO GUIMARÃES DESEMBARGADOR CONVOCADO DO TRF 5 REGIÃO" -> só o nome
    r = re.split(r"\s+\(?(?:desembargador|ju[ií]z)[a]?\s+convocad", r, flags=re.I)[0]
    for _ in range(2):
        r = _PREFIXO_RELATOR.sub("", r)
    return r.strip()


def _uf_do_cabecalho(texto: str, tribunal: str, numero: str) -> str | None:
    cab = texto[:900]
    i = cab.find(numero)
    if i < 0:
        return None
    depois = cab[i + len(numero): i + len(numero) + 60]
    if tribunal == "STJ":
        m = re.match(r"\s*-\s*([A-Z]{2})\b", depois)
        return m.group(1) if m else None
    if tribunal == "STM":
        m = re.match(r"\s*/\s*([A-Z]{2})\b", depois)
        return m.group(1) if m else None
    if tribunal == "STF":
        plano = _sem_acento(depois).upper()
        for nome in sorted(_ESTADOS, key=len, reverse=True):
            if re.match(r"\s*" + nome + r"\b", plano):
                return _ESTADOS[nome]
    return None


_DISP = re.compile(r"Artigo\s+(\d+)\s*([ºo°])?\s+d[oa]s?\s+(.{5,80}?)\s+Art\.?\s", re.I | re.S)


def _chave_lei(desc: str) -> str:
    plano = _sem_acento(desc).lower()
    if "constitui" in plano:
        return "CF"
    m = re.search(r"lei\s+complementar\s+n\S*\s*([\d.]+)", plano)
    if m:
        return "LC" + m.group(1).replace(".", "")
    m = re.search(r"n\S*\s*([\d.]+)", plano)
    return m.group(1).replace(".", "") if m else plano


def carregar(db: str):
    """-> (fichas de acórdão, súmulas, dispositivos). Só leitura."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    fichas, sumulas, disps = [], [], []
    for id_, trib, ano, rel, nat, texto in con.execute(
            "SELECT id, tribunal, ano, relator, natureza, texto FROM documentos"):
        id_ = str(id_)
        if nat == "acordao":
            p = LeitorDeNumeroProprio().ler(texto, trib)
            if not p.chaves or not p.numero:
                continue
            numero = re.sub(r"\s+", "", p.numero)
            fichas.append(Ficha(id_, trib, p.cadeia, numero,
                                _uf_do_cabecalho(texto, trib, p.numero),
                                int(ano) if ano else None, limpar_relator(rel)))
        elif nat == "sumula":
            m = re.search(r"S[uú]mula\s+(Vinculante\s+)?n\.?\s*(\d+)\s+do\s+(\w+)", texto, re.I)
            if m:
                sumulas.append(Sumula(id_, m.group(3).upper(), m.group(2), bool(m.group(1))))
        else:
            m = _DISP.search(texto)
            if not m:
                continue
            corpo = texto[m.end():]
            incisos = tuple(dict.fromkeys(re.findall(r"(?m)^\s*([IVXL]{1,6})\s*[-–]", corpo)))
            pars = tuple(dict.fromkeys(re.findall(r"(?m)^\s*§\s*(\d+º(?:-[A-Z])?)", corpo)))
            alineas = tuple(dict.fromkeys(re.findall(r"(?m)^\s*([a-z])\)", corpo)))
            artigo = m.group(1) + ("º" if m.group(2) else "")
            disps.append(Dispositivo(id_, artigo, _chave_lei(m.group(3)), incisos, pars, alineas))
    return fichas, sumulas, disps
