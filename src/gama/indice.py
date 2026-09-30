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

from .cabecalho import numero_proprio
from .normalizar import achatar, chave_processo

# Apelidos de codigo -> chave da lei (digitos do numero da lei).
# Conjunto fechado: os 13 dispositivos do acervo cobrem 9 diplomas.
APELIDOS_LEI = {
    "cpc": "13105", "codigo de processo civil": "13105", "lei 13105": "13105",
    "cc": "10406", "codigo civil": "10406",
    "clt": "5452", "consolidacao das leis do trabalho": "5452",
    "cpp": "3689", "codigo de processo penal": "3689",
    "cpm": "1001", "codigo penal militar": "1001",
    "cdc": "8078", "codigo de defesa do consumidor": "8078",
    "codigo eleitoral": "4737",
    "cf": "CF", "constituicao federal": "CF", "constituicao da republica": "CF",
    "constituicao": "CF", "carta magna": "CF", "cf/88": "CF",
    "lc 64": "LC64", "lei complementar 64": "LC64",
    "lei complementar n 64": "LC64", "lc 64/1990": "LC64",
}

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


def _chave_lei(descricao: str) -> str:
    """'Lei n 13.105' -> '13105'; 'Constituicao Federal de 1988' -> 'CF'."""
    plano = achatar(descricao)
    if "constitui" in plano:
        return "CF"
    if "complementar" in plano:
        return "LC" + re.sub(r"\D", "", plano.split("complementar")[1][:8])
    digitos = re.sub(r"\D", "", plano.split(",")[0])
    return digitos.lstrip("0") or digitos


@dataclass
class Indice:
    por_processo: dict = field(default_factory=lambda: defaultdict(list))
    sumulas: dict = field(default_factory=dict)      # (num, trib|None, vinc) -> id
    dispositivos: dict = field(default_factory=dict)  # (artigo, chave_lei) -> id
    meta: dict = field(default_factory=dict)          # id -> (tribunal, ano, relator)
    cadeia: dict = field(default_factory=dict)        # id -> cadeia de classe ('ED', 'AgR', 'REspe')

    def candidatos_processo(self, digitos: str) -> list:
        return self.por_processo.get(chave_processo(digitos), [])

    def resolve_sumula(self, numero: str, tribunal: str | None, vinculante: bool) -> list:
        chave = (numero.lstrip("0") or numero, tribunal, vinculante)
        if chave in self.sumulas:
            return [self.sumulas[chave]]
        # Tribunal declarado e diferente: "Sumula 211 do TSE" nao e a 211 do STJ.
        # Resolver pelo numero aqui transformaria inventada em real (tau).
        if tribunal is not None:
            return []
        # Sem tribunal declarado: aceita se o numero for unico no acervo.
        iguais = [i for (n, _t, v), i in self.sumulas.items()
                  if n == chave[0] and v == vinculante]
        return iguais if len(iguais) == 1 else iguais

    def resolve_dispositivo(self, artigo: str, chave_lei: str) -> list:
        chave = (artigo.lstrip("0") or artigo, chave_lei)
        return [self.dispositivos[chave]] if chave in self.dispositivos else []


def construir(caminho_db) -> Indice:
    """Varre o acervo uma vez e monta todos os indices."""
    idx = Indice()
    con = sqlite3.connect(f"file:{caminho_db}?mode=ro", uri=True)
    try:
        linhas = con.execute(
            "SELECT id, natureza, tribunal, ano, relator, texto FROM documentos"
        ).fetchall()
    finally:
        con.close()

    for doc_id, natureza, tribunal, ano, relator, texto in linhas:
        idx.meta[doc_id] = (tribunal, ano, relator)

        if natureza == "acordao":
            # So o NUMERO PROPRIO de cada ficha vira chave (cabecalho.py). O indice
            # antigo indexava todo numero dos primeiros 300 chars e datas, numero
            # de registro e a Lei 13.015/2014 viravam chave: 280 chaves ambiguas,
            # 176 fichas sem chave unica. Agora: 996/996 com numero proprio.
            proprio = numero_proprio(texto, tribunal)
            idx.cadeia[doc_id] = proprio.cadeia
            for chave in proprio.chaves:
                if doc_id not in idx.por_processo[chave]:
                    idx.por_processo[chave].append(doc_id)

        elif natureza == "sumula":
            m = _CAB_SUMULA.search(texto[:120])
            if m:
                vinc = bool(m.group(1))
                idx.sumulas[(m.group(2).lstrip("0"), m.group(3).upper(), vinc)] = doc_id

        elif natureza == "dispositivo":
            m = _CAB_DISPOSITIVO.search(texto[:200])
            if m:
                artigo = m.group(1).lstrip("0") or m.group(1)
                idx.dispositivos[(artigo, _chave_lei(m.group(2)))] = doc_id

    return idx
