# -*- coding: utf-8 -*-
"""Resolucao de uma citacao contra o acervo congelado.

Recebe um Span (ja com o NUCLEO numerico isolado em `span.digitos`) e devolve
os ids candidatos. Nunca aplicar `so_digitos` ao trecho inteiro: o prefixo
("APL", "AgInt", "AREsp") vira digito pela tabela de OCR e destroi a chave --
e o erro e silencioso, so aparece como "zero candidatos".
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .cabecalho import CadeiaDeClasse
from .indice import Indice
from .leis import IdentificadorDeLei
from .normalizar import OCR, Normalizador, NumeroDeProcesso
from .span import Span

_SUM_NUM = re.compile(
    # Tolerante ao ruido documentado: "5umula" (S->5), "Surn." (m->rn), "SumuIa"
    # (l->I), "Sumula" com til, "VincuIante". O numero aceita letra de OCR
    # ("2l1") e passa por TabelaOCR.numero -- digito nunca vira outro digito.
    r"[s5S]\s*[uú]\s*(?:m|rn)(?:\s*[uú]\s*[lI1]\s*[aã]|\.)?\s*(v\S{3,}?[^\W\d_])?\s*"
    r"n?[ºo°.\s]*([\dOolIiSsGgZ]{1,4})\b", re.I)
_SUM_TRIB = re.compile(r"\b(STJ|STF|TST|TSE|STM)\b", re.I)
_ART_NUM = re.compile(r"[aáàãâ]\s*r\s*t(?:\s*[iíl1]\s*g\s*[oóôõ0]|\.|\b)\s*(\d{1,4}(?:\.\d{3})?)", re.I)
_CNJ = re.compile(r"\d{7}\s*-\s*\d{2}\s*\.\s*\d{4}")


@dataclass
class Resolucao:
    ids: list
    via: str          # processo | sumula | dispositivo | vaga
    ambiguidade: int = 0


class ClasseProcessual:
    """A classe processual de uma citação ('resp', 'rr') e o tribunal que ela implica."""

    # Classe processual -> tribunal. Usado so para desempatar candidatos.
    TRIBUNAL_DA_CLASSE = {
        "resp": "STJ", "aresp": "STJ", "rhc": "STJ", "hc": "STJ", "rms": "STJ",
        "re": "STF", "rcl": "STF", "recl": "STF", "reclamacao": "STF", "ms": "STF",
        "respe": "TSE", "arespel": "TSE", "rp": "TSE", "ai": "TSE",
        "rr": "TST", "arr": "TST", "airr": "TST", "ar": "TST",
        "apl": "STM", "rse": "STM",
    }
    # Formas por extenso -> mesma sigla. Sem isto, "AgInt no Recurso Especial
    # n 1.597.443 - PR" nao deduz STJ e um empate de 2 candidatos vira
    # `incompleta` em vez de `real`.
    POR_EXTENSO = {
        "recurso especial": "resp", "agravo em recurso especial": "aresp",
        "recurso em habeas corpus": "rhc", "habeas corpus": "hc",
        "recurso de revista": "rr", "agravo de instrumento": "ai",
        "mandado de seguranca": "ms", "recurso especial eleitoral": "respe",
        "reclamacao": "rcl", "recurso extraordinario": "re",
        "recurso ordinario": "rr", "acao rescisoria": "ar",
    }

    @classmethod
    def ler(cls, trecho: str) -> str | None:
        plano = Normalizador.achatar(trecho)
        for frase in sorted(cls.POR_EXTENSO, key=len, reverse=True):
            if frase in plano:
                return cls.POR_EXTENSO[frase]
        for tok in plano.replace("-", " ").split():
            t = tok.strip(".,")
            if t in cls.TRIBUNAL_DA_CLASSE:
                return t
        return None

    @classmethod
    def tribunal(cls, trecho: str) -> str | None:
        return cls.TRIBUNAL_DA_CLASSE.get(cls.ler(trecho) or "")


# ------------------------------------------------------------------ desempate

class Desempate(ABC):
    """Estreita os candidatos de uma citação de processo; sem critério, devolve todos."""

    @abstractmethod
    def estreitar(self, trecho: str, candidatos: list, idx: Indice) -> list:
        ...


class DesempatePorCadeia(Desempate):
    """Fichas que dividem o numero (ED, AgR e principal do mesmo processo) se
    separam pela cadeia de classe da citacao. Quatro niveis, do mais estrito ao
    mais frouxo; so decide se sobrar exatamente um candidato."""
    NIVEIS = (lambda c: c, CadeiaDeClasse.sem_ordinal, CadeiaDeClasse.primeiro_elo)

    def estreitar(self, trecho: str, candidatos: list, idx: Indice) -> list:
        escolhido = self._escolher(CadeiaDeClasse.ler(trecho), candidatos, idx)
        return [escolhido] if escolhido is not None else candidatos

    def _escolher(self, cit: CadeiaDeClasse, candidatos: list, idx: Indice):
        if not cit:
            return None
        cadeia = {d: idx.processos.cadeia(d) for d in candidatos}
        for nivel in self.NIVEIS:
            alvo = nivel(cit)
            iguais = [d for d in candidatos if nivel(cadeia[d]) == alvo]
            if len(iguais) == 1:
                return iguais[0]
        # Quarto nivel: a cadeia lida com ruido ou abreviacao que o leitor nao reconhece. Entre
        # as fichas em que ela cabe, a de mesmo comprimento ("AgRG" lido como "Ag" ainda conta
        # um elo). Os unicos erros de resolucao do estresse eram deste tipo (N2).
        cabem = [d for d in candidatos if cit.cabe_em(cadeia[d])]
        if len(cabem) > 1:
            cabem = [d for d in cabem if len(cadeia[d]) == len(cit)] or cabem
        return cabem[0] if len(cabem) == 1 else None


class DesempatePorTribunal(Desempate):
    """Pelo tribunal deduzido da classe processual da citação. Se nenhum candidato for
    daquele tribunal, não estreita."""

    def estreitar(self, trecho: str, candidatos: list, idx: Indice) -> list:
        alvo = ClasseProcessual.tribunal(trecho)
        if not alvo:
            return candidatos
        return [i for i in candidatos if idx.tribunal(i) == alvo] or candidatos


# ------------------------------------------------------------------ estratégias por forma

class Estrategia(ABC):
    """Resolve uma forma de citação contra o índice."""
    via: str

    @abstractmethod
    def resolver(self, span: Span, idx: Indice) -> Resolucao:
        ...


class ResolucaoVaga(Estrategia):
    """Citação sem número: nada a procurar."""
    via = "vaga"

    def resolver(self, span: Span, idx: Indice) -> Resolucao:
        return Resolucao([], self.via)


class ResolucaoDeSumula(Estrategia):
    via = "sumula"

    def resolver(self, span: Span, idx: Indice) -> Resolucao:
        m = _SUM_NUM.search(span.trecho)
        numero = OCR.numero(m.group(2)) if m else None
        if not numero:
            return Resolucao([], self.via)
        trib = _SUM_TRIB.search(span.trecho)
        # "Vinculantc", "Vineulantê", "Vinculãnte": vinculante pelo esqueleto da palavra
        vinc = bool(m.group(1)) and "vlneulante" in Normalizador.esqueleto(m.group(1))
        ids = idx.sumulas.resolver(numero, trib.group(1).upper() if trib else None, vinc)
        return Resolucao(ids, self.via, max(0, len(ids) - 1))


class ResolucaoDeDispositivo(Estrategia):
    """Artigo (e tema, que a régua marca com a mesma forma de leitura) de uma lei."""
    via = "dispositivo"

    def __init__(self, leis: IdentificadorDeLei | None = None):
        self.leis = leis or IdentificadorDeLei()

    def resolver(self, span: Span, idx: Indice) -> Resolucao:
        m = _ART_NUM.search(span.trecho)
        if not m:
            return Resolucao([], self.via)
        chave = self.leis.da_citacao(span.trecho)
        if chave is None:
            return Resolucao([], self.via)
        return Resolucao(idx.dispositivos.resolver(re.sub(r"\D", "", m.group(1)), chave), self.via)


class ResolucaoDeProcesso(Estrategia):
    """cnj | processo. O número sai do TEXTO do span, não de um núcleo entregue pelo
    extrator: assim a régua e o extrator neural passam pelo mesmo caminho, e o neural só
    precisa acertar a borda. Empate: os desempates, em ordem, até sobrar um."""
    via = "processo"

    def __init__(self, desempates: list | None = None):
        self.desempates = desempates if desempates is not None else [DesempatePorCadeia(),
                                                                     DesempatePorTribunal()]

    @staticmethod
    def chave(trecho: str) -> str:
        """Núcleo numérico do trecho. Número CNJ tem 20 dígitos: com mais que isso, é a classe
        grudada pelo OCR ("AgR-A1 0603026-69.2018...", 'AI' virou 'A1'), e ficam os 20 últimos."""
        nucleo = NumeroDeProcesso.do_trecho(trecho).chave
        if len(nucleo) > 20 and _CNJ.search(trecho):
            return nucleo[-20:].lstrip("0")
        return nucleo

    def resolver(self, span: Span, idx: Indice) -> Resolucao:
        candidatos = idx.candidatos_processo(self.chave(span.trecho) or span.digitos)
        for desempate in self.desempates:
            if len(candidatos) <= 1:
                break
            candidatos = desempate.estreitar(span.trecho, candidatos, idx)
        return Resolucao(candidatos, self.via, max(0, len(candidatos) - 1))


class Resolvedor:
    """Resolve cada span contra o índice, pela estratégia da sua forma."""

    def __init__(self, indice: Indice, estrategias: dict | None = None):
        self.indice = indice
        if estrategias is None:
            dispositivo = ResolucaoDeDispositivo()
            estrategias = {"vaga": ResolucaoVaga(), "sumula": ResolucaoDeSumula(),
                           "artigo": dispositivo, "tema": dispositivo}
        self.estrategias = estrategias
        self.processo = ResolucaoDeProcesso()

    def resolver(self, span: Span) -> Resolucao:
        return self.estrategias.get(span.forma, self.processo).resolver(span, self.indice)


# ------------------------------------------------------------------ fachadas (até a leva 8)
# Nomes antigos, ainda importados pelo laboratório (bench/, geracao/).

TRIBUNAL_DA_CLASSE = ClasseProcessual.TRIBUNAL_DA_CLASSE
CLASSE_POR_EXTENSO = ClasseProcessual.POR_EXTENSO
_LEIS = IdentificadorDeLei()


def _numero_ocr(bruto: str) -> str | None:
    return OCR.numero(bruto)


def _chave_lei_da_citacao(trecho: str) -> str | None:
    return _LEIS.da_citacao(trecho)


def resolver(span: Span, idx: Indice) -> Resolucao:
    return Resolvedor(idx).resolver(span)
