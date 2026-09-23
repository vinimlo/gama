# -*- coding: utf-8 -*-
"""Resolucao de uma citacao contra o acervo congelado.

Recebe um Span (ja com o NUCLEO numerico isolado em `span.digitos`) e devolve
os ids candidatos. Nunca aplicar `so_digitos` ao trecho inteiro: o prefixo
("APL", "AgInt", "AREsp") vira digito pela tabela de OCR e destroi a chave --
e o erro e silencioso, so aparece como "zero candidatos".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .cabecalho import cadeia_de_classe
from .indice import APELIDOS_LEI, Indice
from .normalizar import OCR_PARA_DIGITO, achatar, esqueleto, nucleo_numerico
from .span import Span

# Classe processual -> tribunal. Usado so para desempatar candidatos.
TRIBUNAL_DA_CLASSE = {
    "resp": "STJ", "aresp": "STJ", "rhc": "STJ", "hc": "STJ", "rms": "STJ",
    "re": "STF", "rcl": "STF", "recl": "STF", "reclamacao": "STF", "ms": "STF",
    "respe": "TSE", "arespel": "TSE", "rp": "TSE", "ai": "TSE",
    "rr": "TST", "arr": "TST", "airr": "TST", "ar": "TST",
    "apl": "STM", "rse": "STM",
}

_SUM_NUM = re.compile(
    # Tolerante ao ruido documentado: "5umula" (S->5), "Surn." (m->rn), "SumuIa"
    # (l->I), "Sumula" com til, "VincuIante". O numero aceita letra de OCR
    # ("2l1") e passa por _numero_ocr -- digito nunca vira outro digito.
    r"[s5S]\s*[uú]\s*(?:m|rn)(?:\s*[uú]\s*[lI1]\s*[aã]|\.)?\s*(v\S{3,}?[^\W\d_])?\s*"
    r"n?[ºo°.\s]*([\dOolIiSsGgZ]{1,4})\b", re.I)
_SUM_TRIB = re.compile(r"\b(STJ|STF|TST|TSE|STM)\b", re.I)
_ART_NUM = re.compile(r"[aáàãâ]\s*r\s*t(?:\s*[iíl1]\s*g\s*[oóôõ0]|\.|\b)\s*(\d{1,4}(?:\.\d{3})?)", re.I)
_OCR = "".join(re.escape(c) for c in sorted(OCR_PARA_DIGITO))
_LEI_EXPL = re.compile(
    # O número da lei aceita qualquer letra da tabela de OCR ("131O5/2015", "473T/1965") e
    # espaço depois do ponto ("13. 105"); vai para _numero_ocr, nunca direto. Entre "Lei" e
    # o número cabe UMA palavra qualquer: "Complementar" ruidoso ("Cornplernentar") não
    # pode derrubar a leitura — o tipo complementar sai do esqueleto, não da grafia.
    r"\bl\s*[eéêc]\s*[iíl1]\s*(?:(?P<pula>(?=\S*[^\W\d_]{3})\S{4,})\s+)?n?[ºo°.\s]*"
    r"(?P<num>[\d" + _OCR + r"][\d" + _OCR + r".\s]{0,10}?)\s*(?:/|,|;|\bd[eao]\b|$)", re.I)


def _numero_ocr(bruto: str) -> str | None:
    """'2l1' -> '211'. Exige ao menos um digito real: palavra pura nao e numero."""
    if not any(c.isdigit() for c in bruto):
        return None
    return "".join(c if c.isdigit() else OCR_PARA_DIGITO.get(c, "") for c in bruto)


@dataclass
class Resolucao:
    ids: list
    via: str          # processo | sumula | dispositivo | vaga
    ambiguidade: int = 0


def _chave_lei_da_citacao(trecho: str) -> str | None:
    """'art. 373, I, do CPC' -> '13105'; 'da Lei n 13.105/2015' -> '13105'."""
    plano = achatar(trecho)
    esq = esqueleto(trecho)
    # Sobre o texto ORIGINAL: achatar() minusculiza, e no OCR a caixa carrega
    # informacao ("807B" e 8078; "807b" seria 8076).
    m = _LEI_EXPL.search(trecho)
    # A palavra pulada entre "Lei" e o número só vale se for "Complementar" (ruidoso ou
    # não): "Lei Municipal nº 8.078" virando o CDC federal seria falso real — τ
    # (revisão independente, rodada 3, achado 2).
    if m and m.group("pula") and "omplementar" not in esqueleto(m.group("pula")):
        return None
    if m:
        digitos = _numero_ocr(re.sub(r"[.\s]", "", m.group("num"))) or ""
        if "omplementar" in esq:
            return "LC" + digitos
        if digitos:
            return digitos.lstrip("0") or digitos
    m = re.search(r"\blc\s*n?[º°o.\s]*(\d+)", plano)       # "LC nº 64/1990", "LC 64/90"
    if m:
        return "LC" + m.group(1)
    if "eonstltul" in esq or "earta magna" in esq:
        return "CF"
    # apelidos mais longos primeiro, senao 'cc' casa dentro de outra palavra.
    # Pelo esqueleto: "Consolldacao das Leis do Trabalho" ainda e a CLT.
    for apelido in sorted(APELIDOS_LEI, key=len, reverse=True):
        if re.search(r"\b" + re.escape(esqueleto(apelido)) + r"\b", esq):
            return APELIDOS_LEI[apelido]
    return None


# Formas por extenso -> mesma sigla. Sem isto, "AgInt no Recurso Especial
# n 1.597.443 - PR" nao deduz STJ e um empate de 2 candidatos vira
# `incompleta` em vez de `real`.
CLASSE_POR_EXTENSO = {
    "recurso especial": "resp", "agravo em recurso especial": "aresp",
    "recurso em habeas corpus": "rhc", "habeas corpus": "hc",
    "recurso de revista": "rr", "agravo de instrumento": "ai",
    "mandado de seguranca": "ms", "recurso especial eleitoral": "respe",
    "reclamacao": "rcl", "recurso extraordinario": "re",
    "recurso ordinario": "rr", "acao rescisoria": "ar",
}


def _classe_processual(trecho: str) -> str | None:
    plano = achatar(trecho)
    for frase in sorted(CLASSE_POR_EXTENSO, key=len, reverse=True):
        if frase in plano:
            return CLASSE_POR_EXTENSO[frase]
    for tok in plano.replace("-", " ").split():
        t = tok.strip(".,")
        if t in TRIBUNAL_DA_CLASSE:
            return t
    return None


def _sem_ordinal(cadeia: tuple) -> tuple:
    return tuple(c.rstrip("0123456789") for c in cadeia)


def _desempatar_por_cadeia(trecho: str, candidatos: list, idx: Indice):
    """Fichas que dividem o numero (ED, AgR e principal do mesmo processo) se
    separam pela cadeia de classe da citacao. Tres niveis, do mais estrito ao
    mais frouxo; so decide se sobrar exatamente um candidato."""
    cit = cadeia_de_classe(trecho)
    if not cit:
        return None
    for chave_de in (lambda c: c, _sem_ordinal, lambda c: _sem_ordinal(c)[:1]):
        alvo = chave_de(cit)
        iguais = [d for d in candidatos if chave_de(idx.cadeia.get(d, ())) == alvo]
        if len(iguais) == 1:
            return iguais[0]
    return None


def resolver(span: Span, idx: Indice) -> Resolucao:
    if span.forma == "vaga":
        return Resolucao([], "vaga")

    if span.forma == "sumula":
        m = _SUM_NUM.search(span.trecho)
        numero = _numero_ocr(m.group(2)) if m else None
        if not numero:
            return Resolucao([], "sumula")
        trib = _SUM_TRIB.search(span.trecho)
        # "Vinculantc", "Vineulantê", "Vinculãnte": vinculante pelo esqueleto da palavra
        vinc = bool(m.group(1)) and "vlneulante" in esqueleto(m.group(1))
        ids = idx.resolve_sumula(numero, trib.group(1).upper() if trib else None, vinc)
        return Resolucao(ids, "sumula", max(0, len(ids) - 1))

    if span.forma in ("artigo", "tema"):
        m = _ART_NUM.search(span.trecho)
        if not m:
            return Resolucao([], "dispositivo")
        artigo = re.sub(r"\D", "", m.group(1))
        chave = _chave_lei_da_citacao(span.trecho)
        if chave is None:
            return Resolucao([], "dispositivo")
        return Resolucao(idx.resolve_dispositivo(artigo, chave), "dispositivo")

    # cnj | processo. O número sai do TEXTO do span (nucleo_numerico), não de um
    # núcleo entregue pelo extrator: assim a régua e o extrator neural passam pelo
    # mesmo caminho, e o neural só precisa acertar a borda.
    candidatos = idx.candidatos_processo(nucleo_numerico(span.trecho) or span.digitos)
    if len(candidatos) > 1:
        escolhido = _desempatar_por_cadeia(span.trecho, candidatos, idx)
        if escolhido is not None:
            return Resolucao([escolhido], "processo", 0)
        # Desempate por tribunal deduzido da classe processual da citacao.
        alvo = TRIBUNAL_DA_CLASSE.get(_classe_processual(span.trecho) or "")
        if alvo:
            filtrados = [i for i in candidatos if idx.meta.get(i, (None,))[0] == alvo]
            if len(filtrados) == 1:
                return Resolucao(filtrados, "processo", 0)
            if filtrados:
                candidatos = filtrados
    return Resolucao(candidatos, "processo", max(0, len(candidatos) - 1))
