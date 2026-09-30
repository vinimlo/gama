# -*- coding: utf-8 -*-
"""Número próprio e cadeia de classe de cada ficha do acervo.

Isto é o lado do ÍNDICE (a base canônica), não o extrator: aqui lemos o cabeçalho
de documentos que a organização nos deu, num formato estável por tribunal, para
saber qual número identifica cada ficha. Não decide o que é citação no documento
de entrada — isso é trabalho do extrator.

Por que por tribunal. O índice anterior indexava todo número dos primeiros 300
caracteres, e datas ("06/11/2014"), números de registro do STJ e a Lei 13.015/2014
viravam chave — 280 chaves ambíguas, 176 fichas sem chave única. Cada tribunal
escreve o próprio número num lugar fixo:

  STF  ... PLENÁRIO EMB.DECL. NO AG.REG. EM MANDADO DE SEGURANÇA 32.714 MATO GROSSO ... RELATORA
  STJ  AgRg no RECURSO ESPECIAL Nº 1.205.500 - SC (2010⁄0146585-7)
  STM  APELAÇÃO Nº 7000101-61.2019.7.00.0000
  TSE  ... AGRAVO DE INSTRUMENTO Nº 10.353 ( 38964-78.2008.6.00.0000) - CLASSE 6
  TST  discutidos estes autos de ... nº TST-ED-ED-E-ED-ED-ARR-1575-04.2016.5.20.0001

A CADEIA DE CLASSE (ED · AgR · MS) é o que separa fichas que dividem o mesmo
número: embargos, agravo e decisão principal do mesmo processo.
"""
from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .normalizar import NumeroDeProcesso

# ------------------------------------------------------------------ cadeia de classe

# Formas por extenso / abreviadas -> sigla canônica. Ordem: mais longa primeiro,
# para "AGRAVO EM RECURSO ESPECIAL" não virar "AGRAVO" + "RECURSO ESPECIAL".
_CLASSES = [
    ("embargos de divergencia", "EDv"), ("emb.div.", "EDv"), ("emb. div.", "EDv"), ("emb div", "EDv"), ("edv", "EDv"),
    ("eresp", "EDv"),                        # Embargos de Divergência em REsp
    ("embargos de declaracao", "ED"), ("emb.decl.", "ED"), ("emb. decl.", "ED"), ("emb decl", "ED"),
    ("edcl", "ED"), ("eds", "ED"), ("ed", "ED"),
    ("agravo regimental", "AgR"), ("ag.reg.", "AgR"), ("ag. reg.", "AgR"), ("ag reg", "AgR"),
    ("agrg", "AgR"), ("agr", "AgR"),
    ("agravo interno", "AgInt"), ("agint", "AgInt"), ("ag.int.", "AgInt"),
    ("agravo em recurso especial eleitoral", "AREspe"),
    ("agravo em recurso especial", "AREsp"), ("aresp", "AREsp"), ("aresp.", "AREsp"),
    ("recurso especial eleitoral", "REspe"), ("respe", "REspe"), ("respel", "REspe"), ("arespel", "AREspe"),
    ("recurso especial", "REsp"), ("resp", "REsp"), ("rec. esp.", "REsp"), ("r.esp.", "REsp"),
    ("recurso extraordinario com agravo", "ARE"), ("are", "ARE"),
    ("recurso extraordinario", "RE"), ("re", "RE"),
    ("recurso em habeas corpus", "RHC"), ("rhc", "RHC"),
    ("habeas corpus", "HC"), ("hc", "HC"), ("h.c.", "HC"),
    ("recurso em mandado de seguranca", "RMS"), ("rms", "RMS"),
    ("mandado de seguranca", "MS"), ("ms", "MS"),
    ("agravo de instrumento", "AI"), ("ai", "AI"),
    ("recurso em sentido estrito", "RSE"), ("rse", "RSE"),
    ("apelacao", "APL"), ("apl", "APL"),
    ("recurso de revista com agravo", "ARR"), ("arr", "ARR"),
    ("agravo de instrumento em recurso de revista", "AIRR"), ("airr", "AIRR"),
    ("recurso de revista", "RR"), ("rr", "RR"),
    ("recurso ordinario", "RO"), ("ro", "RO"),
    ("acao rescisoria", "AR"), ("ar", "AR"),
    ("reclamacao", "Rcl"), ("rcl", "Rcl"), ("recl.", "Rcl"), ("recl", "Rcl"),
    ("representacao", "Rp"), ("rp", "Rp"),
    ("suspensao de liminar e de sentenca", "SLS"), ("sls", "SLS"),
    ("acao direta de inconstitucionalidade", "ADI"), ("adi", "ADI"),
    ("embargos", "E"), ("e", "E"),
    ("agravo", "Ag"), ("ag", "Ag"),
]


def _plano(s: str) -> str:
    d = unicodedata.normalize("NFD", s.lower())
    d = "".join(c for c in d if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", d).strip()


_ORDINAIS = {"segundo": "2", "segundos": "2", "segunda": "2", "terceiro": "3",
             "terceiros": "3", "quarto": "4", "2o": "2", "2º": "2", "3o": "3", "3º": "3"}


def _esq_tok(t: str) -> str:
    """Esqueleto OCR de UM token já achatado ("ernbargos" = "embargos", "aglnt" = "agint",
    "rcc." = "rec."), aplicado aos dois lados. Só em token de 4+ caracteres: sigla curta
    não é esqueletizada — "al" (UF) e "ai" (Agravo de Instrumento), "rel" e "rcl"
    colidiriam (revisão independente, rodada 3, achado 3)."""
    if len(t) < 4:
        return t
    return t.replace("rn", "m").translate(str.maketrans("015ic", "olsle"))


def _esq(p: str) -> str:
    return " ".join(_esq_tok(t) for t in p.split())


class CadeiaDeClasse(tuple):
    """As siglas de classe de um processo, de fora para dentro: ('ED', 'AgR', 'MS').

    É uma tupla: compara, indexa e vira chave como tal. O ordinal fica colado na sigla
    ("SEGUNDO AG.REG." -> 'AgR2')."""
    FORMAS = [(tuple(_esq(f).split()), s) for f, s in _CLASSES]
    ORDINAIS = {_esq(k): v for k, v in _ORDINAIS.items()}

    @classmethod
    def ler(cls, texto: str) -> CadeiaDeClasse:
        """'EMB.DECL. NO AG.REG. EM MANDADO DE SEGURANÇA' -> ('ED', 'AgR', 'MS').

        Também lê cadeias de siglas com hífen: 'TST-ED-ED-E-ED-ED-ARR' ->
        ('ED', 'ED', 'E', 'ED', 'ED', 'ARR'). O prefixo do tribunal ('TST') é descartado.
        Palavra desconhecida é ignorada.
        """
        # Texto colado na exportação do STJ: "AgInt nosEMBARGOS" -> "AgInt nos EMBARGOS".
        texto = re.sub(r"([a-z])([A-Z]{2,})", r"\1 \2", texto)
        p = _plano(texto)
        p = re.sub(r"\b(tst|stf|stj|tse|stm)\b", " ", p)
        toks = _esq(p.replace("-", " - ")).split()      # tolerância a OCR na classe (goldenset v3)
        cadeia = []
        i = 0
        ordinal = ""
        while i < len(toks):
            if toks[i] in cls.ORDINAIS:
                ordinal = cls.ORDINAIS[toks[i]]         # "SEGUNDO AG.REG." -> "AgR2"
                i += 1
                continue
            for partes, sigla in cls.FORMAS:
                if tuple(toks[i:i + len(partes)]) == partes:
                    cadeia.append(sigla + ordinal)
                    ordinal = ""
                    i += len(partes)
                    break
            else:
                i += 1
        return cls(cadeia)

    def sem_ordinal(self) -> CadeiaDeClasse:
        return CadeiaDeClasse(c.rstrip("0123456789") for c in self)

    def primeiro_elo(self) -> CadeiaDeClasse:
        """O elo de fora, sem ordinal: ('AgR2', 'REsp') -> ('AgR',)."""
        return CadeiaDeClasse(self.sem_ordinal()[:1])

    def cabe_em(self, ficha: tuple) -> bool:
        """Esta cadeia (lida na citação) é subsequência da cadeia da ficha, cada elo lido
        sendo prefixo do elo da ficha: "Ag. Int." que o leitor pulou, ou "Ernbarg0s dc
        Divergêneia" lido como "E" diante de "EDv", ainda cabem."""
        i = 0
        for elo in ficha:
            if i < len(self) and elo.startswith(self[i]):
                i += 1
        return i == len(self)


# ------------------------------------------------------------------ número próprio

_NUM_CLASSICO = r"\d{1,3}(?:\.\d{3})+|\d{2,7}"
_CNJ = r"\d{1,7}\s*-\s*\d{2}\s*\.\s*\d{4}\s*\.\s*\d\s*\.\s*\d{2}\s*\.\s*\d{4}"


@dataclass
class NumeroProprio:
    chaves: list = field(default_factory=list)   # chave de cada número próprio (NumeroDeProcesso)
    cadeia: tuple = CadeiaDeClasse()
    fonte: str = ""                               # de onde saiu (para auditoria)
    numero: str = ""                              # como aparece no cabeçalho (gerador sintético)
    classe: str = ""                              # texto bruto da classe (gerador sintético)


Proprio = NumeroProprio                           # nome antigo (até a leva 8)


def _chave(bruto: str) -> str:
    return NumeroDeProcesso.do_bruto(bruto).chave


class LeitorDeCabecalho(ABC):
    """Lê o número próprio no cabeçalho de UM formato. Sem achar, devolve só a `fonte`."""
    fonte: str

    @abstractmethod
    def ler(self, texto: str) -> NumeroProprio:
        ...

    def _achado(self, m: re.Match, grupo_numero: str = "num", fonte: str | None = None) -> NumeroProprio:
        return NumeroProprio([_chave(m.group(grupo_numero))], CadeiaDeClasse.ler(m.group("classe")),
                             fonte or self.fonte, m.group(grupo_numero), m.group("classe"))

    def _sem_padrao(self) -> NumeroProprio:
        return NumeroProprio(fonte=f"{self.fonte}:sem_padrao")


class LeitorSTF(LeitorDeCabecalho):
    """A classe vem em CAIXA-ALTA depois do órgão julgador; o número vem antes da UF
    por extenso e do "RELATOR". Pula "Página N de M" e a data."""
    fonte = "stf"
    PADRAO = re.compile(
        r"(?:PLEN[AÁ]RIO|PRIMEIRA TURMA|SEGUNDA TURMA)\s+(?P<classe>[A-ZÀ-Ý.\s]+?)\s+"
        r"(?P<num>" + _NUM_CLASSICO + r")\s+[A-ZÀ-Ý]")

    def ler(self, texto: str) -> NumeroProprio:
        m = self.PADRAO.search(texto[:600])
        return self._achado(m) if m else self._sem_padrao()


class LeitorSTJ(LeitorDeCabecalho):
    fonte = "stj"
    PADRAO = re.compile(r"^(?P<classe>.{2,120}?)\s*N[ºo°]\s*(?P<num>" + _NUM_CLASSICO + r")\s*-\s*[A-Z]{2}",
                        re.S)

    def ler(self, texto: str) -> NumeroProprio:
        m = self.PADRAO.search(texto[:400])
        return self._achado(m) if m else self._sem_padrao()


class LeitorSTM(LeitorDeCabecalho):
    fonte = "stm"
    PADRAO = re.compile(r"(?P<classe>[A-ZÀ-Ý][A-ZÀ-Ý\s.]{2,80}?)\s*N[ºo°]\s*(?P<num>" + _CNJ + ")")

    def ler(self, texto: str) -> NumeroProprio:
        m = self.PADRAO.search(texto[:500])
        return self._achado(m) if m else self._sem_padrao()


# "Nº", "N.º", "N o", "No" — o cabeçalho do TSE vem com OCR ruim ("TRE3UNAL").
_MARCA_N = r"N\s*[.]?\s*[ºo°]\s*"
# CNJ tolerante a espaço e quebra: "685-65. 2016.6.11.0055"
_CNJ_FROUXO = (r"\d{1,7}\s*-\s*\d\s*\d\s*\.\s*\d\s*\d\s*\d\s*\d\s*\.\s*\d\s*\.\s*\d\s*\d"
               r"\s*\.\s*\d\s*\d\s*\d\s*\d")    # tolera "201 7" (espaço dentro do ano)
_MARCA_N_TSE = r"N\s*[.ºo°‚,]?\s*"                    # "Nº", "N o", "N‚" (OCR), "N"


class LeitorTSE(LeitorDeCabecalho):
    """Dois formatos: número antigo + CNJ entre parênteses, ou CNJ direto.

    A ORDEM importa: o CNJ direto precisa ser tentado ANTES do número antigo. Com a
    ordem inversa, "RECURSO ESPECIAL ELEITORAL Nº 378-82.2016.6.05.0151" casava o
    padrão antigo em "378" e a chave ficava truncada — 10 citações reais do
    gabarito deixavam de resolver.
    """
    fonte = "tse"
    CNJ_DIRETO = re.compile(r"(?P<classe>(?:[A-ZÀ-Ý€„ƒ][\wÀ-ÿ€„ƒ.]*\s+){1,14}?)" + _MARCA_N_TSE +
                            r"(?P<cnj>" + _CNJ_FROUXO + ")", re.S)
    ANTIGO = re.compile(r"AC[OÓ]RD[AÃ]O\s+(?P<classe>.{3,220}?)\s*" + _MARCA_N + r"(?P<num>" + _NUM_CLASSICO + r")"
                        r"(?:\s*\(\s*(?P<cnj>" + _CNJ_FROUXO + r")\s*\))?", re.S | re.I)

    def ler(self, texto: str) -> NumeroProprio:
        cab = texto[:900]
        ini = re.search(r"ELEITORAL", cab)
        cab = cab[ini.end():] if ini else cab
        m = self.CNJ_DIRETO.search(cab)
        if m:
            return self._achado(m, "cnj", "tse:cnj")
        m = self.ANTIGO.search(cab)
        if m:
            chaves = [_chave(m.group("num"))]
            if m.group("cnj"):
                chaves.append(_chave(m.group("cnj")))
            return NumeroProprio(chaves, CadeiaDeClasse.ler(m.group("classe")), "tse:antigo",
                                 m.group("cnj") or m.group("num"), m.group("classe"))
        return self._sem_padrao()


class LeitorTST(LeitorDeCabecalho):
    """A autorreferência canônica. A cadeia de siglas pode ter 7+ elos e espaços
    ("TST-ED-ED-E- ED-ED-ARR-1575-04.2016...") e aparecer depois do caractere 12.000."""
    fonte = "tst"
    AUTOS = re.compile(
        r"(?:discutidos|examinados)\s+(?:estes|os\s+presentes)\s+autos\s+de\s+(?P<ext>[^.]{0,160}?)"
        r"n\s*\.?\s*[ºo°]?\s*\.?\s*(?P<siglas>(?:[A-Za-z]{1,10}\s*-\s*){1,12})(?P<cnj>" + _CNJ + ")",
        re.S)
    AUTOS_FRACO = re.compile(
        r"estes\s+autos\s+de\s+(?P<ext>.{0,200}?)n\s*\.?\s*[ºo°]?\s*\.?\s*"
        r"(?P<siglas>(?:[A-Za-z]{1,10}\s*-\s*){1,12})(?P<cnj>" + _CNJ + ")", re.S)

    def ler(self, texto: str) -> NumeroProprio:
        for rx, fonte in ((self.AUTOS, "tst:autos"), (self.AUTOS_FRACO, "tst:autos_fraco")):
            m = rx.search(texto)
            if m:
                return NumeroProprio([_chave(m.group("cnj"))],
                                     CadeiaDeClasse.ler(m.group("siglas")) or CadeiaDeClasse.ler(m.group("ext")),
                                     fonte, m.group("cnj"), m.group("siglas"))
        return self._sem_padrao()


class LeitorGenerico(LeitorDeCabecalho):
    """Último recurso para formatos raros: primeira ocorrência de
    '<CLASSE EM CAIXA-ALTA> Nº <número>' nos primeiros 3.000 caracteres."""
    fonte = "generico"
    PADRAO = re.compile(r"(?P<classe>(?:[A-ZÀ-Ý][A-ZÀ-Ý.]*\s+){1,10}?)" + _MARCA_N +
                        r"(?P<num>" + _CNJ_FROUXO + "|" + _NUM_CLASSICO + ")")

    def ler(self, texto: str) -> NumeroProprio:
        m = self.PADRAO.search(texto[:3000])
        return self._achado(m) if m else NumeroProprio(fonte="sem_padrao")


class LeitorDeNumeroProprio:
    """Escolhe o leitor pelo tribunal da ficha; sem achar, tenta o genérico.

    Formato raro: exportação do STJ, extrato de ata do STM, OCR ruim no TSE. No TST o
    genérico é perigoso: o cabeçalho não traz o número e o primeiro "CLASSE Nº" costuma
    ser um processo CITADO — por isso fica de fora (`sem_generico`)."""

    def __init__(self, leitores: dict | None = None, generico: LeitorDeCabecalho | None = None,
                 sem_generico=frozenset({"TST"})):
        self.leitores = leitores if leitores is not None else {
            "STF": LeitorSTF(), "STJ": LeitorSTJ(), "STM": LeitorSTM(), "TSE": LeitorTSE(), "TST": LeitorTST()}
        self.generico = generico or LeitorGenerico()
        self.sem_generico = sem_generico

    def ler(self, texto: str, tribunal: str | None) -> NumeroProprio:
        tribunal = (tribunal or "").upper()
        leitor = self.leitores.get(tribunal)
        p = leitor.ler(texto) if leitor else NumeroProprio(fonte="tribunal_desconhecido")
        if p.chaves or tribunal in self.sem_generico:
            return p
        g = self.generico.ler(texto)
        return g if g.chaves else p


# ------------------------------------------------------------------ fachadas (até a leva 8)

_LEITOR = LeitorDeNumeroProprio()


def cadeia_de_classe(texto: str) -> tuple:
    return CadeiaDeClasse.ler(texto)


def numero_proprio(texto: str, tribunal: str) -> NumeroProprio:
    return _LEITOR.ler(texto, tribunal)
