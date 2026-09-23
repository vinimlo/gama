# -*- coding: utf-8 -*-
"""Normalização de identificadores.

A garantia da organização é o que torna esta etapa determinística:
*um dígito nunca é trocado por outro dígito*. O ruído de OCR troca
LETRA por dígito (l->1, S->5, O->0, G->6, g->9). Logo todo ruído
aplicado a uma citação real é reversível — e é exatamente isso que o
nível 2 (peso 2x) mede.
"""
from __future__ import annotations

import re
import unicodedata

# Confusões de OCR observadas no gabarito, no sentido letra -> dígito.
# Fonte: 1.528.4S5, 170076O, 21737l8, 6G.838, 1.45g.779, 5úmula.
OCR_PARA_DIGITO = {
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "l": "1", "I": "1", "i": "1", "|": "1",
    "Z": "2", "z": "2",
    "A": "4",
    "S": "5", "s": "5",
    "G": "6", "b": "6",
    "T": "7",
    "B": "8",
    "g": "9", "q": "9",
}

# Confusões no sentido dígito -> letra, para normalizar o texto corrido
# ("5úmula" -> "súmula", "Temã" -> "tema").
OCR_PARA_LETRA = {"0": "o", "1": "l", "5": "s", "3": "e", "4": "a", "8": "b"}


def nfc(texto: str) -> str:
    """Unicode NFC. Os .txt já vêm normalizados; não os altere — só comparamos."""
    return unicodedata.normalize("NFC", texto)


def sem_acento(texto: str) -> str:
    d = unicodedata.normalize("NFD", texto)
    return "".join(c for c in d if unicodedata.category(c) != "Mn")


def achatar(texto: str) -> str:
    """Minúsculas, sem acento, espaços colapsados. Para casar texto corrido."""
    return re.sub(r"\s+", " ", sem_acento(texto).lower()).strip()


def esqueleto(texto: str) -> str:
    """Forma canônica tolerante a OCR, para casar NOMES (diploma, classe), nunca números.

    'Códig0 de Pr0cesso Civil' e 'Código de Processo Civil' -> 'eodlgo de proeesso elvll'.
    A troca é muitos-para-um e aplicada aos DOIS lados (texto e apelido): o apelido
    não precisa conhecer as variantes de ruído. Cobre as trocas documentadas pela
    organização (0/O, 1/l, 5/S, m/rn) e as vistas no dev set (e/c, i/l).
    """
    s = achatar(texto).replace("rn", "m")
    return s.translate(str.maketrans("015ic", "olsle"))


def so_digitos(bruto: str) -> str:
    """Extrai os dígitos de um identificador, desfazendo o OCR letra->dígito.

    'AgInt no RESP 21737l8 - SP' -> '2173718'
    '1.528.4S5/ RJ'              -> '1528455'
    """
    saida = []
    for ch in bruto:
        if ch.isdigit():
            saida.append(ch)
        elif ch in OCR_PARA_DIGITO:
            saida.append(OCR_PARA_DIGITO[ch])
    return "".join(saida)


def agrupar_milhares(digitos: str) -> str:
    """'1741784' -> '1.741.784'.

    O índice FTS5 usa tokenizador unicode61, que quebra em qualquer caractere
    não alfanumérico: '1.741.784' vira os tokens 1|741|784. Buscar '1741784'
    é um token só, que não existe no índice. Reagrupar é obrigatório.
    """
    if not digitos:
        return ""
    partes = []
    while len(digitos) > 3:
        partes.insert(0, digitos[-3:])
        digitos = digitos[:-3]
    partes.insert(0, digitos)
    return ".".join(partes)


CNJ_RE = re.compile(r"^(\d{7})(\d{2})(\d{4})(\d)(\d{2})(\d{4})$")


def normalizar_cnj(digitos: str) -> str | None:
    """20 dígitos -> 'NNNNNNN-DD.AAAA.J.TR.OOOO' (forma canônica do CNJ).

    O gabarito traz CNJ com e sem pontuação ('0600316-4920206160182'),
    então a chave de comparação é sempre a sequência de 20 dígitos.
    """
    m = CNJ_RE.match(digitos)
    if not m:
        return None
    return "{}-{}.{}.{}.{}.{}".format(*m.groups())


def chave_processo(bruto: str) -> str:
    """Chave canônica de comparação: só os dígitos, sem zeros à esquerda.

    É o que permite casar 'REsp 1.741.784' com 'RECURSO ESPECIAL Nº 1741784'.
    Zeros à esquerda caem porque o gabarito e o acervo divergem neles.
    """
    d = so_digitos(bruto)
    return d.lstrip("0") or d


# Separadores que ficam DENTRO de um número (1.741.784, 7000449-40.2023, "1.741. 784").
# "/", "(", ")" e "," encerram o número: dali em diante vem a UF ou o texto.
_SEP_INTERNO = re.compile(r"[.\-\s ]+")
_PARADA = re.compile(r"[/(),;:]")


def _pedaco_numerico(p: str) -> bool:
    """'45g', '21737l8', '6G' e '170076O' são número; 'Rc1', 'n0', 'Especia1' não.

    Dígito real obrigatório, e pelo menos tantos dígitos reais quanto letras — o
    ruído troca uma letra aqui e ali, nunca a maioria dos dígitos de um grupo.
    """
    reais = sum(c.isdigit() for c in p)
    letras = sum(c.isalpha() for c in p)
    return reais > 0 and reais >= letras and all(c.isdigit() or c in OCR_PARA_DIGITO for c in p)


def nucleo_numerico(trecho: str) -> str:
    """Chave canônica do número de uma citação, a partir do span inteiro.

    Robusto ao ruído da organização: localiza o número por PEDAÇOS entre
    separadores, sem deixar que letras do prefixo que viraram dígito ("Rc1",
    "n0") ou letras-dígito das palavras vizinhas ("AgInt": A=4, g=9, I=1) sejam
    costuradas ao número. Entre as sequências de pedaços numéricos consecutivos,
    vence a de mais dígitos reais.
    """
    texto = _PARADA.split(trecho.replace("\\n", "\n"))
    melhor, melhor_reais = "", 0
    for trecho_livre in texto:
        partes = _SEP_INTERNO.split(trecho_livre)
        seq, reais = [], 0
        for p in partes + [""]:                          # sentinela fecha a última sequência
            if p and _pedaco_numerico(p):
                seq.append(p)
                reais += sum(c.isdigit() for c in p)
                continue
            if reais > melhor_reais:
                melhor, melhor_reais = "".join(seq), reais
            seq, reais = [], 0
    return chave_processo(melhor) if melhor else ""


def variantes_fts(digitos: str) -> list[str]:
    """Formas de frase que o FTS5 consegue casar para o mesmo número."""
    if not digitos:
        return []
    saida = [agrupar_milhares(digitos)]
    cnj = normalizar_cnj(digitos)
    if cnj:
        saida.append(cnj)
    if len(digitos) <= 6:
        saida.append(digitos)
    return list(dict.fromkeys(saida))
