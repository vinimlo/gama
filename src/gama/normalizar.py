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
from dataclasses import dataclass

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


class TabelaOCR:
    """O ruído de OCR no sentido letra -> dígito, e o que se lê de volta através dele."""

    def __init__(self, letra_para_digito: dict = OCR_PARA_DIGITO):
        self.letra_para_digito = dict(letra_para_digito)

    def so_digitos(self, bruto: str) -> str:
        """Os dígitos de um identificador, desfazendo o OCR letra->dígito. Letra fora da
        tabela some. No trecho inteiro, as letras do prefixo viram dígito ('REsp 1.741.784'
        -> '51741784'): para achar o número de uma citação, `NumeroDeProcesso.do_trecho`.

        '21737l8'       -> '2173718'
        '1.528.4S5/ RJ' -> '1528455'
        """
        return "".join(ch if ch.isdigit() else self.letra_para_digito[ch]
                       for ch in bruto if ch.isdigit() or ch in self.letra_para_digito)

    def numero(self, bruto: str) -> str | None:
        """'2l1' -> '211'. Exige ao menos um dígito real: palavra pura não é número."""
        if not any(c.isdigit() for c in bruto):
            return None
        return "".join(c if c.isdigit() else self.letra_para_digito.get(c, "") for c in bruto)

    def pedaco_numerico(self, p: str) -> bool:
        """'45g', '21737l8', '6G' e '170076O' são número; 'Rc1', 'n0', 'Especia1' não.

        Dígito real obrigatório, e pelo menos tantos dígitos reais quanto letras — o
        ruído troca uma letra aqui e ali, nunca a maioria dos dígitos de um grupo.
        """
        reais = sum(c.isdigit() for c in p)
        letras = sum(c.isalpha() for c in p)
        return reais > 0 and reais >= letras and all(c.isdigit() or c in self.letra_para_digito for c in p)

    def letras_do_digito(self) -> dict:
        """A inversa, só com letras: '1' -> ['l', 'I', 'i']. Na ordem da tabela."""
        inversa: dict = {}
        for letra, dig in self.letra_para_digito.items():
            if letra.isalpha():
                inversa.setdefault(dig, []).append(letra)
        return inversa


OCR = TabelaOCR()


class Normalizador:
    """Formas canônicas de texto, para casar texto corrido e nomes."""

    @staticmethod
    def sem_acento(texto: str) -> str:
        d = unicodedata.normalize("NFD", texto)
        return "".join(c for c in d if unicodedata.category(c) != "Mn")

    @staticmethod
    def achatar(texto: str) -> str:
        """Minúsculas, sem acento, espaços colapsados. Para casar texto corrido."""
        return re.sub(r"\s+", " ", Normalizador.sem_acento(texto).lower()).strip()

    @staticmethod
    def esqueleto(texto: str) -> str:
        """Forma canônica tolerante a OCR, para casar NOMES (diploma, classe), nunca números.

        'Códig0 de Pr0cesso Civil' e 'Código de Processo Civil' -> 'eodlgo de proeesso elvll'.
        A troca é muitos-para-um e aplicada aos DOIS lados (texto e apelido): o apelido
        não precisa conhecer as variantes de ruído. Cobre as trocas documentadas pela
        organização (0/O, 1/l, 5/S, m/rn) e as vistas no dev set (e/c, i/l).
        """
        s = Normalizador.achatar(texto).replace("rn", "m")
        return s.translate(str.maketrans("015ic", "olsle"))


# Separadores que ficam DENTRO de um número (1.741.784, 7000449-40.2023, "1.741. 784").
# "/", "(", ")" e "," encerram o número: dali em diante vem a UF ou o texto.
_SEP_INTERNO = re.compile(r"[.\-\s ]+")
_PARADA = re.compile(r"[/(),;:]")


@dataclass(frozen=True)
class NumeroDeProcesso:
    """Os dígitos de um número de processo, já sem o ruído de OCR."""
    digitos: str

    @property
    def chave(self) -> str:
        """Chave canônica de comparação: sem zeros à esquerda.

        É o que permite casar 'REsp 1.741.784' com 'RECURSO ESPECIAL Nº 1741784'.
        Zeros à esquerda caem porque o gabarito e o acervo divergem neles.
        """
        return self.digitos.lstrip("0") or self.digitos

    @classmethod
    def do_bruto(cls, bruto: str, ocr: TabelaOCR = OCR) -> NumeroDeProcesso:
        """O número já isolado ('1.741.784', '7000449-40.2023.7.00.0000')."""
        return cls(ocr.so_digitos(bruto))

    @classmethod
    def do_trecho(cls, trecho: str, ocr: TabelaOCR = OCR) -> NumeroDeProcesso:
        """O número de uma citação, a partir do span inteiro.

        Robusto ao ruído da organização: localiza o número por PEDAÇOS entre
        separadores, sem deixar que letras do prefixo que viraram dígito ("Rc1",
        "n0") ou letras-dígito das palavras vizinhas ("AgInt": A=4, g=9, I=1) sejam
        costuradas ao número. Entre as sequências de pedaços numéricos consecutivos,
        vence a de mais dígitos reais.
        """
        melhor, melhor_reais = "", 0
        for trecho_livre in _PARADA.split(trecho.replace("\\n", "\n")):
            seq, reais = [], 0
            for p in _SEP_INTERNO.split(trecho_livre) + [""]:   # sentinela fecha a última sequência
                if p and ocr.pedaco_numerico(p):
                    seq.append(p)
                    reais += sum(c.isdigit() for c in p)
                    continue
                if reais > melhor_reais:
                    melhor, melhor_reais = "".join(seq), reais
                seq, reais = [], 0
        return cls.do_bruto(melhor, ocr)
