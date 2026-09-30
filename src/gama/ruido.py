# -*- coding: utf-8 -*-
"""Injetor de ruído pela especificação da organização — com offsets rastreados.

A organização descreve o ruído do nível 2 (peso 2×): abreviação, formato do
número, separador de UF, confusões de OCR (0↔O, 1↔l, 5↔S, m↔rn) e quebra de linha
no meio do identificador. O dev set mostra ainda e↔c ("origcm", "dc") e ruído de
diacrítico ("Temã"). Vale para LETRAS e dígitos: 5 de 6 sondas com essas trocas em
palavras-chave quebravam a régua.

Fidelidade aos exemplos da organização (lidos no dev set):
  * número CLÁSSICO recebe troca de letra (21737l8, 1.45g.779, 6G.838, 170076O) e
    formato por número inteiro (1.741.784 / 1741784 / 1.741. 784);
  * número CNJ recebe só ruído de separador (0600316-4920206160182,
    533-80. 2012.6.13.0029, 7220273-- 23.2018...), nunca letra;
  * o PRIMEIRO dígito nunca é trocado (em nenhum exemplo é).

INVARIANTE (garantia da organização): um dígito nunca vira outro dígito. Dígito só
vira LETRA, pela inversa exata de normalizar.OCR_PARA_DIGITO.

Parte do ruído muda o comprimento (m→rn, quebra de linha, "1.741. 784"), então
cada span de entrada é remapeado para o texto de saída.
"""
from __future__ import annotations

import random
import re

from .normalizar import OCR

DIGITO_PARA_LETRA: dict = OCR.letras_do_digito()      # a inversa exata, na ordem da tabela

LETRA_PARA_OUTRA = {
    "l": ["1", "I"], "I": ["l", "1"], "S": ["5"], "O": ["0"], "o": ["0"],
    "e": ["c"], "c": ["e"], "i": ["l"],
}
DIACRITICO = {"a": "ã", "e": "ê", "o": "ô", "ã": "a", "é": "e", "ç": "c", "ó": "o", "á": "a"}

_CNJ = re.compile(r"\d{1,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}")
_CLASSICO = re.compile(r"\d{1,3}(?:\.\d{3})+|\d{4,}")


# Dois canais, medidos no nível 2 da organização em 22/09 (por 1.000 palavras,
# corpo fora das citações): dígito dentro de palavra 0,16 (nosso canal único dava
# 15,7), m->rn ~1,4 (dava 21), "dc" 0,47 (dava 2,5). O ruído forte mora DENTRO das
# citações ("Recl. n° 6G.838/ BA"); fora delas é raro. Fator aplicado fora do span:
FORA = {"digito": 0.1, "rn": 0.07, "letra": 0.2, "letra_digito": 0.01, "diacritico": 0.5}


# Ruído na PALAVRA-CHAVE da citação ("Reclarnação", "Terna", "re1atoria"). O canal por
# caractere raramente acerta a palavra-chave, e é exatamente o ruído que a organização
# documentou (m<->rn, 1<->l, S->5, O->0) e que o dev mostrou (e->c, a->ã). Sem ele o Gama
# v1.1 regrediu em sondas desse tipo (22/09): a palavra-chave ruidosa ficava na fronteira
# de decisão do modelo. Uma troca por citação, nunca no núcleo numérico; letra->dígito só
# em palavra de 5+ letras ("AI"->"A1" colaria no número).
PROB_CHAVE = 0.35
_PALAVRA = re.compile(r"[A-Za-zÀ-ÿ]{3,}")
_CONFUSAO_CHAVE = [("m", "rn"), ("e", "c"), ("l", "1"), ("i", "l"), ("S", "5"),
                   ("O", "0"), ("a", "ã"), ("u", "ú")]


class InjetorDeRuido:
    """Aplica o ruído a um texto com spans marcados e remapeia os spans.

    A ORDEM das chamadas ao `rng` é contrato: o gerador sintético e os conjuntos de
    treino se reproduzem pela semente (test_ruido.py congela a saída).
    """

    def __init__(self, rng: random.Random, intensidade: float = 0.3):
        self.rng = rng
        self.p = intensidade

    @staticmethod
    def _dentro(pos: int, spans: list) -> bool:
        return any(a <= pos < b for a, b in spans)

    @classmethod
    def nucleos(cls, texto: str, spans: list) -> list:
        """(inicio, fim, tipo) dos números dentro de spans; CNJ tem precedência."""
        achados = []
        for m in _CNJ.finditer(texto):
            if cls._dentro(m.start(), spans):
                achados.append((m.start(), m.end(), "cnj"))
        for m in _CLASSICO.finditer(texto):
            if cls._dentro(m.start(), spans) and not any(a <= m.start() < b for a, b, _ in achados):
                achados.append((m.start(), m.end(), "classico"))
        return achados

    def _plano_do_nucleo(self, texto: str, a: int, b: int, tipo: str) -> dict:
        """Decide, por número inteiro, o que cada caractere do núcleo vira."""
        rng, p = self.rng, self.p
        troca = {}
        seg = texto[a:b]
        if tipo == "classico":
            fmt = rng.random()
            pontos = [a + k for k, c in enumerate(seg) if c == "."]
            if pontos and fmt < 0.25 * p * 2:
                for k in pontos:
                    troca[k] = ""                              # 1.741.784 -> 1741784
            elif pontos and fmt < 0.40 * p * 2:
                troca[rng.choice(pontos)] = rng.choice([". ", ".\n"])   # 1.741. 784
            # No máximo UMA letra por número: é o que todos os exemplos da org mostram
            # (21737l8, 1.45g.779, 6G.838, 170076O). Mais que isso deixa de ser "ruído
            # recuperável" e passa a ser outro número.
            digitos = [a + k for k, c in enumerate(seg) if c.isdigit()][1:]   # 1o preservado
            trocaveis = [k for k in digitos if texto[k] in DIGITO_PARA_LETRA]
            if trocaveis and rng.random() < 0.55 * p:
                k = rng.choice(trocaveis)
                troca[k] = rng.choice(DIGITO_PARA_LETRA[texto[k]])
        else:  # cnj: só separadores
            seps = [a + k for k, c in enumerate(seg) if c in ".-"]
            r = rng.random()
            if r < 0.20 * p * 2:
                for k in seps[1:]:
                    troca[k] = ""                              # 0600316-4920206160182
            elif r < 0.35 * p * 2 and seps:
                k = rng.choice(seps)
                # O hífen é a âncora do CNJ: a org o preserva, dobra ou espaça
                # ("7220273-- 23", "0600122- 62"), nunca o troca por ponto.
                troca[k] = rng.choice(["- ", "--", "-\n"] if texto[k] == "-" else [". ", ".\n", " "])
        return troca

    def _plano_palavra_chave(self, texto: str, spans: list, nucleos: list, ocupado: dict) -> dict:
        """Ruído na PALAVRA-CHAVE da citação: no máximo uma troca por span (ver PROB_CHAVE)."""
        rng = self.rng
        troca = {}
        for a, b in spans:
            if rng.random() >= PROB_CHAVE * self.p:
                continue
            palavras = [m for m in _PALAVRA.finditer(texto, a, b)
                        if not any(x <= m.start() < y for x, y, _ in nucleos)]
            if not palavras:
                continue
            # a primeira palavra é a classe ("Reclamação", "Súmula", "Tema", "art", "julgado")
            m = palavras[0] if rng.random() < 0.6 else rng.choice(palavras)
            opcoes = []
            for k, ch in enumerate(m.group()):
                pos = m.start() + k
                if pos in ocupado:
                    continue
                for de, para in _CONFUSAO_CHAVE:
                    if ch == de and not (para.isdigit() and len(m.group()) < 5):
                        opcoes.append((pos, para))
            if opcoes:
                pos, para = rng.choice(opcoes)
                troca[pos] = para
        return troca

    def _caractere(self, texto: str, i: int, spans: list, em_nucleo: bool) -> tuple[str, int]:
        """O que o caractere i vira fora do plano: (emitido, quantos consome)."""
        rng, p = self.rng, self.p
        ch = texto[i]
        span = self._dentro(i, spans)
        if em_nucleo:
            return ch, 1                                    # núcleo: só o plano
        if ch.isdigit() and not span and rng.random() < 0.04 * p * FORA["digito"]:
            return rng.choice(DIGITO_PARA_LETRA.get(ch, [ch])), 1
        if ch == " " and span and rng.random() < 0.18 * p:
            return rng.choice(["\n", "  ", " "]), 1        # quebra no meio do identificador
        if ch == " " and not span and rng.random() < 0.004 * p:
            return rng.choice(["  ", "  ", " "]), 1        # "no  precedente" no corpo
        if ch == "m" and rng.random() < 0.20 * p * (1 if span else FORA["rn"]):
            return "rn", 1                                  # m -> rn
        if (ch == "r" and texto[i + 1:i + 2] == "n" and span == self._dentro(i + 1, spans)
                and not any(i + 1 == a or i + 1 == b for a, b in spans)
                and rng.random() < 0.20 * p * (1 if span else FORA["rn"])):
            return "m", 2                                   # rn -> m
        if ch in LETRA_PARA_OUTRA and rng.random() < 0.07 * p * (
                1 if span else (FORA["letra_digito"] if ch in "lIOoS" else FORA["letra"])):
            return rng.choice(LETRA_PARA_OUTRA[ch]), 1
        if ch in DIACRITICO and rng.random() < 0.03 * p * (1 if span else FORA["diacritico"]):
            return DIACRITICO[ch], 1
        return ch, 1

    def aplicar(self, texto: str, spans: list) -> tuple:
        """Devolve (texto_ruidoso, spans_remapeados). spans = [(inicio, fim)], fim exclusivo."""
        if self.p <= 0:
            return texto, list(spans)
        nucleos = self.nucleos(texto, spans)
        troca = {}
        for a, b, tipo in nucleos:
            troca.update(self._plano_do_nucleo(texto, a, b, tipo))
        troca.update(self._plano_palavra_chave(texto, spans, nucleos, troca))

        saida = []
        mapa = [0] * (len(texto) + 1)
        comp = 0
        i = 0
        while i < len(texto):
            mapa[i] = comp
            if i in troca:
                emit, consome = troca[i], 1
            else:
                emit, consome = self._caractere(texto, i, spans, any(a <= i < b for a, b, _ in nucleos))
            saida.append(emit)
            if consome == 2:
                mapa[i + 1] = comp
            comp += len(emit)
            i += consome
        mapa[len(texto)] = comp
        novo = "".join(saida)
        return novo, self._remapear(spans, mapa, novo)

    @staticmethod
    def _remapear(spans: list, mapa: list, novo: str) -> list:
        """Cada span no texto novo, sem branco nas pontas."""
        remapeados = []
        for a, b in spans:
            na, nb = mapa[a], mapa[b]
            while na < nb and novo[na] in " \n ":
                na += 1
            while nb > na and novo[nb - 1] in " \n ":
                nb -= 1
            remapeados.append((na, nb))
        return remapeados
