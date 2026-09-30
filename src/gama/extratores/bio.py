# -*- coding: utf-8 -*-
"""Rótulos BIO por token <-> spans por caractere. Mesmo código no treino e na inferência.

Duas operações puras sobre o `offset_mapping` do tokenizador (fast), em EsquemaBIO:

    rotular(offsets, spans)       spans de caractere -> um rótulo por token
    decodificar(offsets, rotulos) rótulos por token  -> spans de caractere

O teste de ida e volta: rotular -> decodificar tem que devolver EXATAMENTE os
spans do gabarito (IoU 1,0) em 100% do dev set e do goldenset. É aqui que acento,
NBSP e quebra de linha causam bug silencioso: o token SentencePiece "▁SP" pode
começar no espaço, o token "RS," pode engolir a vírgula. A decodificação apara
espaço nas bordas e pontuação final — nenhum span do gabarito termina em .,;:
(192/192, wiki/conceitos/convencoes-de-borda.md).

Documento longo (Janelas): janelas de `max_len` tokens com sobreposição; cada token
herda o rótulo da janela em que está mais ao centro (menos perda de contexto na borda).

Autocontido: sem import de gama (o treino carrega este arquivo sozinho).
"""
from __future__ import annotations

# Aparo nas bordas: qualquer espaço Unicode (str.isspace cobre NBSP, U+2003 etc.) e,
# no fim, pontuação — nenhum span do gabarito termina em .,;: (192/192).
_PONTUACAO_FINAL = ".,;:"


class EsquemaBIO:
    """Os rótulos (O, B-/I- de cada tipo) e a conversão entre spans e rótulos por token."""
    TIPOS = ("JURIS", "LEI", "VAGA")
    ROTULOS = ["O"] + [f"{p}-{t}" for t in TIPOS for p in ("B", "I")]
    ID = {r: i for i, r in enumerate(ROTULOS)}
    IGNORAR = -100                                  # token especial: fora da perda

    def rotular(self, offsets: list, spans: list) -> list[int]:
        """offsets: [(ini, fim)] por token ((0, 0) = especial). spans: [(ini, fim, tipo)].

        B no primeiro token que cruza o span, I nos seguintes. Token que cruza a borda
        (ex.: "▁SP" começando no espaço, "RS," levando a vírgula) conta como dentro — a
        decodificação apara o excesso.
        """
        rot = []
        ordenados = sorted(spans)
        iniciados: set = set()
        for s, e in offsets:
            if s == e:
                rot.append(self.IGNORAR)
                continue
            r = self.ID["O"]
            for k, (a, b, tipo) in enumerate(ordenados):
                if max(s, a) < min(e, b):
                    r = self.ID[f"I-{tipo}"] if k in iniciados else self.ID[f"B-{tipo}"]
                    iniciados.add(k)
                    break
            rot.append(r)
        return rot

    def decodificar(self, offsets: list, rotulos: list, texto: str) -> list[tuple[int, int, str]]:
        """Rótulos por token -> spans (ini, fim, tipo), aparados e SEM sobreposição.

        Token com o mesmo intervalo do anterior (fragmentos de byte de um caractere fora
        do vocabulário) continua o span corrente: nunca abre um segundo span idêntico —
        duas predições com IoU 1 invalidariam a submissão (revisão independente, rodada 1, achado 1).
        """
        return self._aparar(self._agrupar(offsets, rotulos), texto)

    def _agrupar(self, offsets: list, rotulos: list) -> list:
        spans = []
        atual = None                                   # [ini, fim, tipo]
        anterior = None
        for (s, e), r in zip(offsets, rotulos):
            if s == e or r == self.IGNORAR:
                continue
            if (s, e) == anterior:
                continue
            anterior = (s, e)
            nome = self.ROTULOS[r]
            if nome == "O":
                if atual:
                    spans.append(atual)
                atual = None
                continue
            prefixo, tipo = nome.split("-", 1)
            if prefixo == "B" or atual is None or atual[2] != tipo:
                if atual:
                    spans.append(atual)
                atual = [s, e, tipo]
            else:
                atual[1] = e
        if atual:
            spans.append(atual)
        return spans

    @staticmethod
    def _aparar(spans: list, texto: str) -> list:
        out = []
        for k, (a, b, tipo) in enumerate(spans):
            # Nenhum span do gabarito corta palavra (0 em 34.171: dev + treino + estresse).
            # Token que abre no meio da palavra ("Ter"+"na" -> span em "na 725") estende
            # até a borda da palavra — sem atravessar o span vizinho ("Lei 1Lei 2", espaço
            # perdido no OCR: revisão independente, rodada 2, achado 2).
            teto_esq = spans[k - 1][1] if k > 0 else 0
            teto_dir = spans[k + 1][0] if k + 1 < len(spans) else len(texto)
            while a > teto_esq and texto[a - 1].isalnum() and texto[a].isalnum():
                a -= 1
            while b < teto_dir and texto[b - 1].isalnum() and texto[b].isalnum():
                b += 1
            while a < b and texto[a].isspace():
                a += 1
            while b > a and (texto[b - 1].isspace() or texto[b - 1] in _PONTUACAO_FINAL):
                b -= 1
            if b > a and not any(a < fb and fa < b for fa, fb, _ in out):
                out.append((a, b, tipo))
        return out


class Janelas:
    """Cobertura de um documento de n tokens por janelas [i, j) de até `max_len`,
    avançando `passo`; cada token pertence à janela em que está mais ao centro."""

    def __init__(self, max_len: int, passo: int):
        if max_len < 1 or not 1 <= passo <= max_len:
            # passo 0 nunca avança (laço infinito; revisão independente, rodada 1, achado 6); passo > janela deixa
            # token sem janela nenhuma (revisão independente, rodada 2, achado 7)
            raise ValueError(f"janela inválida: max_len={max_len}, passo={passo}")
        self.max_len = max_len
        self.passo = passo

    def cobrir(self, n_tokens: int) -> list[tuple[int, int]]:
        if n_tokens <= self.max_len:
            return [(0, n_tokens)]
        out, i = [], 0
        while True:
            j = min(i + self.max_len, n_tokens)
            out.append((i, j))
            if j == n_tokens:
                return out
            i += self.passo

    @staticmethod
    def dona(i: int, cobertura: list[tuple[int, int]]) -> int:
        """Índice da janela em que o token i está mais longe da borda."""
        melhor, dist = 0, -1
        for k, (a, b) in enumerate(cobertura):
            if a <= i < b:
                d = min(i - a, b - 1 - i)
                if d > dist:
                    melhor, dist = k, d
        return melhor


# Nomes de módulo: o treino copia este arquivo (treino/empacotar.py) e o carrega pelo
# caminho (treino/destilar.py), e usa estes nomes. Ficam.
_ESQUEMA = EsquemaBIO()
TIPOS, ROTULOS, ID, IGNORAR = EsquemaBIO.TIPOS, EsquemaBIO.ROTULOS, EsquemaBIO.ID, EsquemaBIO.IGNORAR
rotular = _ESQUEMA.rotular
decodificar = _ESQUEMA.decodificar


def janelas(n_tokens: int, max_len: int, passo: int) -> list[tuple[int, int]]:
    """Cobertura de [0, n) por janelas [i, j) de até max_len, avançando `passo`."""
    return Janelas(max_len, passo).cobrir(n_tokens)


janela_dona = Janelas.dona
