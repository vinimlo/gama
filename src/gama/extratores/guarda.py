# -*- coding: utf-8 -*-
"""Guarda do extrator neural para texto fora do molde da organização.

Duas regras, medidas em `wiki/experimentos/2026-09-23_otimizacao-medida.md`:

1. VAGA colada a outro span sai. Na redação da organização a referência vaga é uma frase
   própria ("julgado do STJ proferido em 2021 pela relatoria de ..."), depois de fim de
   frase: a mais próxima de outra citação está a 4 caracteres (". O "). Em ementa real,
   "Rel. Min. Fulano, julgado em ..." colado a um precedente numerado é o rabo dele.
2. Span com confiança < 0,95 sai; no lugar entra o span da régua que o cruza, se houver e se
   não cruzar um span confiante. Nenhum acerto no estilo da organização fica abaixo de 0,98
   (34.171 spans em dev, estresse e final_v1); os fragmentos de ementa real ("Rel", "2011",
   ")") ficam, com confiança mediana entre 0,85 e 0,94.

No estilo da organização as duas regras não mudam nenhum documento (4.626 medidos); em 305
ementas reais, o F1 de extração vai de 0,605 a 0,808. Diferente da união (D-007), a régua
só entra onde o modelo hesita, e no molde ele não hesita.
"""
from __future__ import annotations

from ..span import Span
from .base import Extrator

# Limites de produção. `Guarda.padrao()` os lê a cada chamada: `bench.controles.avaliar` e
# `bench.controles.verificador` trocam CONFIANCA_MINIMA durante a medida.
VAGA_COLADA = 2            # caracteres entre a VAGA e o span vizinho
CONFIANCA_MINIMA = 0.95


class Guarda:
    """As duas regras da D-008, com os limites da instância."""

    def __init__(self, vaga_colada: int = VAGA_COLADA, confianca_minima: float = CONFIANCA_MINIMA):
        self.vaga_colada = vaga_colada
        self.confianca_minima = confianca_minima

    @classmethod
    def padrao(cls) -> Guarda:
        """Os limites do módulo no momento da chamada."""
        return cls(VAGA_COLADA, CONFIANCA_MINIMA)

    def sem_vaga_colada(self, spans: list[Span]) -> list[Span]:
        """Regra 1: VAGA a até `vaga_colada` caracteres de outro span sai."""
        return [s for s in spans if s.forma != "vaga" or all(
            s.distancia(o) > self.vaga_colada for o in spans if o is not s)]

    def trocar_fracos_pela_regua(self, spans: list[Span], regua: list[Span]) -> list[Span]:
        """Regra 2: span com confiança < `confianca_minima` sai; entra o span da régua que o
        cruza, se não cruzar um span confiante. Devolve ordenado por início."""
        fortes = [s for s in spans if s.confiante(self.confianca_minima)]
        fracos = [s for s in spans if not s.confiante(self.confianca_minima)]
        extra = [r for r in regua
                 if any(r.cruza(w) for w in fracos) and not any(r.cruza(s) for s in fortes)]
        return sorted(fortes + extra, key=lambda s: s.inicio)

    def aplicar(self, spans: list[Span], regua: list[Span]) -> list[Span]:
        """Spans do modelo e da régua, já aparados -> spans que ficam, ordenados.

        A ordem das regras é a medida na D-008: a VAGA é comparada também com os spans fracos
        que a regra 2 vai tirar."""
        return self.trocar_fracos_pela_regua(self.sem_vaga_colada(spans), regua)


class ExtratorGuardado(Extrator):
    """O extrator de produção: o modelo, com a guarda e a régua de reserva.

    Sem `guarda`, usa a de produção com os limites do módulo lidos a cada extração."""
    nome = "neural"

    def __init__(self, neural: Extrator, regua: Extrator, guarda: Guarda | None = None):
        self.neural = neural
        self.regua = regua
        self.guarda = guarda

    def extrair(self, texto: str) -> list[Span]:
        guarda = self.guarda or Guarda.padrao()
        return guarda.aplicar(Span.aparar_todos(self.neural.extrair(texto), texto),
                              Span.aparar_todos(self.regua.extrair(texto), texto))


# ------------------------------------------------------------------ fachadas (até a leva 8)
# Nomes antigos, ainda chamados pelo laboratório (bench/hipoteses.py, bench/controles/).

def confiante(s: Span, minimo: float) -> bool:
    return s.confiante(minimo)


def sem_vaga_colada(spans: list[Span], limite: int) -> list[Span]:
    return Guarda(vaga_colada=limite).sem_vaga_colada(spans)


def trocar_fracos_pela_regua(spans: list[Span], regua: list[Span], minimo: float) -> list[Span]:
    return Guarda(confianca_minima=minimo).trocar_fracos_pela_regua(spans, regua)


def guardar(spans: list[Span], regua: list[Span]) -> list[Span]:
    return Guarda.padrao().aplicar(spans, regua)
