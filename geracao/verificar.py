# -*- coding: utf-8 -*-
"""Portões de qualidade de cada documento gerado.

O rótulo é exato por construção; estes portões pegam bug do PRÓPRIO gerador
(offset que escorregou, borda fora da convenção, citação que escapou do rótulo) e
medem, de brinde, onde o resolver discorda do rótulo — cada discordância é um caso
de teste do resolver, não um erro do gerador.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from avaliacao.convencoes import violacoes_de_borda
from gama.classificar import Classificador
from gama.extratores.regua import ExtratorRegua
from gama.formas import DetectorDeForma
from gama.resolver import Resolvedor


@dataclass
class Laudo:
    fatal: list = field(default_factory=list)       # descarta o documento
    resolver: list = field(default_factory=list)    # discordância do resolver (métrica)


def verificar(texto: str, spans: list, idx, nivel: int) -> Laudo:
    """spans = [(inicio, fim, Cit)] já no texto final (depois do ruído, se houver)."""
    laudo = Laudo()
    ordenados = sorted(spans, key=lambda s: s[0])
    for (a1, b1, _), (a2, _b2, _) in zip(ordenados, ordenados[1:]):
        if a2 < b1:
            laudo.fatal.append(f"sobreposição {a1}-{b1} / {a2}")
    for a, b, cit in spans:
        if not (0 <= a < b <= len(texto)):
            laudo.fatal.append(f"offset fora do texto {a}-{b}")
            continue
        trecho = texto[a:b]
        # A quebra de linha troca espaço por "\n" DENTRO do span (47/192 no dev):
        # só isso pode diferir entre o texto final e o renderizado.
        if nivel == 1 and re.sub(r"\s", " ", trecho) != re.sub(r"\s", " ", cit.texto):
            laudo.fatal.append(f"trecho {trecho!r} != {cit.texto!r}")
        if nivel == 1:
            for v in violacoes_de_borda(trecho):
                laudo.fatal.append(f"borda {v}: {trecho!r}")
        sp = DetectorDeForma().span(texto, a, b, cit.rotulo)
        out = Classificador().classificar(sp, Resolvedor(idx).resolver(sp))
        ok = out.classificacao == cit.classe and (cit.classe != "real" or out.id_canonico == cit.id_canonico)
        if not ok:
            if cit.classe == "inventada" and out.classificacao == "real":
                # Inventada que o acervo resolve seria τ no gabarito: nunca aceitar.
                laudo.fatal.append(f"inventada resolve para real: {trecho!r}")
            else:
                laudo.resolver.append((cit.fonte, trecho, cit.classe, out.classificacao))
    # Nenhuma citação fora de rótulo: a régua varre o texto; span dela que não cruza
    # nenhum rótulo é citação que o gerador deixou escapar (ou falso positivo da régua).
    if nivel == 1:
        for s in ExtratorRegua().extrair(texto):
            if not any(s.inicio < b and a < s.fim for a, b, _ in spans):
                laudo.fatal.append(f"citação sem rótulo (régua): {s.trecho!r}")
    return laudo
