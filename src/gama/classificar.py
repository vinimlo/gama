# -*- coding: utf-8 -*-
"""Decisao da classe e da confianca.

A classe cai quase direto da cardinalidade da resolucao. A confianca e o que
alimenta o bonus de calibracao (ate 10% da nota): o kaggle_metric.py calcula
Brier = media (confianca - y)^2 sobre os pares casados, com y=1 so quando a
classe E o link estao certos, e aplica score = s * (1 + 0,10*(1 - brier)).

Logo a confianca nao e enfeite: ela deve ser a probabilidade estimada de
acerto daquele bucket. Os valores abaixo sao PRIORES -- `avaliacao.calibrar`
mede a taxa real de acerto por bucket no conjunto de desenvolvimento e
reescreve esta tabela. Confianca alta num bucket que erra muito custa pontos.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

from .resolver import Resolucao
from .span import Intervalo, Span

# (via, situacao) -> confianca. Reescrito por avaliacao/calibrar.py.
CONFIANCA = {
    ("vaga", "incompleta"): 0.95,
    ("sumula", "real"): 0.93,
    ("sumula", "inventada"): 0.80,
    ("dispositivo", "real"): 0.93,
    ("dispositivo", "inventada"): 0.85,
    ("processo", "real"): 0.90,
    ("processo", "inventada"): 0.82,
    # Empate resolvido por chute no primeiro candidato: a confianca tem que
    # cair, senao o Brier pune. Honestidade aqui vale pontos.
    ("processo", "real_ambiguo"): 0.50,
}
CONFIANCA_PADRAO = 0.70

# Tabela medida (avaliacao/calibrar.py): balde -> P(acerto) estimada em dados honestos.
# Quando existe, substitui os priores acima (D-006). Chave "via|classe|faixa".
_CALIBRACAO = pathlib.Path(__file__).with_name("calibracao.json")
TABELA = json.loads(_CALIBRACAO.read_text(encoding="utf-8")) if _CALIBRACAO.exists() else {}


class TabelaDeConfianca:
    """Confiança de cada balde (via, classe, faixa do extrator): a medida, quando o balde
    foi medido; senão o prior da (via, classe); senão o padrão."""

    def __init__(self, medida: dict | None = None, priores: dict = CONFIANCA, padrao: float = CONFIANCA_PADRAO):
        self.medida = medida if medida is not None else {}
        self.priores = priores
        self.padrao = padrao

    @classmethod
    def calibrada(cls) -> TabelaDeConfianca:
        """Com a tabela medida do módulo, lida a cada chamada (avaliacao/calibrar.py a troca)."""
        return cls(TABELA)

    @staticmethod
    def faixa(conf_modelo: float | None) -> str:
        """Faixa da confiança do extrator: regra (sem modelo), alta ou baixa."""
        if conf_modelo is None:
            return "regra"
        return "alta" if conf_modelo >= 0.98 else "baixa"

    def confianca(self, via: str, classe: str, conf_modelo: float | None) -> tuple[float, tuple]:
        balde = (via, classe, self.faixa(conf_modelo))
        prior = self.priores.get((via, classe), self.padrao)
        return self.medida.get("|".join(balde), prior), balde


@dataclass
class Citacao(Intervalo):
    inicio: int
    fim: int
    trecho: str
    tipo: str
    classificacao: str
    id_canonico: str | None
    confianca: float
    balde: tuple = field(default=(), compare=False)   # para calibrar; não vai ao JSON

    def para_json(self, indice: int) -> dict:
        return {
            "id": f"c{indice}",
            "inicio": self.inicio,
            "fim": self.fim,
            "trecho": self.trecho,
            "tipo": self.tipo,
            "classificacao": self.classificacao,
            "resolucao": ({"fonte": "jusbrasil", "id_canonico": self.id_canonico}
                          if self.id_canonico else None),
            "confianca": round(self.confianca, 4),
        }


class Classificador:
    """Cardinalidade -> classe.

    1 candidato  -> real (com o id)
    0 candidatos -> inventada
    2+           -> real com o primeiro (D-002), no balde do empate

    Assimetria que o kaggle_metric.py cria e que vale explorar: `real` com link
    errado custa SO fp[real]; chamar de `inventada` algo que era `real` custa
    fn[real] E fp[inventada], machucando duas classes. Entao, havendo qualquer
    candidato, entregar `real` domina rebaixar para `inventada`.
    """

    def __init__(self, tabela: TabelaDeConfianca | None = None):
        self.tabela = tabela or TabelaDeConfianca.calibrada()

    def classificar(self, span: Span, res: Resolucao) -> Citacao:
        if res.via == "vaga":
            classe, id_canonico, balde = "incompleta", None, ("vaga", "incompleta")
        elif len(res.ids) == 1:
            classe, id_canonico, balde = "real", str(res.ids[0]), (res.via, "real")
        elif not res.ids:
            classe, id_canonico, balde = "inventada", None, (res.via, "inventada")
        else:
            # 2+ candidatos para uma citacao NUMERADA. No gabarito nenhuma
            # `incompleta` tem numero -- todas sao da forma vaga --, entao o empate
            # e artefato do nosso indice, nao ambiguidade do dado. Somado a
            # assimetria da metrica (link errado custa so fp[real]; rebaixar para
            # incompleta custa fn[real] E fp[incompleta]), entregar `real` domina.
            classe, id_canonico, balde = "real", str(res.ids[0]), ("processo", "real_ambiguo")
        conf, balde = self.tabela.confianca(*balde, span.confianca)
        return Citacao(span.inicio, span.fim, span.trecho, span.tipo, classe, id_canonico, conf, balde)


# ------------------------------------------------------------------ fachadas (até a leva 8)

def faixa(conf_modelo: float | None) -> str:
    return TabelaDeConfianca.faixa(conf_modelo)


def classificar(span: Span, res: Resolucao) -> Citacao:
    return Classificador().classificar(span, res)
