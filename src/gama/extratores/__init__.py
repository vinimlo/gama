# -*- coding: utf-8 -*-
from .base import Extrator
from .regua import ExtratorRegua

__all__ = ["CatalogoDeExtratores", "Extrator", "ExtratorRegua"]


class CatalogoDeExtratores:
    """Monta o extrator pelo nome. O neural importa torch só quando pedido.

    'neural' é o de produção: o modelo com a guarda (`guarda.py`). 'neural-cru' é o modelo
    sozinho, para medir o modelo em si (benchmark, treino). 'uniao' é o modelo mais a régua
    onde ele não marcou nada (D-007: fora da entrega)."""
    NOMES = ("regua", "neural", "neural-cru", "uniao")

    def __init__(self, modelos: str | None = None):
        self.modelos = modelos

    def carregar(self, nome: str) -> Extrator:
        if nome == "regua":
            return ExtratorRegua()
        if nome == "neural":
            from .guarda import ExtratorGuardado
            return ExtratorGuardado(self._neural(), ExtratorRegua())
        if nome == "neural-cru":
            return self._neural()
        if nome == "uniao":
            from .uniao import ExtratorUniao
            return ExtratorUniao(self._neural(), ExtratorRegua())
        raise ValueError(f"extrator desconhecido: {nome}")

    def _neural(self) -> Extrator:
        from .neural import ExtratorNeural
        return ExtratorNeural(self.modelos)
