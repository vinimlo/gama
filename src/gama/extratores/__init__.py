# -*- coding: utf-8 -*-
from .base import Extrator
from .regua import ExtratorRegua

__all__ = ["Extrator", "ExtratorRegua", "carregar"]


def carregar(nome: str, modelos: str | None = None) -> Extrator:
    """'regua' | 'neural' | 'neural-cru' | 'uniao'. O neural importa torch só quando pedido.

    'neural' é o de produção: o modelo com a guarda (`guarda.py`). 'neural-cru' é o modelo
    sozinho, para medir o modelo em si (benchmark, treino)."""
    if nome == "regua":
        return ExtratorRegua()
    if nome == "neural":
        from .guarda import ExtratorGuardado
        return ExtratorGuardado(modelos)
    if nome == "neural-cru":
        from .neural import ExtratorNeural
        return ExtratorNeural(modelos)
    if nome == "uniao":
        from .uniao import ExtratorUniao
        return ExtratorUniao(modelos)
    raise ValueError(f"extrator desconhecido: {nome}")
