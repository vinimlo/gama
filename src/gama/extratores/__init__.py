# -*- coding: utf-8 -*-
from .base import Extrator
from .regua import ExtratorRegua

__all__ = ["Extrator", "ExtratorRegua", "carregar"]


def carregar(nome: str, modelos: str | None = None) -> Extrator:
    """'regua' | 'neural' | 'uniao'. O neural importa torch só quando pedido."""
    if nome == "regua":
        return ExtratorRegua()
    if nome == "neural":
        from .neural import ExtratorNeural
        return ExtratorNeural(modelos)
    if nome == "uniao":
        from .uniao import ExtratorUniao
        return ExtratorUniao(modelos)
    raise ValueError(f"extrator desconhecido: {nome}")
