# -*- coding: utf-8 -*-
"""Configuração comum da suíte.

Fica aqui, e não num pytest.ini na raiz, porque o compose monta só as pastas do código
(`./tests` inclusive): um arquivo solto na raiz não chega ao container.

Três marcas separam o que depende de recurso de fora do repositório. Sem o recurso, o
teste é PULADO com o motivo — nunca falha nem dá erro de fixture:

    dados    acervo e goldenset da organização em ./dados (make dados)
    modelos  pesos do extrator neural em GAMA_MODELOS (make pesos)
    treino   dependências do treino (torch, transformers, huggingface_hub)

Só os testes unitários, em qualquer máquina:  pytest tests/ -m "not dados and not modelos and not treino"
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import pytest

RAIZ = pathlib.Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"
MODELOS = pathlib.Path(os.environ.get("GAMA_MODELOS", str(RAIZ / "modelos")))

# No container o PYTHONPATH já traz /app/src:/app; no host, é aqui.
for p in (RAIZ, RAIZ / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _falta_dados() -> str | None:
    faltam = [n for n in ("desafio1_bracis.db", "goldenset_offsets.csv", "txt")
              if not (DADOS / n).exists()]
    return f"sem {', '.join(faltam)} em {DADOS} (make dados)" if faltam else None


def _falta_modelos() -> str | None:
    if (MODELOS / "config.json").exists():
        return None
    return f"sem pesos do extrator neural em {MODELOS} (make pesos)"


def _falta_treino() -> str | None:
    faltam = [m for m in ("torch", "transformers", "huggingface_hub")
              if importlib.util.find_spec(m) is None]
    return f"sem {', '.join(faltam)} (imagem gama:lab)" if faltam else None


_RECURSOS = {"dados": _falta_dados, "modelos": _falta_modelos, "treino": _falta_treino}


def pytest_configure(config):
    config.addinivalue_line("markers", "dados: precisa do acervo e do goldenset em ./dados")
    config.addinivalue_line("markers", "modelos: precisa dos pesos do extrator neural")
    config.addinivalue_line("markers", "treino: precisa das dependências do treino")


def pytest_collection_modifyitems(config, items):
    motivos = {nome: falta() for nome, falta in _RECURSOS.items()}
    for item in items:
        for nome, motivo in motivos.items():
            if motivo and item.get_closest_marker(nome):
                item.add_marker(pytest.mark.skip(reason=motivo))
