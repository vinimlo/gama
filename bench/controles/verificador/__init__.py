# -*- coding: utf-8 -*-
"""Verificador de candidatos: um modelo de decisão julga os candidatos do Gama v1.3 e da régua.

`nucleo.py` monta os candidatos e as políticas A/B/C e pontua pelo harness (`bench.controles.avaliar`,
importado, nunca alterado); `__main__.py` é o CLI; `laya_job.py` é o script UV do HF Jobs que
grava a probabilidade do Laya por candidato.
"""
