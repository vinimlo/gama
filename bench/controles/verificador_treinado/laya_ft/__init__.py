# -*- coding: utf-8 -*-
"""Candidato `laya_ft`: o Laya multilíngue (convaiinnovations/laya@55cf4c4, subpasta multilingual,
laya==0.3.22) fine-tunado como verificador de candidatos pelo caminho de treino da própria
biblioteca (o notebook de fine-tune do repositório: RLCD + entropia cruzada suave, temperatura
ajustada depois), com a pergunta f3 do experimento zero-shot.

    treinar_job.py   script UV do HF Jobs: treino com grade de lr do encoder, early stopping e
                     temperatura escolhidos só na validação interna -> vinimlo/gama-exp-verif-laya_ft
    inferir_job.py   script UV do HF Jobs: score de cada um dos 13.664 candidatos (e da validação
                     interna) a partir dos pesos gravados, tempo por candidato
    laya_min.py      cópia mínima (Apache-2.0) do forward do Laya 0.3.22 para pontuar em CPU no
                     container sem instalar a biblioteca (checagem de paridade e dev local)
    __main__.py      CLI local (container): políticas pelo CLI do verificador, AUROC, paridade, dev
"""
