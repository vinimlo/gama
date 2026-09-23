# Gama — verificador de citações jurídicas (Desafio Caça-Alucinações, BRACIS 2026 × Jusbrasil)
# Tudo roda em container. Nada de python no host.

SHELL := /bin/bash
RUN := docker compose run --rm --no-deps gama

.PHONY: help build dados predizer avaliar erros submissao test lint shell limpar minerar corpus

help:  ## Lista os alvos
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | expand -t22

build:  ## Constrói a imagem
	docker compose build

dados:  ## Confere se os dados da organização estão no lugar
	@test -f dados/desafio1_bracis.db || { echo "FALTA dados/desafio1_bracis.db — baixe a aba Data do Kaggle"; exit 1; }
	@test -d dados/txt || { echo "FALTA dados/txt/"; exit 1; }
	@echo "dados ok: $$(ls dados/txt/*.txt | wc -l | tr -d ' ') documentos, \
$$(sqlite3 dados/desafio1_bracis.db 'SELECT COUNT(*) FROM documentos' 2>/dev/null || echo '?') registros"

predizer:  ## Roda o pipeline sobre dados/txt -> saidas/json
	$(RUN) python -m gama.pipeline --input /app/dados/txt --output /app/saidas/json

avaliar: predizer  ## Pontua as saídas com a métrica OFICIAL, por nível
	$(RUN) python -m avaliacao.harness --pred /app/saidas/json

erros:  ## Detalha os erros por categoria (motor da wiki)
	$(RUN) python -m avaliacao.erros --pred /app/saidas/json

submissao: predizer  ## Gera saidas/submission.csv pelo conversor OFICIAL
	$(RUN) python vendor/json_to_submission.py /app/saidas/json /app/saidas/submission.csv
	@echo "pronto: saidas/submission.csv"

test:  ## Testes
	$(RUN) python -m pytest tests/ -q

lint:  ## Checagem rápida de sintaxe
	$(RUN) python -m compileall -q src avaliacao

shell:  ## Shell no container
	$(RUN) bash

limpar:  ## Apaga as saídas geradas
	rm -rf saidas/json saidas/submission.csv

minerar:  ## Mede o recall do extrator em citacoes REAIS do acervo
	$(RUN) python -m avaliacao.minerar --amostra 150

corpus:  ## Extrai o corpus de treino minerado do acervo (JSONL)
	$(RUN) python -m avaliacao.minerar --amostra 1 --dump /app/saidas/corpus-minerado.jsonl

-include Makefile.local
