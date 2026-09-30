# Gama — verificador de citações jurídicas (Desafio Caça-Alucinações, BRACIS 2026 × Jusbrasil)
# Tudo roda em container. Nada de python no host.

SHELL := /bin/bash
RUN := docker compose run --rm --no-deps gama

.PHONY: help build pesos dados predizer avaliar erros submissao test lint shell limpar minerar corpus figuras

help:  ## Lista os alvos
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | expand -t22

build:  ## Constrói a imagem
	docker compose build

pesos:  ## Baixa os pesos na revisão fixa do MODELO.md para ./modelos
	@mkdir -p modelos
	docker compose run --rm --no-deps -e HF_HUB_OFFLINE=0 -v "$(CURDIR)/modelos:/pesos" gama python -c \
	  "from huggingface_hub import snapshot_download as s; s('$$(grep -m1 '^repo:' MODELO.md | awk '{print $$2}')', revision='$$(grep -m1 '^revisao:' MODELO.md | awk '{print $$2}')', local_dir='/pesos')"

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

lint:  ## Checagem rápida de sintaxe (bytecode fora das pastas, que estão montadas só para leitura)
	$(RUN) env PYTHONPYCACHEPREFIX=/tmp/pycache python -m compileall -q src avaliacao geracao treino reais bench tests

shell:  ## Shell no container
	$(RUN) bash

limpar:  ## Apaga as saídas geradas
	rm -rf saidas/json saidas/submission.csv

minerar:  ## Mede o recall do extrator em citacoes REAIS do acervo
	$(RUN) python -m avaliacao.minerar --amostra 150

corpus:  ## Extrai o corpus de treino minerado do acervo (JSONL)
	$(RUN) python -m avaliacao.minerar --amostra 1 --dump /app/saidas/corpus-minerado.jsonl

figuras:  ## Redesenha as figuras da documentação (docs/figuras) e o gráfico do benchmark
	$(RUN) python -m bench.grafico
	docker compose run --rm --no-deps -v "$(CURDIR)/docs/figuras:/app/docs/figuras" gama python docs/figuras/gerar.py

-include Makefile.local
