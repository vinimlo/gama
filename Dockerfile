# Imagem única para desenvolvimento e para o bundle de reprodutibilidade.
# Pesos e dados NÃO entram na imagem (exigência da organização): chegam por volume.
FROM python:3.12-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    TZ=America/Sao_Paulo

WORKDIR /app

# torch primeiro, do índice do alvo. Padrão: CUDA 12.6 — roda com drivers NVIDIA a partir
# de ~525; o torch 2.14 do PyPI traz CUDA 13, que exige driver bem mais novo, e a máquina
# da avaliação não é nossa. Sem GPU visível o extrator cai para CPU (~3 s/documento,
# dentro do teto de 60 s). Localmente (Mac/ARM) o compose passa o índice de CPU.
ARG TORCH_INDEX="https://download.pytorch.org/whl/cu126"
RUN if [ -n "$TORCH_INDEX" ]; then \
        pip install --no-cache-dir torch==2.14.0 --index-url "$TORCH_INDEX"; \
    else \
        pip install --no-cache-dir torch==2.14.0; \
    fi

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY vendor/ ./vendor/
COPY avaliacao/ ./avaliacao/

ENV PYTHONPATH=/app/src:/app \
    GAMA_EXTRATOR=neural \
    GAMA_MODELOS=/models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Contrato de execução exigido pela organização:
#   docker run --gpus all -v <pesos>:/models:ro -v <acervo>:/app/dados:ro <img> \
#       --input /data/in --output /data/out
ENTRYPOINT ["python", "-m", "gama.pipeline"]
CMD ["--input", "/data/in", "--output", "/data/out"]

# ---------------------------------------------------------------- lab
# Ferramentas de geração, validação adversarial e gate de tokenização.
# Não entra na imagem avaliada: o alvo padrão (último estágio) é o runtime.
FROM base AS lab
ENV HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
COPY requirements-lab.txt .
RUN pip install --no-cache-dir -r requirements-lab.txt
ENTRYPOINT []

# ---------------------------------------------------------------- runtime
FROM base AS runtime
