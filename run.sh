#!/usr/bin/env bash
# Ponto de entrada único da solução.
#
#   bash run.sh --preparar                                      (com rede, uma vez)
#   bash run.sh <caminho_db> <pasta_txt> <arquivo_saida.csv>    (sem rede)
#
# Gera <arquivo_saida.csv> no formato da submissão (conversor oficial em vendor/) e, ao lado,
# a pasta <arquivo_saida>.json/ com um JSON por documento no formato do contrato de saída.
#
# Só depende de Docker. A preparação constrói a imagem e baixa os pesos na revisão fixa do
# MODELO.md, e é a única parte que precisa de rede. Quem pular o --preparar não perde nada:
# a execução prepara o que faltar antes de rodar. A execução em si roda sem rede
# (--network none). Com runtime NVIDIA no Docker, usa a GPU; sem ele, roda em CPU (cerca de
# 1 s por documento).
#
# Variáveis opcionais:
#   GAMA_GPU=0|1            força CPU ou GPU (padrão: detecta)
#   GAMA_IMAGEM=nome        tag da imagem (padrão: gama)
#   GAMA_MODELOS_DIR=pasta  onde ficam os pesos (padrão: ./modelos)
#   GAMA_TORCH_INDEX=url    índice do torch no build (padrão do Dockerfile: CUDA 12.6)
set -euo pipefail

uso() {
  echo "uso: bash run.sh --preparar" >&2
  echo "     bash run.sh <caminho_db> <pasta_txt> <arquivo_saida.csv>" >&2
}

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGEM="${GAMA_IMAGEM:-gama}"
MODELOS="${GAMA_MODELOS_DIR:-$RAIZ/modelos}"
absoluto() { (cd "$(dirname "$1")" && printf '%s/%s\n' "$(pwd)" "$(basename "$1")"); }

# GPU, se houver
GPU=()
case "${GAMA_GPU:-auto}" in
  1) GPU=(--gpus all) ;;
  0) ;;
  *) if docker info 2>/dev/null | grep -qi nvidia; then GPU=(--gpus all); fi ;;
esac

preparar() {
  # 1. Imagem (uma vez; precisa de rede)
  if ! docker image inspect "$IMAGEM" >/dev/null 2>&1; then
    echo "construindo a imagem $IMAGEM" >&2
    BUILD_ARGS=()
    [ -n "${GAMA_TORCH_INDEX:-}" ] && BUILD_ARGS=(--build-arg "TORCH_INDEX=$GAMA_TORCH_INDEX")
    docker build ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"} -t "$IMAGEM" "$RAIZ"
  fi

  # 2. Pesos na revisão fixa do MODELO.md (uma vez; precisa de rede)
  REPO="$(grep -m1 '^repo:' "$RAIZ/MODELO.md" | awk '{print $2}')"
  REVISAO="$(grep -m1 '^revisao:' "$RAIZ/MODELO.md" | awk '{print $2}')"
  if [ ! -f "$MODELOS/model.safetensors" ]; then
    echo "baixando $REPO@$REVISAO para $MODELOS" >&2
    mkdir -p "$MODELOS"
    docker run --rm -e HF_HUB_OFFLINE=0 -v "$MODELOS:/models" --entrypoint python "$IMAGEM" \
      -c "from huggingface_hub import snapshot_download as s; s('$REPO', revision='$REVISAO', local_dir='/models')"
  fi
}

case "${1:-}" in
  -h|--help) uso; exit 0 ;;
  --preparar)
    [ "$#" -eq 1 ] || { uso; exit 2; }
    preparar
    # Conferência sem rede: a imagem carrega o torch e diz se enxerga a GPU.
    docker run --rm ${GPU[@]+"${GPU[@]}"} --network none --entrypoint python "$IMAGEM" \
      -c "import torch; print('torch', torch.__version__, '| GPU visível:', 'sim' if torch.cuda.is_available() else 'não (roda em CPU)')" >&2
    echo "pronto: imagem $IMAGEM e pesos $REPO@$REVISAO em $MODELOS; a execução já não precisa de rede" >&2
    exit 0 ;;
esac

if [ "$#" -ne 3 ]; then
  uso
  exit 2
fi

[ -f "$1" ] || { echo "base não encontrada: $1" >&2; exit 1; }
[ -d "$2" ] || { echo "pasta não encontrada: $2" >&2; exit 1; }
mkdir -p "$(dirname "$3")"
DB="$(absoluto "$1")"
TXT="$(cd "$2" && pwd)"
SAIDA="$(absoluto "$3")"
JSONS="${SAIDA%.csv}.json"
ls "$TXT"/*.txt >/dev/null 2>&1 || { echo "nenhum .txt em $TXT" >&2; exit 1; }

# 3. O que faltar da preparação (não faz nada se o --preparar já rodou)
preparar

# 4. Execução, sem rede
rm -rf "$JSONS"
mkdir -p "$JSONS"
docker run --rm ${GPU[@]+"${GPU[@]}"} --network none \
  -v "$DB:/data/acervo.db:ro" -v "$TXT:/data/in:ro" -v "$MODELOS:/models:ro" -v "$JSONS:/data/out" \
  "$IMAGEM" --input /data/in --output /data/out --db /data/acervo.db
docker run --rm --network none -v "$JSONS:/data/out:ro" -v "$(dirname "$SAIDA"):/data/csv" \
  --entrypoint python "$IMAGEM" vendor/json_to_submission.py /data/out "/data/csv/$(basename "$SAIDA")"
echo "pronto: $SAIDA ($(ls "$JSONS" | wc -l | tr -d ' ') documentos; um JSON por documento em $JSONS)" >&2
