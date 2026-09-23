#!/usr/bin/env bash
# Baixa os pesos do extrator na REVISÃO FIXA declarada em MODELO.md.
# Uso: scripts/baixar_pesos.sh [pasta_destino]   (padrão: ./modelos)
# Depois: docker run --gpus all -v "$PWD/modelos:/models:ro" ... (ver README)
set -euo pipefail
DESTINO="${1:-modelos}"
REPO="$(grep -m1 '^repo:' MODELO.md | awk '{print $2}')"
REVISAO="$(grep -m1 '^revisao:' MODELO.md | awk '{print $2}')"
if [ -z "$REPO" ] || [ -z "$REVISAO" ]; then
  echo "MODELO.md sem 'repo:' ou 'revisao:'" >&2; exit 1
fi
if command -v hf >/dev/null 2>&1; then
  hf download "$REPO" --revision "$REVISAO" --local-dir "$DESTINO"
else
  python3 -c "from huggingface_hub import snapshot_download as s; s('$REPO', revision='$REVISAO', local_dir='$DESTINO')"
fi
echo "pesos de $REPO@$REVISAO em $DESTINO"
