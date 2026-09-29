# -*- coding: utf-8 -*-
"""H5: monta o job que re-roda o Qwen3-8B do benchmark gravando as respostas brutas.

    # no container: gera o script do job (cabeçalho PEP 723 + extrair_qwen.py em base64 + qwen_job.py)
    docker compose run --rm gama python -m bench.controles.offline.qwen_bruto montar
    # no container: roda o mesmo corpo com um gerador falso (sem GPU, sem upload) em 40 documentos
    docker compose run --rm gama python -m bench.controles.offline.qwen_bruto testar
    # no host: a rodada, igual à registrada (A100, --lote 32)
    hf jobs uv run --flavor a100-large --timeout 60m --secrets HF_TOKEN -d \\
        saidas/bench/controles/offline/job_qwen_bruto.py

Por que um script montado em vez de importar: o job roda longe do repositório (que é privado), e o
`bench/extrair_qwen.py` não está no dataset. Ele vai embutido byte a byte e o job confere o sha256
antes de executar; nenhuma função dele é copiada ou alterada.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import re

APP = pathlib.Path("/app")
ORIGINAL = APP / "bench" / "extrair_qwen.py"
CORPO = pathlib.Path(__file__).with_name("qwen_job.py")
SAIDA = APP / "saidas" / "bench" / "controles" / "offline"
JOB = SAIDA / "job_qwen_bruto.py"

CABECALHO = '''# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.14.0",
#   "transformers==5.17.0",
#   "accelerate",
#   "huggingface_hub",
# ]
# ///
# GERADO por bench/controles/offline/qwen_bruto.py montar. Não editar: remonte.
# torch 2.14.0 do PyPI (a rodada registrada resolveu "torch" sem pino para 2.14.0+cu130).
'''


def montar() -> pathlib.Path:
    fonte = ORIGINAL.read_bytes()
    sha = hashlib.sha256(fonte).hexdigest()
    corpo = CORPO.read_text(encoding="utf-8")
    corpo = re.sub(r"^from __future__ import annotations\n", "", corpo, flags=re.M)
    SAIDA.mkdir(parents=True, exist_ok=True)
    JOB.write_text(
        CABECALHO + "from __future__ import annotations\n\n"
        + f'ORIGINAL_B64 = "{base64.b64encode(fonte).decode()}"\nORIGINAL_SHA256 = "{sha}"\n\n'
        + corpo
        + '\n\nif __name__ == "__main__":\n    raise SystemExit(principal(ORIGINAL_B64, ORIGINAL_SHA256))\n',
        encoding="utf-8")
    print("->", JOB, "extrair_qwen.py sha256", sha, flush=True)
    return JOB


class GeradorFalso:
    """Devolve, para cada texto, uma resposta que exercita os caminhos do alinhamento: um trecho que
    alinha, uma repetição, um tipo inválido e um trecho que não existe."""

    def __call__(self, textos: list[str]) -> list[str]:
        out = []
        for t in textos:
            m = re.search(r"art\.\s*\d+", t) or re.search(r"\S+\s+\S+", t)
            trecho = m.group(0) if m else "x"
            out.append(json.dumps({"citacoes": [{"trecho": trecho, "tipo": "LEI"}, {"trecho": trecho, "tipo": "LEI"},
                                                 {"trecho": trecho, "tipo": "NORMA"},
                                                 {"trecho": "REsp nº 1.234.567/SP", "tipo": "JURIS"}]},
                                  ensure_ascii=False))
        return out


def testar() -> int:
    import torch

    from . import qwen_job
    torch.cuda.get_device_name = lambda i=0: "teste local (sem GPU)"      # main() grava no meta
    fonte = ORIGINAL.read_bytes()
    sha = hashlib.sha256(fonte).hexdigest()
    destino = SAIDA / "teste"
    destino.mkdir(parents=True, exist_ok=True)
    cod = qwen_job.principal(base64.b64encode(fonte).decode(), sha,
                             ["--entrada", str(APP / "saidas" / "bench" / "entrada_remota.jsonl"), "--limite", "40",
                              "--saida-dir", str(destino), "--sem-upload"], gerador_falso=GeradorFalso)
    linhas = [json.loads(x) for x in (destino / "qwen_bruto.jsonl").read_text(encoding="utf-8").splitlines()]
    spans = [json.loads(x) for x in (destino / "qwen_rerun.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(linhas) == 41 and len(spans) == 41, (len(linhas), len(spans))
    for b, s in zip(linhas[1:], spans[1:]):
        assert (b["conjunto"], b["id"]) == (s["conjunto"], s["id"])
        assert b["spans_alinhados"] == s["spans"] and b["nao_alinhados"] == s["nao_alinhados"]
    print("teste ok:", cod, "docs", len(linhas) - 1, "não alinhados", sum(x["nao_alinhados"] for x in linhas[1:]))
    return cod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=["montar", "testar"])
    a = ap.parse_args()
    if a.acao == "montar":
        montar()
        return 0
    return testar()


if __name__ == "__main__":
    raise SystemExit(main())
