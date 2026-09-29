# -*- coding: utf-8 -*-
"""Corpo do job H5: roda `bench/extrair_qwen.py` INTACTO e grava a resposta bruta de cada documento.

O `bench/extrair_qwen.py` do benchmark guarda só os spans alinhados e a contagem de trechos não
alinhados; a resposta do modelo se perde. Este corpo não copia nem muda nenhuma função dele: o
arquivo original vai embutido em base64 (sha256 conferido antes de executar), vira o módulo
`extrair_qwen` e o `main()` dele roda com os mesmos argumentos da rodada registrada
(job 6ab3cf4f51992417dfcd789e: `--lote 32`, A100). O invólucro troca só dois nomes globais do
módulo, `ler_json` e `alinhar`, por versões que chamam as originais e anotam, na ordem, a resposta
bruta e o que o alinhamento fez com ela.

O script do job é montado por `bench/controles/offline/qwen_bruto.py montar` (cabeçalho PEP 723 +
constantes + este arquivo). Localmente, `qwen_bruto.py testar` roda este corpo com um gerador falso.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import sys
import time
import types

REPO = "vinimlo/gama-goldenset"
REVISAO_DADOS = "2fff5f670e13f77f339937cfe3da52ed0af9572d"
DESTINO = "bench/saida/controles/offline"


def _sha(b: bytes | str) -> str:
    return hashlib.sha256(b.encode("utf-8") if isinstance(b, str) else b).hexdigest()


def carregar_original(b64: str, sha: str) -> types.ModuleType:
    fonte = base64.b64decode(b64)
    if _sha(fonte) != sha:
        raise SystemExit(f"extrair_qwen.py embutido tem sha256 {_sha(fonte)}, esperado {sha}")
    mod = types.ModuleType("extrair_qwen")
    mod.__file__ = "extrair_qwen.py"
    sys.modules["extrair_qwen"] = mod
    exec(compile(fonte, "extrair_qwen.py", "exec"), mod.__dict__)
    return mod


def instrumentar(mod: types.ModuleType) -> list:
    """Troca `ler_json` e `alinhar` do módulo por versões que registram e delegam às originais.
    `main()` resolve os dois nomes no dicionário do módulo a cada chamada, então usa as novas."""
    registro: list[dict] = []
    ler_json, alinhar = mod.ler_json, mod.alinhar

    def ler_json_registrado(saida: str):
        r = ler_json(saida)
        registro.append({"bruta": saida, "json_ok": r is not None})
        return r

    def alinhar_registrado(texto: str, citacoes: list):
        spans, perdidos = alinhar(texto, citacoes)
        registro[-1].update({"texto_sha": _sha(texto), "citacoes": citacoes, "spans_alinhados": spans,
                             "nao_alinhados": perdidos})
        return spans, perdidos

    mod.ler_json, mod.alinhar = ler_json_registrado, alinhar_registrado
    return registro


def gravar_bruto(mod, docs: list, registro: list, destino: pathlib.Path, meta: dict, tok=None) -> dict:
    """Reordena o registro (ordem de processamento do main) para a ordem da entrada e grava."""
    ordem = sorted(range(len(docs)), key=lambda k: len(docs[k]["texto"]))   # a mesma do main()
    if len(registro) != len(docs):
        raise SystemExit(f"registro com {len(registro)} respostas para {len(docs)} documentos")
    por_doc = {}
    for pos, k in enumerate(ordem):
        r = registro[pos]
        if r["texto_sha"] != _sha(docs[k]["texto"]):
            raise SystemExit(f"registro fora de ordem na posição {pos}")
        por_doc[k] = r
    tokens_max = 0
    with destino.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for k, d in enumerate(docs):
            r = por_doc[k]
            linha = {"conjunto": d["conjunto"], "id": d["id"], "ordem_processamento": ordem.index(k),
                     "json_ok": r["json_ok"], "nao_alinhados": r["nao_alinhados"],
                     "citacoes": r["citacoes"], "spans_alinhados": r["spans_alinhados"], "bruta": r["bruta"]}
            if tok is not None:
                linha["tokens_saida_reencode"] = len(tok(r["bruta"], add_special_tokens=False)["input_ids"])
                tokens_max = max(tokens_max, linha["tokens_saida_reencode"])
            fh.write(json.dumps(linha, ensure_ascii=False) + "\n")
    return {"documentos": len(docs), "nao_alinhados": sum(r["nao_alinhados"] for r in registro),
            "json_invalidos": sum(not r["json_ok"] for r in registro), "tokens_saida_max": tokens_max}


def principal(b64: str, sha_original: str, argv=None, gerador_falso=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", help="JSONL local (teste); padrão: bench/entrada_remota.jsonl do dataset")
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--limite", type=int)
    ap.add_argument("--saida-dir", default="/tmp")
    ap.add_argument("--sem-upload", action="store_true")
    a = ap.parse_args(argv)

    mod = carregar_original(b64, sha_original)
    registro = instrumentar(mod)
    if gerador_falso is not None:                        # teste local: sem GPU
        mod.GeradorHF = gerador_falso

    if a.entrada:
        entrada = pathlib.Path(a.entrada)
    else:
        from huggingface_hub import hf_hub_download
        entrada = pathlib.Path(hf_hub_download(REPO, "bench/entrada_remota.jsonl", repo_type="dataset",
                                               revision=REVISAO_DADOS))
    docs = [json.loads(x) for x in entrada.read_text(encoding="utf-8").splitlines()][: a.limite]
    saida_dir = pathlib.Path(a.saida_dir)
    spans_arq, bruto_arq = saida_dir / "qwen_rerun.jsonl", saida_dir / "qwen_bruto.jsonl"

    argv_original = ["extrair_qwen.py", "--entrada", str(entrada), "--saida", str(spans_arq), "--lote", str(a.lote)]
    if a.limite:
        argv_original += ["--limite", str(a.limite)]
    print("ORIGINAL", sha_original, "argv", argv_original[1:], flush=True)
    t0 = time.perf_counter()
    antes, sys.argv = sys.argv, argv_original
    try:
        cod = mod.main()
    finally:
        sys.argv = antes
    if cod:
        return cod

    meta = {"experimento": "offline/H5", "extrator": "qwen", "modelo": f"{mod.MODELO}@{mod.REVISAO[:7]}",
            "extrair_qwen_sha256": sha_original, "argv_original": argv_original[1:],
            "entrada": f"{REPO}@{REVISAO_DADOS}:bench/entrada_remota.jsonl" if not a.entrada else str(entrada),
            "entrada_sha256": _sha(entrada.read_bytes()), "segundos_total_main": round(time.perf_counter() - t0, 1)}
    tok = None
    if gerador_falso is None:
        import accelerate
        import torch
        import transformers
        from transformers import AutoTokenizer
        meta.update({"torch": torch.__version__, "transformers": transformers.__version__,
                     "accelerate": accelerate.__version__, "dispositivo": torch.cuda.get_device_name(0)})
        tok = AutoTokenizer.from_pretrained(mod.MODELO, revision=mod.REVISAO)
    resumo = gravar_bruto(mod, docs, registro, bruto_arq, meta, tok)
    print("RESUMO", json.dumps(resumo), flush=True)

    if not a.sem_upload:
        from huggingface_hub import HfApi
        api = HfApi()
        for arq in (spans_arq, bruto_arq):
            info = api.upload_file(path_or_fileobj=str(arq), repo_id=REPO, repo_type="dataset",
                                   path_in_repo=f"{DESTINO}/{arq.name}",
                                   commit_message=f"controles offline (H5): {arq.name} do Qwen3-8B, respostas brutas")
            print("REVISAO", arq.name, info.oid, flush=True)
    return 0
