# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "transformers==5.17.0",
#   "accelerate",
#   "huggingface_hub",
# ]
# ///
# -*- coding: utf-8 -*-
"""Extrator cru 2: Qwen3-8B em zero-shot, sem nenhum treino nosso.

O prompt descreve os três tipos e as convenções de borda do gabarito, com exemplos de
formato genéricos (nenhuma frase dos conjuntos avaliados). O modelo devolve os trechos
copiados do texto; cada trecho volta para o documento por alinhamento exato, com espaço
em branco flexível (o texto tem quebra de linha dentro das citações). Trecho que não se
alinha é descartado e contado. Decodificação gulosa, sem modo de raciocínio.

    # estresse + reais, bf16 na L4 (hardware-alvo)
    hf jobs uv run --flavor l4x1 --secrets HF_TOKEN bench/extrair_qwen.py \\
        --dados vinimlo/gama-goldenset --revisao <sha>
    # dev, local pelo Ollama (quantizado; os documentos da organização não saem da máquina)
    python -m bench.extrair_qwen --backend ollama --entrada ... --saida ...
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import time
import urllib.request

MODELO = "Qwen/Qwen3-8B"
REVISAO = "b968826d9c46dd6066d109eabc6255188de91218"
OLLAMA = "qwen3:8b"

INSTRUCAO = """Liste TODAS as citações jurídicas do documento abaixo, na ordem em que aparecem.
Copie cada trecho exatamente como está no texto, com a mesma grafia, inclusive erros de digitação.

Tipos:
- JURIS: precedente identificado por número. Ex.: "REsp nº 1.234.567/SP", "AgInt no AREsp 123.456/RJ", "processo nº 0001234-56.2020.8.26.0000", "Súmula 7 do STJ", "Tema 1.046 da repercussão geral".
- LEI: dispositivo de lei. Ex.: "art. 5º, X, da Constituição Federal", "artigo 927 do CPC", "art. 14 da Lei nº 8.078/1990".
- VAGA: julgado citado sem número, identificado por tribunal, ano e relator. Ex.: "precedente do STJ julgado em 2021 sob a relatoria de Nancy Andrighi".

Bordas:
- não inclua artigo ou preposição antes da citação ("o", "a", "no", "do");
- não inclua a pontuação depois da citação;
- inclua a UF no fim do número do processo quando houver ("/SP");
- em dispositivo de lei, inclua o nome do diploma;
- em referência vaga, termine no nome do relator.
Não liste referência genérica sem fonte (ex.: "a jurisprudência pacífica dos tribunais superiores").

Responda só com JSON: {"citacoes": [{"trecho": "...", "tipo": "JURIS"}]}

Documento:
"""
TIPOS = {"JURIS", "LEI", "VAGA"}


def mensagens(texto: str) -> list[dict]:
    return [{"role": "user", "content": INSTRUCAO + texto}]


def ler_json(saida: str) -> list | None:
    s = saida.strip()
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b < a:
        return None
    try:
        return json.loads(s[a:b + 1]).get("citacoes") or []
    except (json.JSONDecodeError, AttributeError):
        return None


def alinhar(texto: str, citacoes: list) -> tuple[list, int]:
    """Trechos -> spans em codepoints. Busca a próxima ocorrência livre a partir da anterior."""
    spans, perdidos, cursor = [], 0, 0
    for c in citacoes:
        trecho, tipo = str(c.get("trecho", "")), str(c.get("tipo", "")).upper()
        partes = trecho.split()
        if tipo not in TIPOS or not partes:
            perdidos += 1
            continue
        rx = re.compile(r"\s+".join(map(re.escape, partes)))
        achou = None
        for inicio in (cursor, 0):
            for m in rx.finditer(texto, inicio):
                if all(m.end() <= s[0] or s[1] <= m.start() for s in spans):
                    achou = m
                    break
            if achou:
                break
        if achou is None:
            perdidos += 1
            continue
        spans.append([achou.start(), achou.end(), tipo])
        cursor = achou.end()
    return sorted(spans), perdidos


def gerar_ollama(textos: list[str], url: str) -> list[str]:
    out = []
    for t in textos:
        corpo = json.dumps({"model": OLLAMA, "messages": mensagens(t), "stream": False, "think": False,
                            "format": "json", "options": {"temperature": 0, "seed": 13, "num_ctx": 8192}})
        req = urllib.request.Request(f"{url}/api/chat", data=corpo.encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            out.append(json.loads(r.read())["message"]["content"])
    return out


class GeradorHF:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODELO, revision=REVISAO, padding_side="left")
        self.modelo = AutoModelForCausalLM.from_pretrained(
            MODELO, revision=REVISAO, dtype=torch.bfloat16, device_map="cuda").eval()

    def __call__(self, textos: list[str]) -> list[str]:
        prompts = [self.tok.apply_chat_template(mensagens(t), tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False) for t in textos]
        lote = self.tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with self.torch.inference_mode():
            gerado = self.modelo.generate(**lote, max_new_tokens=1536, do_sample=False)
        return self.tok.batch_decode(gerado[:, lote["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "ollama"], default="hf")
    ap.add_argument("--ollama-url", default="http://host.docker.internal:11434")
    ap.add_argument("--entrada")
    ap.add_argument("--saida")
    ap.add_argument("--dados")
    ap.add_argument("--revisao")
    ap.add_argument("--lote", type=int, default=4)
    ap.add_argument("--limite", type=int)
    a = ap.parse_args()

    if a.entrada:
        entrada = pathlib.Path(a.entrada)
    else:
        from huggingface_hub import hf_hub_download
        entrada = pathlib.Path(hf_hub_download(a.dados, "bench/entrada_remota.jsonl",
                                               repo_type="dataset", revision=a.revisao))
    saida = pathlib.Path(a.saida or "/tmp/qwen.jsonl")
    docs = [json.loads(x) for x in entrada.read_text(encoding="utf-8").splitlines()][: a.limite]

    if a.backend == "hf":
        gerar = GeradorHF()
        import torch
        import transformers
        meta = {"extrator": "qwen", "modelo": f"{MODELO}@{REVISAO[:7]}", "backend": "transformers bf16",
                "transformers": transformers.__version__, "torch": torch.__version__,
                "dispositivo": torch.cuda.get_device_name(0)}
        lote = a.lote
    else:
        meta = {"extrator": "qwen", "modelo": OLLAMA, "backend": "ollama (q4_K_M)", "dispositivo": "local"}
        lote = 1

        def gerar(ts):
            return gerar_ollama(ts, a.ollama_url)
    print(json.dumps(meta), flush=True)

    # Lotes por comprimento parecido: menos padding, mesma saída (decodificação gulosa).
    ordem = sorted(range(len(docs)), key=lambda k: len(docs[k]["texto"]))
    resultados, t_total, falhas_json, perdidos = {}, 0.0, 0, 0
    for i in range(0, len(ordem), lote):
        grupo = [docs[k] for k in ordem[i:i + lote]]
        t0 = time.perf_counter()
        brutas = gerar([d["texto"] for d in grupo])
        dt = (time.perf_counter() - t0) / len(grupo)
        t_total += dt * len(grupo)
        for d, bruta in zip(grupo, brutas):
            cits = ler_json(bruta)
            if cits is None:
                falhas_json += 1
                cits = []
            spans, p = alinhar(d["texto"], cits)
            perdidos += p
            resultados[(d["conjunto"], d["id"])] = {"conjunto": d["conjunto"], "id": d["id"], "spans": spans,
                                                    "segundos": round(dt, 4), "nao_alinhados": p}
            if a.limite:
                print("AMOSTRA", d["id"], json.dumps(bruta[:300], ensure_ascii=False),
                      [d["texto"][x:y] for x, y, _ in spans][:4], flush=True)
        if (i // lote) % 25 == 0:
            print(f"{i + len(grupo)}/{len(docs)} {t_total / (i + len(grupo)):.2f} s/doc", flush=True)

    meta.update({"falhas_json": falhas_json, "trechos_nao_alinhados": perdidos})
    with saida.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for d in docs:
            fh.write(json.dumps(resultados[(d["conjunto"], d["id"])], ensure_ascii=False) + "\n")
    print(f"FIM {len(docs)} documentos, {t_total / max(1, len(docs)):.2f} s/doc, "
          f"{falhas_json} JSON inválidos, {perdidos} trechos não alinhados", flush=True)
    if a.dados and not a.saida:
        from huggingface_hub import HfApi
        info = HfApi().upload_file(path_or_fileobj=str(saida), path_in_repo="bench/saida/qwen.jsonl",
                                   repo_id=a.dados, repo_type="dataset",
                                   commit_message="bench: spans do Qwen3-8B zero-shot (estresse + reais)")
        print("REVISAO", info.oid, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
