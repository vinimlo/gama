# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.14.0",
#   "transformers==5.17.0",
#   "accelerate",
#   "huggingface_hub",
#   "numpy",
# ]
# ///
# -*- coding: utf-8 -*-
"""Fine-tuning do extrator (classificação de tokens BIO: JURIS, LEI, VAGA).

Roda em HF Jobs (GPU) ou local. Lê um repositório de DADOS no Hub com:
    codigo/bio.py                 o MESMO módulo de rótulos da inferência
    goldenset/txt/*.txt           documentos
    goldenset/goldenset_offsets.csv
    goldenset/meta.jsonl          split treino | estresse

    hf jobs uv run --flavor a10g-large --secrets HF_TOKEN treino/treinar.py \\
        --dados vinimlo/gama-goldenset --revisao <sha> \\
        --modelo-base jhu-clsp/mmBERT-base --saida vinimlo/gama

Publica os pesos num repositório de modelo (privado até a entrega) e imprime a
revisão — é ela que entra no MODELO.md da submissão. Semente fixa.
"""
from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import json
import os
import pathlib
import random
import time

import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments,
                          set_seed)

ROT = {"incompleta": "VAGA"}


def _bio(pasta: pathlib.Path):
    spec = importlib.util.spec_from_file_location("bio", pasta / "codigo" / "bio.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def carregar(pasta: pathlib.Path) -> tuple[dict, dict]:
    spans = collections.defaultdict(list)
    for r in csv.DictReader(open(pasta / "goldenset_offsets.csv", encoding="utf-8-sig")):
        tipo = ROT.get(r["classificacao"]) or ("LEI" if r["tipo"] == "lei" else "JURIS")
        spans[r["documento_id"]].append((int(r["inicio"]), int(r["fim"]), tipo))
    split = {}
    meta = pasta / "meta.jsonl"
    if meta.exists():
        for linha in meta.read_text(encoding="utf-8").splitlines():
            m = json.loads(linha)
            split[m["documento_id"]] = m.get("split", "treino")
    docs = {}
    for arq in sorted((pasta / "txt").glob("*.txt")):
        # newline="": mesmo texto que a inferência lê (ler_texto) — read_text() trocaria
        # \r\n por \n e deslocaria os rótulos (revisão independente, rodada 2, achado 3)
        with open(arq, encoding="utf-8", newline="") as fh:
            docs[arq.stem] = (fh.read(), spans.get(arq.stem, []))
    return docs, split


def _deslocamento(com_especiais: list, corpo: list) -> int:
    """Onde a janela começa depois dos tokens especiais — vale para qualquer tokenizador
    ([CLS] no BERT, BOS no Gemma/mmBERT, nada em outros)."""
    n = len(corpo)
    for k in range(len(com_especiais) - n + 1):
        if com_especiais[k:k + n] == corpo:
            return k
    raise ValueError("janela não encontrada após tokens especiais")


def especiais(tok) -> tuple[list, list]:
    """(prefixo, sufixo) de tokens especiais, descobertos empiricamente — a API de
    montar especiais mudou entre as versões 4 e 5 do transformers."""
    com = tok("a")["input_ids"]
    sem = tok("a", add_special_tokens=False)["input_ids"]
    k = _deslocamento(com, sem)
    return com[:k], com[k + len(sem):]


def exemplos(tok, docs: dict, bio, max_len: int) -> list[dict]:
    """Documento -> janelas de até max_len tokens (com [CLS]/[SEP] do modelo)."""
    out = []
    pre_esp, suf_esp = especiais(tok)
    corpo = max_len - len(pre_esp) - len(suf_esp)
    for doc, (texto, spans) in docs.items():
        enc = tok(texto, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
        rot = bio.rotular(offs, spans)
        for a, b in bio.janelas(len(ids), corpo, corpo // 2):
            wi = pre_esp + ids[a:b] + suf_esp                # especiais: rótulo IGNORAR
            wl = [bio.IGNORAR] * len(pre_esp) + rot[a:b] + [bio.IGNORAR] * len(suf_esp)
            out.append({"input_ids": wi, "attention_mask": [1] * len(wi), "labels": wl})
    return out


def prever(modelo, tok, texto: str, bio, max_len: int, dispositivo) -> list:
    enc = tok(texto, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
    pre_esp, suf_esp = especiais(tok)
    corpo = max_len - len(pre_esp) - len(suf_esp)
    cob = bio.janelas(len(ids), corpo, corpo // 2)
    logits_por_janela = []
    with torch.no_grad():
        for a, b in cob:
            wi = pre_esp + ids[a:b] + suf_esp
            t = torch.tensor([wi], device=dispositivo)
            lg = modelo(input_ids=t, attention_mask=torch.ones_like(t)).logits[0]
            pre = len(pre_esp)
            logits_por_janela.append(lg[pre:pre + (b - a)].float().cpu().numpy())
    rot = []
    for i in range(len(ids)):
        k = bio.janela_dona(i, cob)
        a, _ = cob[k]
        rot.append(int(np.argmax(logits_por_janela[k][i - a])))
    return bio.decodificar(offs, rot, texto)


def f1_spans(docs: dict, predicoes: dict) -> dict:
    """F1 por tipo com casamento exato e com IoU >= 0,5 (critério da métrica oficial)."""
    res = {}
    for criterio in ("exato", "iou50"):
        tp = collections.Counter(); fp = collections.Counter(); fn = collections.Counter()
        for doc, (_, gold) in docs.items():
            pred = list(predicoes.get(doc, []))
            usados = set()
            for a, b, t in gold:
                achou = None
                for k, (pa, pb, pt) in enumerate(pred):
                    if k in usados or pt != t:
                        continue
                    if criterio == "exato":
                        ok = (pa, pb) == (a, b)
                    else:
                        inter = max(0, min(b, pb) - max(a, pa))
                        ok = inter / (max(b, pb) - min(a, pa)) >= 0.5
                    if ok:
                        achou = k
                        break
                if achou is None:
                    fn[t] += 1
                else:
                    usados.add(achou)
                    tp[t] += 1
            for k, (_, _, pt) in enumerate(pred):
                if k not in usados:
                    fp[pt] += 1
        por_tipo = {}
        for t in ("JURIS", "LEI", "VAGA"):
            p = tp[t] / max(1, tp[t] + fp[t]); r = tp[t] / max(1, tp[t] + fn[t])
            por_tipo[t] = round(2 * p * r / max(1e-9, p + r), 4)
        res[criterio] = por_tipo
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", required=True, help="repo de dados no Hub (dataset) ou pasta local")
    ap.add_argument("--revisao", default=None)
    ap.add_argument("--subpasta", default="goldenset")
    ap.add_argument("--modelo-base", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--saida", required=True, help="repo de modelo no Hub (ou pasta local)")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--epocas", type=float, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--semente", type=int, default=13)
    a = ap.parse_args()

    set_seed(a.semente)
    random.seed(a.semente)
    t0 = time.time()
    if pathlib.Path(a.dados).exists():
        raiz = pathlib.Path(a.dados)
    else:
        raiz = pathlib.Path(snapshot_download(a.dados, repo_type="dataset", revision=a.revisao))
    bio = _bio(raiz)
    docs, split = carregar(raiz / a.subpasta)
    treino = {d: v for d, v in docs.items() if split.get(d, "treino") == "treino"}
    estresse = {d: v for d, v in docs.items() if split.get(d) == "estresse"}
    print(f"docs: {len(treino)} treino, {len(estresse)} estresse", flush=True)

    tok = AutoTokenizer.from_pretrained(a.modelo_base)
    modelo = AutoModelForTokenClassification.from_pretrained(
        a.modelo_base, num_labels=len(bio.ROTULOS),
        id2label=dict(enumerate(bio.ROTULOS)), label2id=bio.ID)
    ex_treino = exemplos(tok, treino, bio, a.max_len)
    random.shuffle(ex_treino)
    print(f"janelas de treino: {len(ex_treino)}", flush=True)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir="/tmp/saida", per_device_train_batch_size=a.lote,
        learning_rate=a.lr, num_train_epochs=a.epocas, warmup_steps=0.1, weight_decay=0.01,  # v5: float < 1 = fração
        lr_scheduler_type="linear", logging_steps=25, save_strategy="no", report_to=[],
        bf16=bf16, seed=a.semente, data_seed=a.semente, dataloader_num_workers=2,
    )
    Trainer(model=modelo, args=args, train_dataset=ex_treino,
            data_collator=DataCollatorForTokenClassification(tok)).train()

    modelo.eval()
    disp = modelo.device
    pred = {d: prever(modelo, tok, texto, bio, a.max_len, disp) for d, (texto, _) in estresse.items()}
    metricas = {"estresse": f1_spans(estresse, pred), "n_estresse": len(estresse),
                "n_treino": len(treino), "janelas": len(ex_treino), "args": vars(a),
                "minutos": round((time.time() - t0) / 60, 1)}
    print(json.dumps(metricas, ensure_ascii=False), flush=True)

    destino = pathlib.Path("/tmp/modelo")
    modelo.save_pretrained(destino)
    tok.save_pretrained(destino)
    (destino / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=1))
    (destino / "bio.py").write_text((raiz / "codigo" / "bio.py").read_text())
    if a.saida.startswith(("/", ".")):              # caminho local, não repo do Hub
        import shutil
        shutil.copytree(destino, a.saida, dirs_exist_ok=True)
        print("salvo em", a.saida)
        return 0
    api = HfApi()
    api.create_repo(a.saida, private=True, exist_ok=True)
    info = api.upload_folder(folder_path=str(destino), repo_id=a.saida,
                             commit_message=f"treino {a.modelo_base} semente {a.semente} "
                                            f"dados {a.dados}@{a.revisao}")
    print("REVISAO", info.oid, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
