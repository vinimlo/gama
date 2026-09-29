# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.14.0",
#   "transformers==5.17.0",
#   "accelerate",
#   "huggingface_hub",
#   "numpy",
# ]
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cu126" }
# ///
# -*- coding: utf-8 -*-
"""Controle H2: mmBERT-base ORIGINAL com o encoder congelado; só a cabeça de classificação de tokens treina.

Pergunta: quanto a representação pré-treinada já extrai sozinha, sem adaptar o encoder? O v1.2 é a
mesma receita com o encoder treinável. Tudo o mais é o `treino/treinar.py` do v1.2: dados
(`vinimlo/gama-goldenset@31474b1`, subpasta `final_v3`, 5.410 docs de treino), rótulos BIO de
`codigo/bio.py` do dataset, janelas de 1.024 tokens com passo de meia janela, 3 épocas, lote 8,
semente 13, os mesmos `TrainingArguments` (AdamW, aquecimento de 10%, decaimento linear, weight
decay 0,01, bf16), o mesmo `Trainer` e o mesmo colador.

O que treina. `ModernBertForTokenClassification` = `model` (o encoder: embeddings + 22 camadas +
norma final) -> `head` (ModernBertPredictionHead: dense 768x768 sem viés, GELU, LayerNorm só com
peso) -> `drop` (p = 0,0) -> `classifier` (Linear 768->7 com viés). Aqui `model.*` fica com
`requires_grad=False` e treinam `head.dense.weight`, `head.norm.weight`, `classifier.weight` e
`classifier.bias` (595.975 parâmetros). O `head` NÃO nasce aleatório: o checkpoint MLM do
mmBERT-base tem `head.dense.weight` e `head.norm.weight` com os mesmos nomes e eles são carregados
(o job confere tensor a tensor contra o checkpoint). Só o `classifier` nasce aleatório (semente
13, a mesma do v1.2). Não é uma sonda estritamente linear: a cabeça é uma MLP de uma camada
oculta, inicializada da transformação do MLM, mais o classificador linear.

Taxa de aprendizado. Não assume o 5e-5 do fine-tuning completo: varre a grade `--lrs` e escolhe no
split interno de validação do `final_v3` (os 590 docs com `split = estresse` no meta.jsonl, os
mesmos em que o `treinar.py` mediu o v1.2; não são o estresse difícil de 600 docs do benchmark,
que é outro gerador). Critério, fixado antes de rodar: maior média das F1 por tipo (JURIS, LEI,
VAGA) com IoU >= 0,5 de `f1_spans`; empate em 4 casas -> maior média com borda exata; depois menor
perda de validação (entropia cruzada por token nas janelas dos 590 docs, `Trainer.evaluate`). Se
a melhor for a maior taxa já rodada, roda a próxima de `--extensao` (uma de cada vez). As ementas
reais não entram em nada disto. A melhor sobe para um repo PRIVADO `<prefixo>-<lr>`.

Por que as funções de `treino/treinar.py` estão copiadas (não importadas): o `hf jobs uv run`
envia um arquivo só, e `treinar.py` não está no dataset. `ROT`, `_bio`, `carregar`,
`_deslocamento`, `especiais`, `exemplos`, `prever` e `f1_spans` são cópias literais;
`bench/controles/sonda/conferir_copia.py` compara a AST de cada uma com a original.

    hf jobs uv run --flavor a100-large --timeout 75m --secrets HF_TOKEN \\
        bench/controles/sonda/treinar_sonda.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import pathlib
import random
import statistics
import time

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments,
                          set_seed)

ROT = {"incompleta": "VAGA"}


# ---------------------------------------------------------------- cópia literal de treino/treinar.py

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


# ---------------------------------------------------------------- sonda

def _sha(tensores) -> str:
    h = hashlib.sha256()
    for nome, t in tensores:
        h.update(nome.encode())
        h.update(t.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def impressao_encoder(modelo) -> str:
    """sha256 de todos os parâmetros e buffers do encoder (`model.*`), em ordem de nome."""
    return _sha(sorted((n, t) for n, t in modelo.state_dict().items() if n.startswith("model.")))


def pesos_checkpoint(repo: str, rev: str) -> dict:
    """state_dict do checkpoint MLM publicado (safetensors ou .bin, o que a revisão tiver)."""
    try:
        from safetensors.torch import load_file
        return load_file(hf_hub_download(repo, "model.safetensors", revision=rev))
    except Exception:  # noqa: BLE001 — a revisão fixada do mmBERT-base só tem pytorch_model.bin
        return torch.load(hf_hub_download(repo, "pytorch_model.bin", revision=rev), map_location="cpu",
                          weights_only=True)


def novo(a, bio):
    """Base original + cabeça de tokens; o classificador nasce da semente (igual em toda taxa)."""
    set_seed(a.semente)
    modelo, info = AutoModelForTokenClassification.from_pretrained(
        a.modelo_base, revision=a.revisao_base, num_labels=len(bio.ROTULOS),
        id2label=dict(enumerate(bio.ROTULOS)), label2id=bio.ID, output_loading_info=True)
    return modelo, {k: sorted(v) if isinstance(v, (list, set)) else str(v) for k, v in info.items()}


def congelar(modelo) -> dict:
    for p in modelo.model.parameters():
        p.requires_grad = False
    return {n: p.numel() for n, p in modelo.named_parameters() if p.requires_grad}


def medir_validacao(modelo, tok, bio, validacao: dict, max_len: int) -> dict:
    modelo.eval()
    pred = {d: prever(modelo, tok, texto, bio, max_len, modelo.device) for d, (texto, _) in validacao.items()}
    f1 = f1_spans(validacao, pred)
    return {"f1": f1, "media_iou50": round(statistics.mean(f1["iou50"].values()), 4),
            "media_exato": round(statistics.mean(f1["exato"].values()), 4),
            "spans_previstos": sum(len(v) for v in pred.values()),
            "spans_ouro": sum(len(g) for _, g in validacao.values())}


def chave(r: dict) -> tuple:
    return (r["validacao"]["media_iou50"], r["validacao"]["media_exato"], -round(r["perda_validacao"], 6))


def treinar_um(lr: str, a, bio, tok, ex_treino, ex_val, validacao, enc_ref: str) -> tuple[dict, dict]:
    modelo, info = novo(a, bio)
    treinaveis = congelar(modelo)
    cls0 = _sha([("classifier.weight", modelo.classifier.weight)])
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=f"/tmp/saida_{lr}", per_device_train_batch_size=a.lote, per_device_eval_batch_size=a.lote,
        learning_rate=float(lr), num_train_epochs=a.epocas, warmup_steps=0.1, weight_decay=0.01,  # v5: float < 1 = fração
        lr_scheduler_type="linear", logging_steps=25, save_strategy="no", report_to=[],
        bf16=bf16, seed=a.semente, data_seed=a.semente, dataloader_num_workers=2, max_steps=a.passos_max,
    )
    tr = Trainer(model=modelo, args=args, train_dataset=ex_treino, eval_dataset=ex_val,
                 data_collator=DataCollatorForTokenClassification(tok))
    t0 = time.time()
    saida = tr.train()
    minutos = round((time.time() - t0) / 60, 2)
    perda_val = tr.evaluate()["eval_loss"]
    val = medir_validacao(modelo, tok, bio, validacao, a.max_len)
    enc_depois = impressao_encoder(modelo)
    perdas = [x["loss"] for x in tr.state.log_history if "loss" in x]
    res = {"lr": lr, "validacao": val, "perda_validacao": perda_val, "perda_treino_media": saida.training_loss,
           "perda_treino_ultimos_logs": perdas[-4:], "passos": tr.state.global_step, "minutos_treino": minutos,
           "encoder_intacto": enc_depois == enc_ref, "encoder_sha": enc_depois,
           "classifier_inicial_sha": cls0, "treinaveis": treinaveis, "n_treinaveis": sum(treinaveis.values()),
           "bf16": bf16, "carga": info}
    print("RESULTADO " + json.dumps(res, ensure_ascii=False), flush=True)
    cabeca = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items() if not k.startswith("model.")}
    del tr, modelo
    torch.cuda.empty_cache()
    return res, cabeca


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default="vinimlo/gama-goldenset", help="repo de dados no Hub (dataset) ou pasta local")
    ap.add_argument("--revisao", default="31474b1f7db9096c4ca4f2a4eae2e9b82852d7a7")
    ap.add_argument("--subpasta", default="final_v3")
    ap.add_argument("--modelo-base", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--revisao-base", default="c5955035435e2bf121cde7f3c8863ef52ff35d82")
    ap.add_argument("--saida", default="vinimlo/gama-exp-sonda", help="prefixo do repo PRIVADO (<saida>-<lr>) ou pasta local")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--epocas", type=float, default=3)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--semente", type=int, default=13)
    ap.add_argument("--lrs", default="5e-5,1e-4,3e-4,1e-3,3e-3,1e-2", help="grade, em ordem crescente")
    ap.add_argument("--extensao", default="3e-2,1e-1", help="se a melhor for a maior já rodada, roda a próxima")
    ap.add_argument("--limite", type=int, help="ensaio: só os N primeiros docs de treino (e N/4 de validação)")
    ap.add_argument("--passos-max", type=int, default=-1, help="ensaio: corta o treino em N passos")
    a = ap.parse_args()

    set_seed(a.semente)
    random.seed(a.semente)
    t0 = time.time()
    if pathlib.Path(a.dados).exists():
        raiz = pathlib.Path(a.dados)
    else:
        raiz = pathlib.Path(snapshot_download(a.dados, repo_type="dataset", revision=a.revisao,
                                              allow_patterns=["codigo/bio.py", f"{a.subpasta}/**"]))
    bio = _bio(raiz)
    docs, split = carregar(raiz / a.subpasta)
    treino = {d: v for d, v in docs.items() if split.get(d, "treino") == "treino"}
    validacao = {d: v for d, v in docs.items() if split.get(d) == "estresse"}   # os 10% internos do final_v3
    if a.limite:
        treino = dict(list(treino.items())[: a.limite])
        validacao = dict(list(validacao.items())[: max(1, a.limite // 4)])
    print(f"docs: {len(treino)} treino, {len(validacao)} validação interna", flush=True)

    tok = AutoTokenizer.from_pretrained(a.modelo_base, revision=a.revisao_base)
    ex_treino = exemplos(tok, treino, bio, a.max_len)
    random.shuffle(ex_treino)
    ex_val = exemplos(tok, validacao, bio, a.max_len)
    ordem = hashlib.sha256(json.dumps([e["input_ids"][:8] for e in ex_treino]).encode()).hexdigest()[:12]
    print(f"janelas: {len(ex_treino)} treino (ordem {ordem}), {len(ex_val)} validação", flush=True)

    # carga: o head vem do checkpoint MLM? o encoder carregado é o do checkpoint?
    modelo, info = novo(a, bio)
    ck = pesos_checkpoint(a.modelo_base, a.revisao_base)
    head_do_mlm = {k: bool(torch.equal(ck[k], v.detach().cpu())) if k in ck else None
                   for k, v in modelo.state_dict().items() if not k.startswith("model.")}
    enc_ref = impressao_encoder(modelo)
    enc_ck = _sha(sorted((k, v) for k, v in ck.items() if k.startswith("model.")))
    del ck
    if torch.cuda.is_available():
        modelo.to("cuda")
    base_val = medir_validacao(modelo, tok, bio, validacao, a.max_len)   # cabeça aleatória, passo 0
    carga = {"info": info, "cabeca_igual_ao_checkpoint_mlm": head_do_mlm, "encoder_sha": enc_ref,
             "encoder_sha_checkpoint": enc_ck, "encoder_igual_ao_checkpoint": enc_ref == enc_ck,
             "classifier_inicial_sha": _sha([("classifier.weight", modelo.classifier.weight)]),
             "parametros_total": sum(p.numel() for p in modelo.parameters()),
             "validacao_cabeca_aleatoria": base_val}
    print("CARGA " + json.dumps(carga, ensure_ascii=False), flush=True)
    del modelo
    torch.cuda.empty_cache()

    lrs = [x for x in a.lrs.split(",") if x]
    extensao = [x for x in a.extensao.split(",") if x]
    resultados, cabecas = {}, {}
    fila = list(lrs)
    while fila:
        lr = fila.pop(0)
        resultados[lr], cabecas[lr] = treinar_um(lr, a, bio, tok, ex_treino, ex_val, validacao, enc_ref)
        melhor = max(resultados, key=lambda k: chave(resultados[k]))
        for k in list(cabecas):                       # guarda só a melhor até aqui
            if k != melhor:
                del cabecas[k]
        if not fila and extensao and float(melhor) == max(map(float, resultados)):
            fila.append(extensao.pop(0))              # melhor na borda de cima: estende a grade
    melhor = max(resultados, key=lambda k: chave(resultados[k]))
    grade = {lr: {"media_iou50": r["validacao"]["media_iou50"], "media_exato": r["validacao"]["media_exato"],
                  "perda_validacao": round(r["perda_validacao"], 6), "f1": r["validacao"]["f1"]}
             for lr, r in resultados.items()}
    print("GRADE " + json.dumps({"escolhida": melhor, "grade": grade}, ensure_ascii=False), flush=True)

    modelo, _ = novo(a, bio)
    faltam, sobram = modelo.load_state_dict(cabecas[melhor], strict=False)
    assert not sobram and all(k.startswith("model.") for k in faltam), (faltam, sobram)
    assert impressao_encoder(modelo) == enc_ref
    metricas = {"escolhida": melhor, "criterio": "maior média das F1 por tipo com IoU >= 0,5 nos 590 docs de "
                "validação interna do final_v3; empate: média exata, depois menor perda de validação",
                "grade": grade, "resultados": resultados, "carga": carga, "args": vars(a),
                "n_treino": len(treino), "n_validacao": len(validacao), "janelas": len(ex_treino),
                "ordem_janelas": ordem, "minutos": round((time.time() - t0) / 60, 1)}
    destino = pathlib.Path("/tmp/modelo")
    modelo.save_pretrained(destino)
    tok.save_pretrained(destino)
    (destino / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=1))
    (destino / "bio.py").write_text((raiz / "codigo" / "bio.py").read_text())
    if a.saida.startswith(("/", ".")):              # caminho local, não repo do Hub
        import shutil
        shutil.copytree(destino, a.saida, dirs_exist_ok=True)
        print("salvo em", a.saida, flush=True)
        return 0
    repo = f"{a.saida}-{melhor}"
    assert repo.startswith("vinimlo/gama-exp-sonda-"), repo    # nunca vinimlo/gama
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    info = api.upload_folder(folder_path=str(destino), repo_id=repo,
                             commit_message=f"sonda H2: {a.modelo_base}@{a.revisao_base[:7]} congelado, cabeça "
                                            f"lr {melhor}, dados {a.dados}@{a.revisao[:7]}/{a.subpasta}")
    print("REPO", repo, flush=True)
    print("REVISAO", info.oid, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
