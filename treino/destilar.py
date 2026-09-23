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
"""Destila o Gama (professor) num aluno menor, sem mudar o que a solução entrega.

Dois alunos:
    podado    o próprio Gama com parte das camadas (padrão global/local preservado), herdando
              embeddings, cabeça e as camadas escolhidas; fica ~1,7x mais leve em conta
    <repo>    um encoder pré-treinado menor com o mesmo tokenizador (jhu-clsp/mmBERT-small)

A guarda (confiança < 0,95) e a calibração (faixa 0,98) dependem da confiança do modelo,
não só do rótulo; por isso o aluno aprende as probabilidades do professor, não só o ouro:

    L = alfa · CE(ouro) + beta · T² · KL(professor/T ‖ aluno/T) + gama · KL(professor ‖ aluno)

O professor roda aqui mesmo, em FP32 e janelas de 2.048 tokens, como na produção. Dados:
o final_v3 (o mesmo treino do professor, com ouro) e ementas reais sem rótulo, fora de
qualquer teste (`destilacao/reais_ids.json`), onde só vale o termo do professor.
`--sem-destilacao` treina só com o ouro, como controle.

    hf jobs uv run --flavor a100-large --timeout 3h --secrets HF_TOKEN treino/destilar.py \\
        --dados vinimlo/gama-goldenset --revisao <sha> --professor vinimlo/gama \\
        --professor-rev <sha> --aluno podado --saida vinimlo/gama-aluno-a
"""
from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import json
import pathlib
import random
import sys
import time

import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, snapshot_download
from transformers import (AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments,
                          set_seed)

CAMADAS_PADRAO = "0,1,2,6,7,8,12,13,14,18,19,20,21"      # globais do professor em 0,6,12,18,21


def _modulo(caminho: pathlib.Path, nome: str):
    spec = importlib.util.spec_from_file_location(nome, caminho)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def carregar_sinteticos(pasta: pathlib.Path) -> tuple[dict, dict]:
    spans = collections.defaultdict(list)
    for r in csv.DictReader(open(pasta / "goldenset_offsets.csv", encoding="utf-8-sig")):
        tipo = "VAGA" if r["classificacao"] == "incompleta" else ("LEI" if r["tipo"] == "lei" else "JURIS")
        spans[r["documento_id"]].append((int(r["inicio"]), int(r["fim"]), tipo))
    split = {}
    for linha in (pasta / "meta.jsonl").read_text(encoding="utf-8").splitlines():
        m = json.loads(linha)
        split[m["documento_id"]] = m.get("split", "treino")
    docs = {}
    for arq in sorted((pasta / "txt").glob("*.txt")):
        with open(arq, encoding="utf-8", newline="") as fh:      # o mesmo texto que a inferência lê
            docs[arq.stem] = (fh.read(), spans.get(arq.stem, []))
    return docs, split


def especiais(tok) -> tuple[list, list]:
    com = tok("a")["input_ids"]
    sem = tok("a", add_special_tokens=False)["input_ids"]
    k = next(i for i in range(len(com)) if com[i:i + len(sem)] == sem)
    return com[:k], com[k + len(sem):]


def janelas_de(tok, bio, texto: str, spans: list | None, max_len: int) -> list[dict]:
    """Janelas como na inferência (`bio.janelas`, passo de meia janela). Sem ouro, os rótulos
    ficam todos ignorados e só o termo do professor conta."""
    pre, suf = especiais(tok)
    corpo = max_len - len(pre) - len(suf)
    enc = tok(texto, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
    rot = bio.rotular(offs, spans) if spans is not None else [bio.IGNORAR] * len(ids)
    out = []
    for a, b in bio.janelas(len(ids), corpo, corpo // 2) if ids else []:
        wi = pre + ids[a:b] + suf
        out.append({"input_ids": wi, "attention_mask": [1] * len(wi),
                    "labels": [bio.IGNORAR] * len(pre) + rot[a:b] + [bio.IGNORAR] * len(suf),
                    "kd_mask": [0] * len(pre) + [1] * (b - a) + [0] * len(suf)})
    return out


def juntar(pad_id: int):
    def f(lote: list[dict]) -> dict:
        n = max(len(x["input_ids"]) for x in lote)
        def pad(k, v):
            return torch.tensor([x[k] + [v] * (n - len(x[k])) for x in lote])
        return {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0),
                "labels": pad("labels", -100), "kd_mask": pad("kd_mask", 0)}
    return f


def podar(professor_dir: str, camadas: list[int]):
    """O professor com só as camadas escolhidas, na ordem. `layer_types` vai explícito na
    configuração, então cada camada mantém o tipo de atenção (global/local) e a RoPE que
    tinha no professor; a camada 0 (norma de atenção Identity) é mantida na posição 0."""
    modelo = AutoModelForTokenClassification.from_pretrained(professor_dir)
    tipos = list(modelo.config.layer_types)
    assert camadas[0] == 0, "a camada 0 tem norma de atenção própria e precisa abrir o aluno"
    modelo.model.layers = torch.nn.ModuleList([modelo.model.layers[i] for i in camadas])
    modelo.config.num_hidden_layers = len(camadas)
    modelo.config.layer_types = [tipos[i] for i in camadas]
    for j, camada in enumerate(modelo.model.layers):
        camada.layer_idx = j
        camada.attn.layer_idx = j
    return modelo


class Destilador(Trainer):
    def __init__(self, *args, professor=None, alfa=0.5, beta=1.0, gama=0.5, temperatura=2.0, **kw):
        super().__init__(*args, **kw)
        self.professor, self.alfa, self.beta, self.gama, self.T = professor, alfa, beta, gama, temperatura

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        kd = inputs.pop("kd_mask").bool()
        saida = model(**inputs)
        zs = saida.logits.float()
        c = zs.shape[-1]
        perda = zs.new_zeros(())
        if self.alfa and (labels != -100).any():
            perda = perda + self.alfa * F.cross_entropy(zs.view(-1, c), labels.view(-1), ignore_index=-100)
        if self.professor is not None and kd.any():
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                zt = self.professor(**inputs).logits.float()     # FP32, como na produção
            s, t = zs[kd], zt[kd]
            T = self.T
            kl_t = F.kl_div(F.log_softmax(s / T, -1), F.log_softmax(t / T, -1), log_target=True,
                            reduction="batchmean") * T * T
            kl_1 = F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1), log_target=True, reduction="batchmean")
            perda = perda + self.beta * kl_t + self.gama * kl_1
        return (perda, saida) if return_outputs else perda


def comparar(gama_pkg: pathlib.Path, professor_dir: str, aluno_dir: str, docs: dict) -> dict:
    """Aluno contra professor pelo extrator de produção (mesma janela, decodificação e
    confiança): documentos com spans diferentes, acertos, e a cauda de confiança."""
    sys.path.insert(0, str(gama_pkg))
    from gama.extratores.neural import ExtratorNeural
    prof, alu = ExtratorNeural(professor_dir), ExtratorNeural(aluno_dir)
    dif, confs_alu, confs_prof, tp = 0, [], [], 0
    t_prof = t_alu = 0.0
    for texto, gold in docs.values():
        t0 = time.perf_counter(); p = prof.extrair(texto); t1 = time.perf_counter()
        a = alu.extrair(texto); t2 = time.perf_counter()
        t_prof += t1 - t0; t_alu += t2 - t1
        rot = lambda s: "LEI" if s.tipo == "lei" else ("VAGA" if s.forma == "vaga" else "JURIS")
        dif += [(s.inicio, s.fim, s.forma) for s in p] != [(s.inicio, s.fim, s.forma) for s in a]
        ouro = {(x, y, t) for x, y, t in gold}
        for s in a:
            if (s.inicio, s.fim, rot(s)) in ouro:
                tp += 1
                confs_alu.append(s.confianca)
        confs_prof += [s.confianca for s in p if (s.inicio, s.fim, rot(s)) in ouro]
    n_ouro = sum(len(g) for _, g in docs.values())
    return {"docs": len(docs), "docs_com_spans_diferentes_do_professor": dif,
            "acertos_exatos_aluno": tp, "citacoes_ouro": n_ouro,
            "conf_min_acerto_aluno": round(min(confs_alu), 4) if confs_alu else None,
            "acertos_aluno_abaixo_098": sum(c < 0.98 for c in confs_alu),
            "conf_min_acerto_professor": round(min(confs_prof), 4) if confs_prof else None,
            "s_por_doc_professor": round(t_prof / len(docs), 4), "s_por_doc_aluno": round(t_alu / len(docs), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", required=True)
    ap.add_argument("--revisao", required=True)
    ap.add_argument("--subpasta", default="final_v3")
    ap.add_argument("--professor", required=True)
    ap.add_argument("--professor-rev", required=True)
    ap.add_argument("--aluno", required=True, help="'podado' ou um repo de encoder (ex.: jhu-clsp/mmBERT-small)")
    ap.add_argument("--aluno-rev", default=None)
    ap.add_argument("--camadas", default=CAMADAS_PADRAO)
    ap.add_argument("--sem-destilacao", action="store_true", help="controle: só o ouro, sem professor")
    ap.add_argument("--sem-reais", action="store_true", help="não usar as ementas reais sem rótulo")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--epocas", type=float, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--alfa", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gama", type=float, default=0.5)
    ap.add_argument("--temperatura", type=float, default=2.0)
    ap.add_argument("--semente", type=int, default=13)
    ap.add_argument("--limite", type=int, help="ensaio: poucos documentos e poucos passos")
    a = ap.parse_args()

    set_seed(a.semente)
    random.seed(a.semente)
    t0 = time.time()
    raiz = pathlib.Path(snapshot_download(a.dados, repo_type="dataset", revision=a.revisao, allow_patterns=[
        f"{a.subpasta}/**", "codigo/*", "bench/codigo/**", "reais/amostra.jsonl", "destilacao/*"]))
    bio = _modulo(raiz / "codigo" / "bio.py", "bio")
    professor_dir = snapshot_download(a.professor, revision=a.professor_rev)
    tok = AutoTokenizer.from_pretrained(professor_dir)

    docs, split = carregar_sinteticos(raiz / a.subpasta)
    treino = {d: v for d, v in docs.items() if split.get(d, "treino") == "treino"}
    reserva = {d: v for d, v in docs.items() if split.get(d) == "estresse"}
    if a.limite:
        treino = dict(list(treino.items())[: a.limite])
        reserva = dict(list(reserva.items())[: max(4, a.limite // 4)])
    ex = [w for texto, spans in treino.values() for w in janelas_de(tok, bio, texto, spans, a.max_len)]
    n_sint = len(ex)
    n_reais = 0
    if not (a.sem_destilacao or a.sem_reais):
        ids = set(json.loads((raiz / "destilacao" / "reais_ids.json").read_text()))
        reais = [json.loads(x)["texto"] for x in (raiz / "reais" / "amostra.jsonl").read_text(encoding="utf-8").splitlines()
                 if json.loads(x)["id"] in ids]
        if a.limite:
            reais = reais[: a.limite]
        for texto in reais:
            ws = janelas_de(tok, bio, texto, None, a.max_len)
            ex += ws
            n_reais += len(ws)
    random.shuffle(ex)
    print(f"janelas: {n_sint} sintéticas com ouro, {n_reais} de ementas reais sem rótulo", flush=True)

    if a.aluno == "podado":
        camadas = [int(x) for x in a.camadas.split(",")]
        aluno = podar(professor_dir, camadas)
    else:
        aluno = AutoModelForTokenClassification.from_pretrained(
            a.aluno, revision=a.aluno_rev, num_labels=len(bio.ROTULOS),
            id2label=dict(enumerate(bio.ROTULOS)), label2id=bio.ID)
        tok_aluno = AutoTokenizer.from_pretrained(a.aluno, revision=a.aluno_rev)
        amostra = next(iter(treino.values()))[0][:5000]
        assert tok_aluno(amostra)["input_ids"] == tok(amostra)["input_ids"], "tokenizadores diferentes"
    parametros = sum(p.numel() for p in aluno.parameters())
    print(f"aluno: {a.aluno}, {parametros / 1e6:.1f}M parâmetros", flush=True)

    professor = None
    if not a.sem_destilacao:
        professor = AutoModelForTokenClassification.from_pretrained(professor_dir).eval().to("cuda")
        for p in professor.parameters():
            p.requires_grad_(False)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir="/tmp/saida", per_device_train_batch_size=a.lote, learning_rate=a.lr,
        num_train_epochs=a.epocas, max_steps=20 if a.limite else -1, warmup_steps=0.1, weight_decay=0.01,
        lr_scheduler_type="linear", logging_steps=25, save_strategy="no", report_to=[], bf16=bf16,
        seed=a.semente, data_seed=a.semente, dataloader_num_workers=2, remove_unused_columns=False,
        label_names=["labels"])
    Destilador(model=aluno, args=args, train_dataset=ex, data_collator=juntar(tok.pad_token_id or 0),
               professor=professor, alfa=1.0 if a.sem_destilacao else a.alfa, beta=a.beta,
               gama=a.gama, temperatura=a.temperatura).train()
    del professor
    torch.cuda.empty_cache()

    destino = pathlib.Path("/tmp/aluno")
    aluno.eval().save_pretrained(destino)
    tok.save_pretrained(destino)
    (destino / "bio.py").write_text((raiz / "codigo" / "bio.py").read_text())
    comparacao = comparar(raiz / "bench" / "codigo", professor_dir, str(destino), reserva)
    metricas = {"reserva_final_v3": comparacao, "parametros": parametros, "janelas_sinteticas": n_sint,
                "janelas_reais_sem_rotulo": n_reais, "args": vars(a), "minutos": round((time.time() - t0) / 60, 1)}
    print(json.dumps(metricas, ensure_ascii=False), flush=True)
    (destino / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=1))
    if a.limite:
        return 0
    api = HfApi()
    api.create_repo(a.saida, private=True, exist_ok=True)
    info = api.upload_folder(folder_path=str(destino), repo_id=a.saida,
                             commit_message=f"aluno {a.aluno} destilado de {a.professor}@{a.professor_rev[:7]}")
    print("REVISAO", info.oid, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
