# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gliner==0.2.29",
#   "torch==2.14.0",
#   "transformers>=4.51,<5",
#   "accelerate",
#   "sentencepiece",
#   "protobuf",
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
"""Controle H4: o GLiNER do benchmark zero-shot, agora fine-tunado no mesmo final_v3 do Gama.

Pergunta: um GLiNER treinado na mesma tarefa não ganharia do mmBERT? Parte do mesmo
`urchade/gliner_multi-v2.1@443d26d` do benchmark, com os MESMOS quatro nomes de rótulo e o
MESMO janelamento de inferência (200 palavras, passo 150, sobreposição resolvida pelo maior
score) de `bench/extrair_gliner.py`.

Dados de treino: `final_v3` do dataset `vinimlo/gama-goldenset@31474b1` (o do v1.2), split
`treino` (5.410 documentos sintéticos). Época e limiar de decisão se escolhem no split
`estresse` do meta.jsonl (590 documentos, os ~10% de reserva do final_v3). Nenhuma ementa real
entra no treino nem na escolha; os conjuntos de avaliação (estresse 600, 305, 172) só são
extraídos depois da escolha, para a pontuação local em `bench.controles.avaliar`.

Conversão (documento -> exemplo do GLiNER): janelas de até 200 palavras (\\S+) com passo 150,
como na inferência, mas com a borda recuada/adiantada para nunca cortar uma citação do ouro;
cada janela vira `tokenized_text` pelo mesmo divisor de palavras do GLiNER
(`\\w+(?:[-_]\\w+)*|\\S`, conferido no job contra `model.data_processor.words_splitter`) e cada
citação vira [palavra_inicial, palavra_final, rótulo] (fim inclusivo). `ner_labels` fixa os
quatro rótulos em toda janela (o recomendado pela documentação para conjunto fixo de rótulos),
na mesma ordem da inferência; janela sem citação entra como exemplo negativo.

Rótulo do ouro -> nome do zero-shot: VAGA (classificação incompleta) -> "referência a julgado
sem número"; LEI -> "artigo de lei"; JURIS -> "súmula ou tema de tribunal" se `gama.formas.forma`
do trecho der súmula/tema, senão "precedente judicial com número".

    # estatísticas da conversão, no container (sem GLiNER, sem torch)
    docker compose run --rm gama python bench/controles/gliner_ft/treinar_gliner.py \\
        estatisticas --pasta corpus/goldenset/v3
    # o que rodou em 29/09: treino + escolha + publicação na A10G (job 6abc04f5, 79 min) ...
    hf jobs uv run --flavor a10g-large --timeout 3h --secrets HF_TOKEN \\
        bench/controles/gliner_ft/treinar_gliner.py treinar --saida vinimlo/gama-exp-gliner-ft --sem-extrair
    # ... e a extração dos conjuntos de avaliação na L4, o hardware do benchmark (job 6abc17d3)
    hf jobs uv run --flavor l4x1 --timeout 1h --secrets HF_TOKEN bench/controles/gliner_ft/treinar_gliner.py \\
        extrair --modelo vinimlo/gama-exp-gliner-ft@12f4e1873b6effd13cc8631b2a255c8b540d2d2a --limiar 0.5
    # (sem --sem-extrair, o `treinar` também extrai no fim, no mesmo job)

Copiado de `bench/extrair_gliner.py` (o job do HF só recebe este arquivo): MODELO, REVISAO,
ROTULOS, PALAVRAS, PASSO, `janelas` e a fusão das janelas de `extrair` (esta com o score
guardado e as janelas de um documento num lote só). `bench/controles/gliner_ft/conferir.py`
confere no container que as cópias batem com o original.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import random
import re
import sys
import time

# ------------------------------------------------------------------ copiado de bench/extrair_gliner.py
MODELO = "urchade/gliner_multi-v2.1"
REVISAO = "443d26d654e0324125a96bebd8e796c14ff2efe6"
ROTULOS = {
    "precedente judicial com número": "JURIS",
    "súmula ou tema de tribunal": "JURIS",
    "artigo de lei": "LEI",
    "referência a julgado sem número": "VAGA",
}
LIMIAR_ZERO_SHOT = 0.5
PALAVRAS, PASSO = 200, 150      # 200 palavras ficam abaixo dos 384 tokens do GLiNER


def janelas(texto: str) -> list[tuple[int, int]]:
    palavras = [m.span() for m in re.finditer(r"\S+", texto)]
    out, i = [], 0
    while palavras:
        bloco = palavras[i:i + PALAVRAS]
        out.append((bloco[0][0], bloco[-1][1]))
        if i + PALAVRAS >= len(palavras):
            break
        i += PASSO
    return out


def fundir(achados: list) -> list[list]:
    """A fusão de `extrair`: na sobreposição entre janelas fica o de maior score."""
    ficam: list = []
    for s in sorted(achados, key=lambda x: -x[3]):
        if all(s[1] <= f[0] or f[1] <= s[0] for f in ficam):
            ficam.append(s)
    return [[a, b, r, round(sc, 4)] for a, b, r, sc in sorted(ficam)]
# ------------------------------------------------------------------ fim do trecho copiado

DADOS = "vinimlo/gama-goldenset"
REV_TREINO = "31474b1f7db9096c4ca4f2a4eae2e9b82852d7a7"   # final_v3 do v1.2 (MODELO.md)
REV_BENCH = "2fff5f670e13f77f339937cfe3da52ed0af9572d"    # entradas do benchmark + bench/codigo
EXPERIMENTO = "gliner_ft"
PALAVRA = re.compile(r"\w+(?:[-_]\w+)*|\S")               # WhitespaceTokenSplitter do GLiNER
ROT_OURO = {"incompleta": "VAGA"}                          # como treino/treinar.py
NOME = {"LEI": "artigo de lei", "VAGA": "referência a julgado sem número"}
PISO = 0.10                                                # limiar mínimo extraído (a grade filtra acima)
GRADE = [round(0.10 + 0.05 * i, 2) for i in range(18)]     # 0,10 a 0,95
SEMENTE = 13


# ------------------------------------------------------------------ dados

def carregar(pasta: pathlib.Path) -> tuple[dict, dict]:
    """final_v3 -> ({doc: (texto, [(ini, fim, JURIS|LEI|VAGA)])}, {doc: split}), como treinar.py."""
    spans = collections.defaultdict(list)
    for r in csv.DictReader(open(pasta / "goldenset_offsets.csv", encoding="utf-8-sig")):
        tipo = ROT_OURO.get(r["classificacao"]) or ("LEI" if r["tipo"] == "lei" else "JURIS")
        spans[r["documento_id"]].append((int(r["inicio"]), int(r["fim"]), tipo))
    split = {}
    for linha in (pasta / "meta.jsonl").read_text(encoding="utf-8").splitlines():
        m = json.loads(linha)
        split[m["documento_id"]] = m.get("split", "treino")
    docs = {}
    for arq in sorted((pasta / "txt").glob("*.txt")):
        with open(arq, encoding="utf-8", newline="") as fh:     # mesmo texto da inferência
            docs[arq.stem] = (fh.read(), sorted(spans.get(arq.stem, [])))
    return docs, split


def impressao_textos(docs: dict) -> str:
    h = hashlib.sha256()
    for d in sorted(docs):
        h.update(d.encode())
        h.update(docs[d][0].encode("utf-8"))
    return h.hexdigest()[:16]


def nome_rotulo(texto: str, a: int, b: int, tipo: str, forma) -> str:
    if tipo in NOME:
        return NOME[tipo]
    return "súmula ou tema de tribunal" if forma(texto[a:b]) in ("sumula", "tema") else \
        "precedente judicial com número"


def palavras(texto: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in PALAVRA.finditer(texto)]


def janelas_treino(texto: str, ouro: list) -> list[tuple[int, int]]:
    """As janelas da inferência (200/150), com a borda movida para não cortar citação do ouro:
    a que cruza o fim sai da janela (o fim recua para antes dela); a que cruza o início entra
    inteira (o início recua até ela). Com passo 150 e citações curtas, toda citação cabe
    inteira em alguma janela."""
    out = []
    for a, b in janelas(texto):
        for ga, gb in ouro:                  # o ouro não se sobrepõe: no máximo uma cruza cada borda
            if ga < a < gb:
                a = ga
            if ga < b < gb:
                b = len(texto[:ga].rstrip())
        if b > a:
            out.append((a, b))
    return out


def exemplos(docs: dict, forma, max_width: int | None = None) -> tuple[list, dict]:
    """Documentos -> exemplos do GLiNER + estatísticas da conversão."""
    ex, st = [], collections.Counter()
    larguras = collections.Counter()
    comprimentos = []
    vistos = set()
    for doc, (texto, ouro) in docs.items():
        pares = [(a, b) for a, b, _ in ouro]
        st["citacoes"] += len(ouro)
        st["sobrepostas"] += sum(1 for (a1, b1), (a2, b2) in zip(pares, pares[1:]) if a2 < b1)
        for wa, wb in janelas_treino(texto, pares):
            ws = palavras(texto[wa:wb])
            comprimentos.append(len(ws))
            ner = []
            for a, b, tipo in ouro:
                if not (wa <= a and b <= wb):
                    if a < wb and wa < b:
                        st["cortadas_pela_janela"] += 1
                    continue
                ra, rb = a - wa, b - wa
                i = next((k for k, (_, s, e) in enumerate(ws) if e > ra), None)
                j = next((k for k in range(len(ws) - 1, -1, -1) if ws[k][1] < rb), None)
                if i is None or j is None or j < i:
                    st["sem_palavra"] += 1
                    continue
                if ws[i][1] != ra or ws[j][2] != rb:
                    st["borda_desalinhada"] += 1
                w = j - i + 1
                larguras[w] += 1
                if max_width and w > max_width:
                    st["mais_largas_que_max_width"] += 1
                    continue
                ner.append([i, j, nome_rotulo(texto, a, b, tipo, forma)])
                vistos.add((doc, a, b))
            st["janelas"] += 1
            st["janelas_sem_citacao"] += not ner
            ex.append({"tokenized_text": [w for w, _, _ in ws], "ner": ner, "ner_labels": list(ROTULOS),
                       "_doc": doc, "_janela": [wa, wb]})
    st["citacoes_em_alguma_janela"] = len(vistos)
    st["citacoes_fora_de_toda_janela"] = st["citacoes"] - len(vistos)
    comprimentos.sort()
    n = len(comprimentos)
    st["palavras_gliner_por_janela"] = {"mediana": comprimentos[n // 2], "p99": comprimentos[int(0.99 * n)],
                                        "max": comprimentos[-1], "acima_de_384": sum(c > 384 for c in comprimentos)} if n else {}
    tot = sum(larguras.values())
    acum, quantis = 0, {}
    for w in sorted(larguras):
        acum += larguras[w]
        for q in (0.99, 0.995, 0.999, 1.0):
            if acum / tot >= q and q not in quantis:
                quantis[q] = w
    st["largura_citacao_palavras"] = {"max": max(larguras) if larguras else 0,
                                      "quantis": {str(q): w for q, w in quantis.items()},
                                      "acima_de_12": sum(v for k, v in larguras.items() if k > 12),
                                      "total": tot}
    st["rotulos"] = dict(collections.Counter(r for e in ex for *_, r in e["ner"]))
    return ex, dict(st)


# ------------------------------------------------------------------ métrica (a de bench/pontuar.extracao)

def extracao(gold: dict, pred: dict, juntar_vaga: bool = False) -> dict:
    """F1 por tipo e micro, casamento 1-para-1 com o mesmo tipo e IoU >= 0,5 (cópia de
    `bench.pontuar.extracao`, conferida em conferir.py) + F1 exato."""
    def norm(t):
        return "JURIS" if juntar_vaga and t == "VAGA" else t
    tp, fp, fn = collections.Counter(), collections.Counter(), collections.Counter()
    for doc, gs in gold.items():
        ps = [(a, b, norm(t)) for a, b, t in pred.get(doc, [])]
        usados = set()
        for a, b, t in gs:
            t = norm(t)
            k = next((k for k, (pa, pb, pt) in enumerate(ps) if k not in usados and pt == t
                      and max(0, min(b, pb) - max(a, pa)) / (max(b, pb) - min(a, pa)) >= 0.5), None)
            if k is None:
                fn[t] += 1
            else:
                usados.add(k)
                tp[t] += 1
        for k, (_, _, pt) in enumerate(ps):
            if k not in usados:
                fp[pt] += 1

    def f1(t, f, n):
        return round(2 * t / max(1, 2 * t + f + n), 4)
    tipos = sorted(set(tp) | set(fp) | set(fn))
    return {"f1": f1(sum(tp.values()), sum(fp.values()), sum(fn.values())),
            "por_tipo": {t: {"f1": f1(tp[t], fp[t], fn[t]), "tp": tp[t], "fp": fp[t], "fn": fn[t]}
                         for t in tipos}}


def f1_exato(gold: dict, pred: dict) -> float:
    tp = fp = fn = 0
    for doc, gs in gold.items():
        g = collections.Counter((a, b, t) for a, b, t in gs)
        p = collections.Counter((a, b, t) for a, b, t, *_ in pred.get(doc, []))
        inter = sum((g & p).values())
        tp += inter
        fn += sum(g.values()) - inter
        fp += sum(p.values()) - inter
    return round(2 * tp / max(1, 2 * tp + fp + fn), 4)


def filtrar(spans: list, t: float) -> list:
    """Spans extraídos no piso -> os que o limiar t daria. Exato: a decodificação do GLiNER e a
    fusão das janelas são gulosas por score, então elevar o limiar só corta o fim da fila."""
    return [s for s in spans if s[3] >= t]


# ------------------------------------------------------------------ modelo (só no job)

class Extrator:
    """O laço de `extrair` em `bench/extrair_gliner.py`: `predict_entities` janela a janela,
    offsets somados ao início da janela, fusão pelo maior score. Só o limiar vira parâmetro."""

    def __init__(self, modelo):
        self.modelo = modelo

    def extrair(self, texto: str, limiar: float) -> list[list]:
        import torch
        achados = []
        with torch.inference_mode():
            for a, b in janelas(texto):
                for e in self.modelo.predict_entities(texto[a:b], list(ROTULOS), threshold=limiar):
                    achados.append((a + e["start"], a + e["end"], ROTULOS[e["label"]], float(e["score"])))
        return fundir(achados)


def avaliar_reserva(ext: Extrator, reserva: dict) -> dict:
    """Extrai a reserva no PISO e mede F1 (IoU >= 0,5, VAGA separada, como no estresse) e F1
    exato em cada limiar da grade."""
    t0 = time.time()
    pred = {d: ext.extrair(texto, PISO) for d, (texto, _) in reserva.items()}
    gold = {d: ouro for d, (_, ouro) in reserva.items()}
    tabela = {}
    for t in GRADE:
        p = {d: [(a, b, r) for a, b, r, _ in filtrar(ss, t)] for d, ss in pred.items()}
        e = extracao(gold, p)
        tabela[f"{t:.2f}"] = {"f1": e["f1"], "f1_exato": f1_exato(gold, p), "por_tipo": e["por_tipo"]}
    return {"tabela": tabela, "segundos": round(time.time() - t0, 1)}


def escolher(historico: list) -> dict:
    """Critério fixado antes de treinar: maior F1 (IoU >= 0,5) na reserva; empate -> maior F1
    exato; depois o limiar mais perto de 0,5 (o do zero-shot); depois a época mais cedo."""
    cands = [(h["epoca"], float(t), v) for h in historico for t, v in h["tabela"].items()]
    e, t, v = max(cands, key=lambda c: (c[2]["f1"], c[2]["f1_exato"], -abs(c[1] - LIMIAR_ZERO_SHOT), -c[0]))
    return {"epoca": e, "limiar": t, "f1": v["f1"], "f1_exato": v["f1_exato"],
            "criterio": "maior F1 IoU>=0,5 na reserva de 590; empate: F1 exato, limiar mais perto de 0,5, "
                        "época mais cedo"}


def conferir_divisor(modelo, textos: list[str]) -> int:
    """Diferenças entre o divisor de palavras daqui e o do modelo (tem que ser 0)."""
    dif = 0
    for t in textos:
        dele = [(w, s, e) for w, s, e in modelo.data_processor.words_splitter(t)]
        dif += dele != palavras(t)
    return dif


def ler_entradas(raiz: pathlib.Path) -> list[dict]:
    docs = []
    for arq in ("entrada_remota.jsonl", "entrada_novas.jsonl"):      # estresse + reais, novas
        docs += [json.loads(x) for x in (raiz / "bench" / arq).read_text(encoding="utf-8").splitlines()]
    return docs


def extrair_avaliacao(modelo, meta: dict, limiar: float, raiz_bench: pathlib.Path, limite: int | None,
                      upload: bool) -> dict:
    """Estresse (600) + 305 + as 200 novas, no limiar escolhido (formato curto com score) e no
    PISO (diagnóstico). Sobe para bench/saida/controles/gliner_ft/."""
    import torch
    from huggingface_hub import HfApi
    ext = Extrator(modelo)
    docs = ler_entradas(raiz_bench)[:limite]
    ext.extrair("aquecimento: REsp nº 1.234.567/SP e art. 927 do Código Civil.", limiar)
    # o piso filtrado no limiar = a extração direta no limiar (a decodificação é gulosa por score)
    amostra = docs[:: max(1, len(docs) // 20)][:20]
    difs = sum(filtrar(ext.extrair(d["texto"], PISO), limiar) != ext.extrair(d["texto"], limiar) for d in amostra)
    print(f"conferencia piso filtrado x extracao no limiar: {difs} de {len(amostra)} documentos diferentes", flush=True)
    saida, piso = pathlib.Path("/tmp/modelo_gliner_ft.jsonl"), pathlib.Path("/tmp/modelo_gliner_ft_piso.jsonl")
    meta = {**meta, "limiar_decisao": limiar, "conferencia_piso_filtrado": {"docs": len(amostra), "diferentes": difs}}
    tempos = collections.defaultdict(list)
    with saida.open("w", encoding="utf-8") as fh, piso.open("w", encoding="utf-8") as fp:
        fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        fp.write(json.dumps({"meta": {**meta, "limiar_decisao": PISO, "nota": "diagnóstico: todos os spans "
                                      "com score >= piso; a saída do candidato é a outra"}}, ensure_ascii=False) + "\n")
        for k, d in enumerate(docs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            escolhidos = ext.extrair(d["texto"], limiar)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            tempos[d["conjunto"]].append(dt)
            fh.write(json.dumps({"conjunto": d["conjunto"], "id": d["id"], "segundos": round(dt, 4),
                                 "spans": escolhidos}, ensure_ascii=False) + "\n")
            t0 = time.perf_counter()
            ss = ext.extrair(d["texto"], PISO)
            fp.write(json.dumps({"conjunto": d["conjunto"], "id": d["id"],
                                 "segundos": round(time.perf_counter() - t0, 4), "spans": ss}, ensure_ascii=False) + "\n")
            if k % 200 == 0:
                print(f"extracao {k}/{len(docs)}", flush=True)
    res = {"s_por_doc": {c: round(sum(v) / len(v), 4) for c, v in tempos.items()},
           "docs": {c: len(v) for c, v in tempos.items()}}
    print("EXTRACAO", json.dumps(res), flush=True)
    if upload:
        api = HfApi()
        for arq in (saida, piso):
            info = api.upload_file(path_or_fileobj=str(arq), repo_id=DADOS, repo_type="dataset",
                                   path_in_repo=f"bench/saida/controles/{EXPERIMENTO}/{arq.name}",
                                   commit_message=f"controles: spans do GLiNER fine-tunado ({arq.name})")
            print("REVISAO", arq.name, info.oid, flush=True)
    return res


def cmd_treinar(a) -> int:
    import gliner
    import torch
    import transformers
    from gliner import GLiNER
    from gliner.training import Trainer, TrainingArguments
    from huggingface_hub import HfApi, snapshot_download
    from transformers import TrainerCallback

    random.seed(SEMENTE)
    transformers.set_seed(SEMENTE)
    t_ini = time.time()
    raiz_t = pathlib.Path(snapshot_download(DADOS, repo_type="dataset", revision=REV_TREINO,
                                            allow_patterns=["final_v3/**"]))
    raiz_b = pathlib.Path(snapshot_download(DADOS, repo_type="dataset", revision=REV_BENCH, allow_patterns=[
        "bench/entrada_remota.jsonl", "bench/entrada_novas.jsonl", "bench/codigo/**"]))
    sys.path.insert(0, str(raiz_b / "bench" / "codigo"))
    from gama.formas import forma

    docs, split = carregar(raiz_t / "final_v3")
    print("impressao dos textos do final_v3:", impressao_textos(docs), flush=True)
    treino = {d: v for d, v in docs.items() if split.get(d, "treino") == "treino"}
    reserva = {d: v for d, v in docs.items() if split.get(d) == "estresse"}
    if a.limite:
        treino = dict(list(treino.items())[: a.limite])
        reserva = dict(list(reserva.items())[: max(10, a.limite // 10)])
    print(f"docs: {len(treino)} treino, {len(reserva)} reserva", flush=True)

    # max_width: o GLiNER só enumera spans de até max_width palavras (12 no multi-v2.1). Regra
    # fixada antes de treinar: a maior largura de citação do split treino (0 = automático).
    _, st0 = exemplos(treino, forma)
    if not a.max_width:
        a.max_width = st0["largura_citacao_palavras"]["max"]
    print(f"max_width = {a.max_width} (maior citação do treino: {st0['largura_citacao_palavras']})", flush=True)
    modelo = GLiNER.from_pretrained(MODELO, revision=REVISAO, max_width=a.max_width)
    print("config:", json.dumps({k: getattr(modelo.config, k, None) for k in
                                 ("max_width", "max_len", "span_mode", "words_splitter_type", "model_name")}), flush=True)
    textos_amostra = [v[0] for v in list(treino.values())[:200]]
    dif_div = conferir_divisor(modelo, textos_amostra)
    print(f"divisor de palavras: {dif_div} diferenças em {len(textos_amostra)} textos", flush=True)
    assert dif_div == 0, "o divisor de palavras daqui não é o do modelo"

    ex_treino, st = exemplos(treino, forma, a.max_width)
    tok = modelo.data_processor.transformer_tokenizer
    subpalavras = sorted(len(tok(e["tokenized_text"], is_split_into_words=True)["input_ids"]) for e in ex_treino[:2000])
    st["subpalavras_por_janela"] = {"mediana": subpalavras[len(subpalavras) // 2], "max": subpalavras[-1]}
    print("conversao:", json.dumps(st, ensure_ascii=False), flush=True)
    for e in ex_treino:
        e.pop("_doc")
        e.pop("_janela")
    random.shuffle(ex_treino)

    modelo.to("cuda")
    ext = Extrator(modelo)
    historico: list = []
    pasta_ep = pathlib.Path("/tmp/epocas")

    class Reserva(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kw):
            ep = round(state.epoch or 0)
            r = avaliar_reserva(ext, reserva)
            melhor = max(r["tabela"].items(), key=lambda x: (x[1]["f1"], x[1]["f1_exato"]))
            print(f"EPOCA {ep}: melhor limiar {melhor[0]} F1 {melhor[1]['f1']} exato {melhor[1]['f1_exato']} "
                  f"| 0,50: {r['tabela']['0.50']['f1']} ({r['segundos']} s)", flush=True)
            historico.append({"epoca": ep, "passo": state.global_step, **r})
            modelo.save_pretrained(pasta_ep / f"epoca{ep}")
            modelo.train()

    bf16 = torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir="/tmp/saida", learning_rate=a.lr, others_lr=a.lr_outros, weight_decay=0.01,
        others_weight_decay=0.01, lr_scheduler_type="cosine", warmup_ratio=0.1,
        per_device_train_batch_size=a.lote, num_train_epochs=a.epocas, max_steps=a.passos or -1,
        max_grad_norm=1.0, focal_loss_alpha=-1, focal_loss_gamma=0, loss_reduction="sum",
        negatives=1.0, masking="none", save_strategy="no", logging_steps=100, report_to="none",
        bf16=bf16, seed=SEMENTE, data_seed=SEMENTE, dataloader_num_workers=2, remove_unused_columns=False)
    kw = {"tokenizer": tok} if int(transformers.__version__.split(".")[0]) < 5 else {"processing_class": tok}
    trainer = Trainer(model=modelo, args=args, train_dataset=ex_treino, data_collator=modelo._create_data_collator(),
                      callbacks=[Reserva()], **kw)
    t_treino = time.time()
    trainer.train()
    t_treino = time.time() - t_treino
    if a.passos and not historico:                      # smoke: max_steps antes do fim da 1ª época
        Reserva().on_epoch_end(None, trainer.state, None)

    escolha = escolher(historico)
    print("ESCOLHA", json.dumps(escolha, ensure_ascii=False), flush=True)
    del trainer
    torch.cuda.empty_cache()
    final = GLiNER.from_pretrained(str(pasta_ep / f"epoca{escolha['epoca']}"), local_files_only=True).to("cuda").eval()

    metricas = {"base": f"{MODELO}@{REVISAO}", "dados": f"{DADOS}@{REV_TREINO} final_v3",
                "n_treino": len(treino), "n_reserva": len(reserva), "conversao": st,
                "hiperparametros": {"lr_encoder": a.lr, "lr_outros": a.lr_outros, "lote": a.lote, "epocas": a.epocas,
                                    "max_width": a.max_width, "scheduler": "cosine", "warmup_ratio": 0.1,
                                    "weight_decay": 0.01, "max_grad_norm": 1.0, "perda": "focal desligada, soma",
                                    "negatives": 1.0, "masking": "none", "bf16": bf16, "semente": SEMENTE,
                                    "passos": a.passos},
                "historico_reserva": historico, "escolha": escolha, "minutos_treino": round(t_treino / 60, 1),
                "versoes": {"gliner": gliner.__version__, "transformers": transformers.__version__,
                            "torch": torch.__version__}, "dispositivo": torch.cuda.get_device_name(0)}
    rev = "local"
    if not a.sem_upload:
        destino = pathlib.Path("/tmp/modelo")
        final.save_pretrained(destino)
        (destino / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=1))
        (destino / "README.md").write_text(
            "Repositório temporário e privado de um controle experimental do Gama (GLiNER multi v2.1 "
            "fine-tunado no final_v3). Será apagado depois da revisão. Detalhes em metricas.json.\n")
        api = HfApi()
        api.create_repo(a.saida, private=True, exist_ok=True)
        info = api.upload_folder(folder_path=str(destino), repo_id=a.saida,
                                 commit_message=f"GLiNER fine-tunado: época {escolha['epoca']}, limiar "
                                                f"{escolha['limiar']}, dados {DADOS}@{REV_TREINO[:7]}")
        rev = info.oid
        print("REVISAO_MODELO", rev, flush=True)
    print("METRICAS", json.dumps(metricas, ensure_ascii=False), flush=True)
    if a.sem_extrair:
        return 0
    meta = {"extrator": EXPERIMENTO, "modelo": f"{a.saida}@{rev[:7]}", "base": f"{MODELO}@{REVISAO[:7]}",
            "parametros": sum(p.numel() for p in final.parameters()), "gliner": gliner.__version__,
            "transformers": transformers.__version__, "torch": torch.__version__,
            "dispositivo": torch.cuda.get_device_name(0), "epoca": escolha["epoca"], "max_width": a.max_width,
            "janelas": f"{PALAVRAS} palavras, passo {PASSO}", "rotulos": ROTULOS}
    extrair_avaliacao(final, meta, escolha["limiar"], raiz_b, a.limite and 3 * a.limite, not a.sem_upload)
    print(f"FIM {round((time.time() - t_ini) / 60, 1)} min", flush=True)
    return 0


def cmd_extrair(a) -> int:
    import gliner
    import torch
    import transformers
    from gliner import GLiNER
    from huggingface_hub import snapshot_download
    repo, rev = a.modelo.split("@")
    raiz_b = pathlib.Path(snapshot_download(DADOS, repo_type="dataset", revision=REV_BENCH, allow_patterns=[
        "bench/entrada_remota.jsonl", "bench/entrada_novas.jsonl"]))
    modelo = GLiNER.from_pretrained(repo, revision=rev).to("cuda").eval()
    meta = {"extrator": EXPERIMENTO, "modelo": f"{repo}@{rev[:7]}", "base": f"{MODELO}@{REVISAO[:7]}",
            "parametros": sum(p.numel() for p in modelo.parameters()), "gliner": gliner.__version__,
            "transformers": transformers.__version__, "torch": torch.__version__,
            "dispositivo": torch.cuda.get_device_name(0), "max_width": modelo.config.max_width,
            "janelas": f"{PALAVRAS} palavras, passo {PASSO}", "rotulos": ROTULOS}
    extrair_avaliacao(modelo, meta, a.limiar, raiz_b, a.limite, not a.sem_upload)
    return 0


def cmd_estatisticas(a) -> int:
    from gama.formas import forma                       # src/ no container (= bench/codigo do dataset)
    docs, split = carregar(pathlib.Path(a.pasta))
    print("impressao dos textos:", impressao_textos(docs))
    for nome in ("treino", "estresse"):
        sub = {d: v for d, v in docs.items() if split.get(d, "treino") == nome}
        _, st = exemplos(sub, forma, a.max_width)
        print(nome, len(sub), json.dumps(st, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("treinar")
    t.add_argument("--saida", default="vinimlo/gama-exp-gliner-ft")
    t.add_argument("--epocas", type=float, default=3)
    t.add_argument("--lr", type=float, default=1e-5)             # lr_encoder da documentação
    t.add_argument("--lr-outros", type=float, default=5e-5)      # lr_others da documentação
    t.add_argument("--lote", type=int, default=8)
    t.add_argument("--max-width", type=int, default=0, help="0 = maior citação do treino")
    t.add_argument("--passos", type=int, default=0, help="max_steps (só para o teste de fumaça)")
    t.add_argument("--limite", type=int, help="só N documentos de treino (teste de fumaça)")
    t.add_argument("--sem-upload", action="store_true")
    t.add_argument("--sem-extrair", action="store_true")
    x = sub.add_parser("extrair")
    x.add_argument("--modelo", required=True, help="repo@revisao")
    x.add_argument("--limiar", type=float, required=True)
    x.add_argument("--limite", type=int)
    x.add_argument("--sem-upload", action="store_true")
    s = sub.add_parser("estatisticas")
    s.add_argument("--pasta", required=True)
    s.add_argument("--max-width", type=int)
    a = ap.parse_args()
    return {"treinar": cmd_treinar, "extrair": cmd_extrair, "estatisticas": cmd_estatisticas}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
