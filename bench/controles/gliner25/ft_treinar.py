# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gliner2[local]==2.0.0",
#   "torch==2.14.0",
#   "transformers==4.57.6",
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
"""GLiNER 2.5 multi fine-tunado no final_v3, com o protocolo do fine-tune da v2.1.

Refaz `bench/controles/gliner_ft/treinar_gliner.py` trocando a base: `fastino/gliner2.5-multi-v1`
@2ca71aa (287M, mDeBERTa-v3-base, arquitetura "boundary"), biblioteca `gliner2==2.0.0`, carregado
por `AutoExtractor` a partir de um `snapshot_download` da revisão (o tokenizer da biblioteca não
recebe a revisão; carregar da pasta fixa pesos e tokenizer juntos).

O que é igual ao fine-tune da v2.1
    dados      `final_v3` de `vinimlo/gama-goldenset@31474b1`: split `treino` (5.410 documentos)
               para treinar, split `estresse` do meta.jsonl (590, a reserva interna de ~10%) para
               escolher época e limiar. Nenhuma ementa real entra no treino nem na escolha.
    janelas    as da inferência (200 palavras \\S+, passo 150), com a borda movida para nunca cortar
               citação do ouro (`janelas_treino`); janela sem citação entra como negativa.
    rótulos    os mesmos quatro nomes em português, presentes em TODA janela (os ausentes viram
               consultas sem ouro, o negativo da 2.5); VAGA -> "referência a julgado sem número";
               LEI -> "artigo de lei"; JURIS -> "súmula ou tema de tribunal" se `DetectorDeForma.forma`
               der súmula/tema, senão "precedente judicial com número".
    receita    3 épocas, lote 8, cosseno, aquecimento 10%, decaimento 0,01, norma do gradiente 1,0,
               bf16, semente 13; taxas da documentação da biblioteca, sem busca (encoder 1e-5,
               cabeças 5e-4, o exemplo mínimo do tutorial de treino da gliner2).
    escolha    ao fim de cada época: extração da reserva no PISO e F1 (IoU >= 0,5, VAGA separada)
               e F1 exato em cada limiar da GRADE 0,10..0,95; maior F1, empate F1 exato, limiar
               mais perto de 0,5, época mais cedo (`escolher`, copiado).
    inferência janelas de 200/150, fusão entre janelas pelo maior score (`fundir`, copiado).

O que muda porque a biblioteca é outra (declarado)
    treino     `gliner2.training.trainer.ExtractorTrainer`, o caminho de treino local da biblioteca,
               com os padrões dela (amostragem de rótulos sintéticos em 20% dos exemplos, injeção de
               ouro nas propostas no começo do treino, agrupamento por comprimento). A avaliação da
               reserva roda num gancho de `_save_checkpoint` ao fim de cada época.
    ouro       a biblioteca recebe a menção como TEXTO e marca como ouro toda ocorrência dela na
               janela (`SchemaTransformer._find_sublist`, sem caixa). A v2.1 recebia posições. A
               auditoria da conversão (`auditar`) conta as ocorrências a mais; o subcomando
               `estatisticas` mede isso no container antes do treino.
    largura    a 2.5 não tem grade de largura de span (pares início/fim esparsos): não há max_width.
    decodific. "flat" por rótulo (intervalo ponderado, conjunto sem sobreposição de maior soma), o
               padrão do checkpoint. Com ele, filtrar a extração do piso NÃO dá a extração no limiar.
               Por isso o piso sai com a política "allow" (todos os candidatos acima do piso, por
               janela e rótulo) e cada limiar é reconstruído aplicando o limiar e depois
               `gliner2.inference.overlap.resolve_overlaps(..., "flat")`, a função da própria
               biblioteca, antes da fusão entre janelas. A extração final também roda direto no
               limiar escolhido e o job confere, documento a documento, que as duas coincidem.

    # conversão e auditoria do ouro por texto, no container (sem gliner2, sem torch)
    docker compose run --rm gama python bench/controles/gliner25/ft_treinar.py \\
        estatisticas --pasta corpus/goldenset/v3
    # fumaça (A10G small): poucos documentos, poucos passos, sem subir nada
    hf jobs uv run --flavor a10g-small --timeout 20m --secrets HF_TOKEN \\
        bench/controles/gliner25/ft_treinar.py treinar --limite 60 --passos 30 --sem-upload
    # treino + escolha + publicação + extração dos três conjuntos, num job só
    hf jobs uv run --flavor a10g-small --timeout <T> --secrets HF_TOKEN \\
        bench/controles/gliner25/ft_treinar.py treinar --saida vinimlo/gama-exp-gliner25-ft
    # só a extração, de um modelo já publicado
    hf jobs uv run --flavor a10g-small --timeout 30m --secrets HF_TOKEN \\
        bench/controles/gliner25/ft_treinar.py extrair --modelo vinimlo/gama-exp-gliner25-ft@<rev> --limiar <t>

Saídas no dataset, em bench/saida/controles/gliner25/ft/:
    modelo_gliner25_ft.jsonl        spans no limiar escolhido, formato curto com score (o harness)
    modelo_gliner25_ft_piso.jsonl   diagnóstico: a extração no PISO (flat + fusão)
    candidatos_gliner25_ft.jsonl    por documento e janela, todos os candidatos acima do PISO com
                                    score em precisão cheia (reconstrói qualquer limiar)

Copiado (o job só recebe este arquivo): de `bench/extrair_gliner.py` ROTULOS, PALAVRAS, PASSO,
`janelas` e a fusão; de `bench/controles/gliner_ft/treinar_gliner.py` DADOS, REV_TREINO, REV_BENCH,
ROT_OURO, NOME, PISO, GRADE, SEMENTE, LIMIAR_ZERO_SHOT, `carregar`, `impressao_textos`,
`nome_rotulo`, `janelas_treino`, `extracao`, `f1_exato`, `escolher`, `ler_entradas`. De
`gliner2/processing/word_splitter.py` o padrão do divisor de palavras (o job confere contra o do
modelo). `bench/controles/gliner25/ft_conferir.py` confere as cópias no container.
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
ROTULOS = {
    "precedente judicial com número": "JURIS",
    "súmula ou tema de tribunal": "JURIS",
    "artigo de lei": "LEI",
    "referência a julgado sem número": "VAGA",
}
PALAVRAS, PASSO = 200, 150


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
# ------------------------------------------------------------------ copiado de gliner_ft/treinar_gliner.py
LIMIAR_ZERO_SHOT = 0.5
DADOS = "vinimlo/gama-goldenset"
REV_TREINO = "31474b1f7db9096c4ca4f2a4eae2e9b82852d7a7"   # final_v3 do v1.2 (MODELO.md)
REV_BENCH = "2fff5f670e13f77f339937cfe3da52ed0af9572d"    # entradas do benchmark + bench/codigo
ROT_OURO = {"incompleta": "VAGA"}                          # como treino/treinar.py
NOME = {"LEI": "artigo de lei", "VAGA": "referência a julgado sem número"}
PISO = 0.10                                                # limiar mínimo extraído (a grade filtra acima)
GRADE = [round(0.10 + 0.05 * i, 2) for i in range(18)]     # 0,10 a 0,95
SEMENTE = 13


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


def escolher(historico: list) -> dict:
    """Critério fixado antes de treinar: maior F1 (IoU >= 0,5) na reserva; empate -> maior F1
    exato; depois o limiar mais perto de 0,5 (o do zero-shot); depois a época mais cedo."""
    cands = [(h["epoca"], float(t), v) for h in historico for t, v in h["tabela"].items()]
    e, t, v = max(cands, key=lambda c: (c[2]["f1"], c[2]["f1_exato"], -abs(c[1] - LIMIAR_ZERO_SHOT), -c[0]))
    return {"epoca": e, "limiar": t, "f1": v["f1"], "f1_exato": v["f1_exato"],
            "criterio": "maior F1 IoU>=0,5 na reserva de 590; empate: F1 exato, limiar mais perto de 0,5, "
                        "época mais cedo"}


def ler_entradas(raiz: pathlib.Path) -> list[dict]:
    docs = []
    for arq in ("entrada_remota.jsonl", "entrada_novas.jsonl"):      # estresse + reais, novas
        docs += [json.loads(x) for x in (raiz / "bench" / arq).read_text(encoding="utf-8").splitlines()]
    return docs
# ------------------------------------------------------------------ copiado de gliner2 (word_splitter.py)
PALAVRA = re.compile(
    r"""(?:https?://[^\s]+|www\.[^\s]+)
    |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
    |@[a-z0-9_]+
    |\w+(?:[-_]\w+)*
    |\S""",
    re.VERBOSE | re.IGNORECASE,
)
# ------------------------------------------------------------------ fim dos trechos copiados

MODELO = "fastino/gliner2.5-multi-v1"
REVISAO = "2ca71aafb3446d9014e1c55c7ff51c9bc7209c47"
SAIDA_MODELO = "vinimlo/gama-exp-gliner25-ft"
PREFIXO = "bench/saida/controles/gliner25/ft"
LOTE_MAX = 32                                              # janelas por chamada de inferência


def palavras(texto: str) -> list[tuple[str, int, int]]:
    """O divisor da gliner2 com `lower=True`: (token minúsculo, início, fim)."""
    return [(m.group().lower(), m.start(), m.end()) for m in PALAVRA.finditer(texto)]


def texto_do_processador(texto: str) -> str:
    """`SchemaTransformer._collate_batch` acrescenta "." ao texto que não termina em . ! ?"""
    if texto and not texto.endswith((".", "!", "?")):
        return texto + "."
    return texto or "."


def ocorrencias(sub: list[str], lst: list[str]) -> list[tuple[int, int]]:
    """`SchemaTransformer._find_sublist` (sem caixa já aplicada): (início, fim inclusivo)."""
    n = len(sub)
    return [(i, i + n - 1) for i in range(len(lst) - n + 1) if lst[i:i + n] == sub]


# ------------------------------------------------------------------ conversão

def exemplos(docs: dict, forma) -> tuple[list, dict]:
    """Documentos -> registros {"input", "output": {"entities": {rótulo: [menções]}}} da gliner2 +
    estatísticas + auditoria do ouro por texto (o que a biblioteca vai marcar como ouro)."""
    ex, st = [], collections.Counter()
    comprimentos, vistos = [], set()
    auditoria = collections.Counter()
    amostras_extra: list = []
    for doc, (texto, ouro) in docs.items():
        pares = [(a, b) for a, b, _ in ouro]
        st["citacoes"] += len(ouro)
        st["sobrepostas"] += sum(1 for (a1, b1), (a2, b2) in zip(pares, pares[1:]) if a2 < b1)
        for wa, wb in janelas_treino(texto, pares):
            wtexto = texto[wa:wb]
            toks = palavras(texto_do_processador(wtexto))
            lista = [w for w, _, _ in toks]
            comprimentos.append(len(toks))
            entidades = {r: [] for r in ROTULOS}
            ouro_pos = collections.defaultdict(set)          # rótulo -> {(i, j)} do ouro
            for a, b, tipo in ouro:
                if not (wa <= a and b <= wb):
                    if a < wb and wa < b:
                        st["cortadas_pela_janela"] += 1
                    continue
                ra, rb = a - wa, b - wa
                i = next((k for k, (_, s, e) in enumerate(toks) if e > ra), None)
                j = next((k for k in range(len(toks) - 1, -1, -1) if toks[k][1] < rb), None)
                if i is None or j is None or j < i:
                    st["sem_palavra"] += 1
                    continue
                if toks[i][1] != ra or toks[j][2] != rb:
                    st["borda_desalinhada"] += 1
                rot = nome_rotulo(texto, a, b, tipo, forma)
                sup = texto[a:b]
                if sup not in entidades[rot]:
                    entidades[rot].append(sup)
                ouro_pos[rot].add((i, j))
                vistos.add((doc, a, b))
            # auditoria: posições que a biblioteca marca (toda ocorrência de cada menção) x ouro
            todas_ouro = {p for ps in ouro_pos.values() for p in ps}
            for rot, sups in entidades.items():
                marcadas = set()
                for sup in sups:
                    marcadas |= set(ocorrencias([w for w, _, _ in palavras(sup)], lista))
                auditoria["menções_unicas"] += len(sups)
                auditoria["posicoes_ouro"] += len(ouro_pos[rot])
                auditoria["posicoes_marcadas"] += len(marcadas)
                auditoria["ouro_nao_marcado"] += len(ouro_pos[rot] - marcadas)
                for p in sorted(marcadas - ouro_pos[rot]):
                    dentro = any(q[0] <= p[0] and p[1] <= q[1] for q in todas_ouro)
                    cruza = any(p[0] <= q[1] and q[0] <= p[1] for q in todas_ouro)
                    chave = ("extra_ouro_de_outro_rotulo" if p in todas_ouro else
                             "extra_dentro_de_outra_citacao" if dentro else
                             "extra_cruzando_citacao" if cruza else "extra_fora_de_citacao")
                    auditoria[chave] += 1
                    auditoria["extra_total"] += 1
                    if len(amostras_extra) < 12:
                        amostras_extra.append({"doc": doc, "rotulo": rot, "tipo": chave,
                                               "trecho": " ".join(lista[p[0]:p[1] + 1])[:80]})
            st["janelas"] += 1
            st["janelas_sem_citacao"] += not any(entidades.values())
            ex.append({"input": wtexto, "output": {"entities": entidades}, "_doc": doc, "_janela": [wa, wb]})
    st["citacoes_em_alguma_janela"] = len(vistos)
    st["citacoes_fora_de_toda_janela"] = st["citacoes"] - len(vistos)
    comprimentos.sort()
    n = len(comprimentos)
    st["palavras_gliner2_por_janela"] = {"mediana": comprimentos[n // 2], "p99": comprimentos[int(0.99 * n)],
                                         "max": comprimentos[-1]} if n else {}
    st["rotulos"] = dict(collections.Counter(r for e in ex for r, v in e["output"]["entities"].items()
                                             for _ in v))
    mx = collections.Counter()
    for e in ex:
        for r, v in e["output"]["entities"].items():
            mx[r] = max(mx[r], len(v))
    st["max_mencoes_unicas_por_rotulo_na_janela"] = dict(mx)
    aud = dict(auditoria)
    aud["frac_extra_sobre_ouro"] = round(aud.get("extra_total", 0) / max(1, aud.get("posicoes_ouro", 0)), 6)
    aud["amostras"] = amostras_extra
    st["auditoria_ouro_por_texto"] = aud
    return ex, dict(st)


# ------------------------------------------------------------------ extração (só no job)

class Extrator:
    """O laço de `extrair` de `bench/extrair_gliner.py` com a 2.5: janelas de 200/150, offsets
    somados ao início da janela, fusão pelo maior score. As janelas de um documento vão num lote."""

    def __init__(self, modelo):
        from gliner2.inference.overlap import resolve_overlaps
        self.modelo = modelo
        self.resolver = resolve_overlaps

    def _chamar(self, pedacos: list[str], limiar: float, politica: str | None) -> list[dict]:
        out = []
        for k in range(0, len(pedacos), LOTE_MAX):
            bloco = pedacos[k:k + LOTE_MAX]
            out += self.modelo.batch_extract_entities(bloco, list(ROTULOS), batch_size=len(bloco),
                                                      threshold=limiar, include_confidence=True,
                                                      include_spans=True, overlap_policy=politica)
        return out

    def candidatos(self, texto: str) -> list[dict]:
        """Por janela: {"janela": [a, b], "c": {rótulo: [[ini, fim, score], ...]}} no PISO, política
        "allow" (todo candidato distinto acima do piso; offsets da janela)."""
        js = janelas(texto)
        res = self._chamar([texto[a:b] for a, b in js], PISO, "allow")
        out = []
        for (a, b), r in zip(js, res):
            c = {nome: [[int(e["start"]), int(e["end"]), float(e["confidence"])] for e in ents]
                 for nome, ents in (r.get("entities") or {}).items() if ents}
            out.append({"janela": [a, b], "c": c})
        return out

    def decodificar(self, cands: list[dict], limiar: float) -> list[list]:
        """A extração no limiar a partir dos candidatos do piso: limiar, depois "flat" por rótulo
        (a função da biblioteca), depois a fusão entre janelas."""
        achados = []
        for jan in cands:
            a = jan["janela"][0]
            for nome, cs in jan["c"].items():
                ok = [c for c in cs if c[2] >= limiar]
                for s, f, sc in self.resolver(ok, "flat", score=lambda x: x[2], start=lambda x: x[0],
                                              end=lambda x: x[1]):
                    achados.append((a + s, a + f, ROTULOS[nome], sc))
        return fundir(achados)

    def direto(self, texto: str, limiar: float) -> list[list]:
        """A extração no limiar com a política padrão do checkpoint (a que um usuário rodaria)."""
        js = janelas(texto)
        achados = []
        for (a, b), r in zip(js, self._chamar([texto[a:b] for a, b in js], limiar, None)):
            for nome, ents in (r.get("entities") or {}).items():
                for e in ents or []:
                    achados.append((a + int(e["start"]), a + int(e["end"]), ROTULOS[nome], float(e["confidence"])))
        return fundir(achados)


def avaliar_reserva(ext: Extrator, reserva: dict) -> dict:
    """Candidatos da reserva no PISO e, em cada limiar da GRADE, F1 (IoU >= 0,5, VAGA separada,
    como no estresse) e F1 exato."""
    t0 = time.time()
    cands = {d: ext.candidatos(texto) for d, (texto, _) in reserva.items()}
    gold = {d: ouro for d, (_, ouro) in reserva.items()}
    tabela = {}
    for t in GRADE:
        p = {d: [(a, b, r) for a, b, r, _ in ext.decodificar(c, t)] for d, c in cands.items()}
        e = extracao(gold, p)
        tabela[f"{t:.2f}"] = {"f1": e["f1"], "f1_exato": f1_exato(gold, p), "por_tipo": e["por_tipo"]}
    return {"tabela": tabela, "segundos": round(time.time() - t0, 1)}


def conferir_divisor(modelo, textos: list[str]) -> int:
    """Diferenças entre o divisor daqui e o do modelo (tem que ser 0)."""
    return sum([(w, s, e) for w, s, e in modelo.processor.word_splitter(t, lower=True)] != palavras(t)
               for t in textos)


def carregar_base():
    from gliner2 import AutoExtractor
    from huggingface_hub import snapshot_download
    pasta = snapshot_download(MODELO, revision=REVISAO)
    return AutoExtractor.from_pretrained(pasta)


def extrair_avaliacao(modelo, meta: dict, limiar: float, raiz_bench: pathlib.Path, limite: int | None,
                      upload: bool) -> dict:
    """Estresse (600) + 305 + as 200 novas. Por documento: a extração direta no limiar (tempo
    medido, a saída do candidato), os candidatos do piso (uma passada, "allow") e, deles, a
    extração no PISO e a reconstrução no limiar, conferida contra a direta."""
    import torch
    from huggingface_hub import HfApi
    ext = Extrator(modelo)
    docs = ler_entradas(raiz_bench)
    if limite:
        por = collections.defaultdict(list)
        for d in docs:
            por[d["conjunto"]].append(d)
        docs = [d for v in por.values() for d in v[:limite]]
    ext.direto("aquecimento: REsp nº 1.234.567/SP e art. 927 do Código Civil. " * 30, limiar)
    saida = pathlib.Path("/tmp/modelo_gliner25_ft.jsonl")
    piso = pathlib.Path("/tmp/modelo_gliner25_ft_piso.jsonl")
    cand = pathlib.Path("/tmp/candidatos_gliner25_ft.jsonl")
    tempos = collections.defaultdict(list)
    conf = {"docs": 0, "spans_diferentes": 0, "so_score_diferente": 0, "max_dif_score": 0.0, "exemplos": []}
    meta = {**meta, "limiar_decisao": limiar}
    with saida.open("w", encoding="utf-8") as fh, piso.open("w", encoding="utf-8") as fp, \
            cand.open("w", encoding="utf-8") as fc:
        fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        fp.write(json.dumps({"meta": {**meta, "limiar_decisao": PISO, "nota": "diagnóstico: a extração no piso "
                                      "(flat por rótulo + fusão); a saída do candidato é a outra"}},
                            ensure_ascii=False) + "\n")
        fc.write(json.dumps({"meta": {**meta, "piso": PISO, "nota": "candidatos por janela e rótulo, política "
                                      "allow, offsets da janela, score em precisão cheia"}},
                            ensure_ascii=False) + "\n")
        for k, d in enumerate(docs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            escolhidos = ext.direto(d["texto"], limiar)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            tempos[d["conjunto"]].append(dt)
            fh.write(json.dumps({"conjunto": d["conjunto"], "id": d["id"], "segundos": round(dt, 4),
                                 "spans": escolhidos}, ensure_ascii=False) + "\n")
            t0 = time.perf_counter()
            cs = ext.candidatos(d["texto"])
            dt_c = time.perf_counter() - t0
            fc.write(json.dumps({"conjunto": d["conjunto"], "id": d["id"], "segundos": round(dt_c, 4),
                                 "janelas": cs}, ensure_ascii=False) + "\n")
            fp.write(json.dumps({"conjunto": d["conjunto"], "id": d["id"], "segundos": round(dt_c, 4),
                                 "spans": ext.decodificar(cs, PISO)}, ensure_ascii=False) + "\n")
            rec = ext.decodificar(cs, limiar)
            conf["docs"] += 1
            if [s[:3] for s in rec] != [s[:3] for s in escolhidos]:
                conf["spans_diferentes"] += 1
                if len(conf["exemplos"]) < 5:
                    conf["exemplos"].append({"id": d["id"], "direto": escolhidos[:6], "reconstruido": rec[:6]})
            else:
                m = max((abs(p[3] - q[3]) for p, q in zip(rec, escolhidos)), default=0.0)
                conf["so_score_diferente"] += m > 0
                conf["max_dif_score"] = max(conf["max_dif_score"], round(m, 6))
            if k % 200 == 0:
                print(f"extracao {k}/{len(docs)}", flush=True)
    res = {"s_por_doc": {c: round(sum(v) / len(v), 4) for c, v in tempos.items()},
           "docs": {c: len(v) for c, v in tempos.items()}, "conferencia_reconstrucao": conf}
    print("EXTRACAO", json.dumps(res, ensure_ascii=False), flush=True)
    if upload:
        api = HfApi()
        for arq in (saida, piso, cand):
            info = api.upload_file(path_or_fileobj=str(arq), repo_id=DADOS, repo_type="dataset",
                                   path_in_repo=f"{PREFIXO}/{arq.name}",
                                   commit_message=f"controles: GLiNER 2.5 fine-tunado ({arq.name})")
            print("REVISAO", arq.name, info.oid, flush=True)
    return res


# ------------------------------------------------------------------ comandos

def cmd_treinar(a) -> int:
    import gliner2
    import torch
    import transformers
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
    from huggingface_hub import HfApi, snapshot_download

    random.seed(SEMENTE)
    transformers.set_seed(SEMENTE)
    t_ini = time.time()
    raiz_t = pathlib.Path(snapshot_download(DADOS, repo_type="dataset", revision=REV_TREINO,
                                            allow_patterns=["final_v3/**"]))
    raiz_b = pathlib.Path(snapshot_download(DADOS, repo_type="dataset", revision=REV_BENCH, allow_patterns=[
        "bench/entrada_remota.jsonl", "bench/entrada_novas.jsonl", "bench/codigo/**"]))
    sys.path.insert(0, str(raiz_b / "bench" / "codigo"))
    from gama.formas import DetectorDeForma

    docs, split = carregar(raiz_t / "final_v3")
    print("impressao dos textos do final_v3:", impressao_textos(docs), flush=True)
    treino = {d: v for d, v in docs.items() if split.get(d, "treino") == "treino"}
    reserva = {d: v for d, v in docs.items() if split.get(d) == "estresse"}
    if a.limite:
        treino = dict(list(treino.items())[: a.limite])
        reserva = dict(list(reserva.items())[: max(10, a.limite // 10)])
    print(f"docs: {len(treino)} treino, {len(reserva)} reserva", flush=True)

    modelo = carregar_base()
    print("dtype carregado:", str(next(modelo.parameters()).dtype), flush=True)
    modelo.float()                                        # pesos mestres em fp32; bf16 só no autocast
    cfg = modelo.config
    bh = getattr(cfg, "boundary_head", None)
    info_modelo = {"classe": type(modelo).__name__, "arquitetura": getattr(cfg, "architecture", None),
                   "max_len": getattr(cfg, "max_len", None), "politica_padrao": modelo._resolved_overlap_policy(None),
                   "max_gold_per_query": (bh or {}).get("max_gold_per_query") if isinstance(bh, dict) else
                   getattr(bh, "max_gold_per_query", None),
                   "parametros": sum(p.numel() for p in modelo.parameters()),
                   "dtype": str(next(modelo.parameters()).dtype)}
    print("modelo:", json.dumps(info_modelo, ensure_ascii=False, default=str), flush=True)
    assert info_modelo["politica_padrao"] == "disallow", "a política padrão do checkpoint não é flat"
    textos_amostra = [v[0] for v in list(treino.values())[:200]]
    dif_div = conferir_divisor(modelo, textos_amostra)
    print(f"divisor de palavras: {dif_div} diferenças em {len(textos_amostra)} textos", flush=True)
    assert dif_div == 0, "o divisor de palavras daqui não é o do modelo"

    registros, st = exemplos(treino, DetectorDeForma().forma)
    tok = modelo.processor.tokenizer
    sub = sorted(len(tok(r["input"], add_special_tokens=False)["input_ids"]) for r in registros[:2000])
    st["subpalavras_por_janela"] = {"mediana": sub[len(sub) // 2], "max": sub[-1]}
    print("conversao:", json.dumps(st, ensure_ascii=False), flush=True)
    for r in registros:
        r.pop("_doc")
        r.pop("_janela")

    ext_holder: dict = {}
    historico: list = []
    saida_tr = pathlib.Path("/tmp/saida")

    class Treinador(ExtractorTrainer):
        """O trainer da biblioteca, com a avaliação da reserva ao salvar o checkpoint da época."""

        def _save_checkpoint(self, name: str):
            super()._save_checkpoint(name)
            if not name.startswith("checkpoint-epoch-"):
                return
            ep = int(name.rsplit("-", 1)[1])
            r = avaliar_reserva(Extrator(self.model), reserva)
            melhor = max(r["tabela"].items(), key=lambda x: (x[1]["f1"], x[1]["f1_exato"]))
            print(f"EPOCA {ep}: melhor limiar {melhor[0]} F1 {melhor[1]['f1']} exato {melhor[1]['f1_exato']} "
                  f"| 0,50: {r['tabela']['0.50']['f1']} ({r['segundos']} s)", flush=True)
            historico.append({"epoca": ep, "passo": self.global_step, "checkpoint": name, **r})
            self.model.train()
            self.processor.change_mode(is_training=True)

    config = TrainingConfig(
        output_dir=str(saida_tr), experiment_name="gama-gliner25-ft", num_epochs=a.epocas,
        max_steps=a.passos or -1, batch_size=a.lote, eval_batch_size=a.lote, encoder_lr=a.lr,
        task_lr=a.lr_cabecas, weight_decay=0.01, max_grad_norm=1.0, scheduler_type="cosine",
        warmup_ratio=0.1, fp16=False, bf16=True, eval_strategy="epoch", save_total_limit=10,
        save_best=False, logging_steps=100, seed=SEMENTE, num_workers=a.workers,
        validate_data=True, report_to_wandb=False)
    trainer = Treinador(modelo, config)
    t_treino = time.time()
    resumo_treino = trainer.train(train_data=registros)
    t_treino = time.time() - t_treino
    if not historico:                                     # fumaça: max_steps antes do fim da época
        trainer._save_checkpoint("checkpoint-epoch-1")
    print("TREINO", json.dumps({k: v for k, v in resumo_treino.items() if not k.endswith("history")},
                               default=str), flush=True)

    escolha = escolher(historico)
    print("ESCOLHA", json.dumps(escolha, ensure_ascii=False), flush=True)
    pasta_escolhida = saida_tr / next(h["checkpoint"] for h in historico if h["epoca"] == escolha["epoca"])
    del trainer
    torch.cuda.empty_cache()
    final = AutoExtractor.from_pretrained(str(pasta_escolhida)).to("cuda").eval()
    # o tokenizer salvo no checkpoint recarrega com um aviso genérico do transformers (regex do
    # Mistral, só aviso): confere que ele tokeniza igual ao do treino, com os tokens especiais
    amostra = [r["input"] for r in registros[:300]] + [
        "[E] precedente judicial com número [E] artigo de lei [SEP_STRUCT] [SEP_TEXT] " + r["input"]
        for r in registros[300:400]]
    tok_dif = sum(tok(t)["input_ids"] != final.processor.tokenizer(t)["input_ids"] for t in amostra)
    st["tokenizer_recarregado_diferente"] = {"textos": len(amostra), "diferentes": tok_dif}
    print("tokenizer recarregado:", tok_dif, "de", len(amostra), "textos diferentes", flush=True)
    assert tok_dif == 0, "o tokenizer do checkpoint não tokeniza como o do treino"
    del modelo
    torch.cuda.empty_cache()

    metricas = {"base": f"{MODELO}@{REVISAO}", "dados": f"{DADOS}@{REV_TREINO} final_v3",
                "n_treino": len(treino), "n_reserva": len(reserva), "conversao": st, "modelo": info_modelo,
                "hiperparametros": {"encoder_lr": a.lr, "task_lr": a.lr_cabecas, "lote": a.lote,
                                    "epocas": a.epocas, "scheduler": "cosine", "warmup_ratio": 0.1,
                                    "weight_decay": 0.01, "max_grad_norm": 1.0, "bf16": True,
                                    "semente": SEMENTE, "passos": a.passos, "num_workers": a.workers,
                                    "demais": "padrões de TrainingConfig e SamplingConfig da gliner2 2.0.0"},
                "historico_reserva": historico, "escolha": escolha, "minutos_treino": round(t_treino / 60, 1),
                "resumo_treino": {k: v for k, v in resumo_treino.items() if not k.endswith("history")},
                "versoes": {"gliner2": gliner2.__version__, "transformers": transformers.__version__,
                            "torch": torch.__version__}, "dispositivo": torch.cuda.get_device_name(0)}
    rev = "local"
    if not a.sem_upload:
        (pasta_escolhida / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=1, default=str))
        (pasta_escolhida / "README.md").write_text(
            "Repositório temporário e privado de um controle experimental do Gama (GLiNER 2.5 multi "
            "fine-tunado no final_v3). Será apagado depois da revisão. Detalhes em metricas.json.\n")
        api = HfApi()
        api.create_repo(a.saida, private=True, exist_ok=True)
        info = api.upload_folder(folder_path=str(pasta_escolhida), repo_id=a.saida,
                                 commit_message=f"GLiNER 2.5 fine-tunado: época {escolha['epoca']}, limiar "
                                                f"{escolha['limiar']}, dados {DADOS}@{REV_TREINO[:7]}")
        rev = info.oid
        print("REVISAO_MODELO", rev, flush=True)
    print("METRICAS", json.dumps(metricas, ensure_ascii=False, default=str), flush=True)
    if a.sem_extrair:
        return 0
    meta = {"extrator": "gliner25_ft", "modelo": f"{a.saida}@{rev[:7]}", "revisao_modelo": rev,
            "base": f"{MODELO}@{REVISAO[:7]}", "parametros": sum(p.numel() for p in final.parameters()),
            "gliner2": gliner2.__version__, "transformers": transformers.__version__, "torch": torch.__version__,
            "dispositivo": torch.cuda.get_device_name(0), "epoca": escolha["epoca"],
            "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "janelas": f"{PALAVRAS} palavras, passo {PASSO}", "rotulos": ROTULOS,
            "decodificacao": "flat por rótulo (padrão do checkpoint), fusão entre janelas pelo maior score"}
    extrair_avaliacao(final, meta, escolha["limiar"], raiz_b, a.limite and 5, not a.sem_upload)
    print(f"FIM {round((time.time() - t_ini) / 60, 1)} min", flush=True)
    return 0


def cmd_extrair(a) -> int:
    import gliner2
    import torch
    import transformers
    from gliner2 import AutoExtractor
    from huggingface_hub import snapshot_download
    torch.backends.cuda.matmul.allow_tf32 = True          # como no job de treino (padrão do trainer)
    torch.backends.cudnn.allow_tf32 = True
    repo, rev = a.modelo.split("@")
    raiz_b = pathlib.Path(snapshot_download(DADOS, repo_type="dataset", revision=REV_BENCH, allow_patterns=[
        "bench/entrada_remota.jsonl", "bench/entrada_novas.jsonl"]))
    pasta = snapshot_download(repo, revision=rev)
    modelo = AutoExtractor.from_pretrained(pasta).to("cuda").eval()
    meta = {"extrator": "gliner25_ft", "modelo": f"{repo}@{rev[:7]}", "revisao_modelo": rev,
            "base": f"{MODELO}@{REVISAO[:7]}", "parametros": sum(p.numel() for p in modelo.parameters()),
            "gliner2": gliner2.__version__, "transformers": transformers.__version__, "torch": torch.__version__,
            "dispositivo": torch.cuda.get_device_name(0), "tf32": True,
            "janelas": f"{PALAVRAS} palavras, passo {PASSO}", "rotulos": ROTULOS,
            "decodificacao": "flat por rótulo (padrão do checkpoint), fusão entre janelas pelo maior score"}
    extrair_avaliacao(modelo, meta, a.limiar, raiz_b, a.limite, not a.sem_upload)
    return 0


def cmd_estatisticas(a) -> int:
    from gama.formas import DetectorDeForma                       # src/ no container (= bench/codigo do dataset)
    docs, split = carregar(pathlib.Path(a.pasta))
    print("impressao dos textos:", impressao_textos(docs))
    for nome in ("treino", "estresse"):
        sub = {d: v for d, v in docs.items() if split.get(d, "treino") == nome}
        _, st = exemplos(sub, DetectorDeForma().forma)
        print(nome, len(sub), json.dumps(st, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("treinar")
    t.add_argument("--saida", default=SAIDA_MODELO)
    t.add_argument("--epocas", type=int, default=3)
    t.add_argument("--lr", type=float, default=1e-5)             # encoder_lr do tutorial de treino
    t.add_argument("--lr-cabecas", type=float, default=5e-4)     # task_lr do tutorial de treino
    t.add_argument("--lote", type=int, default=8)
    t.add_argument("--workers", type=int, default=3)
    t.add_argument("--passos", type=int, default=0, help="max_steps (só para o teste de fumaça)")
    t.add_argument("--limite", type=int, help="só N documentos de treino (teste de fumaça)")
    t.add_argument("--sem-upload", action="store_true")
    t.add_argument("--sem-extrair", action="store_true")
    x = sub.add_parser("extrair")
    x.add_argument("--modelo", required=True, help="repo@revisao")
    x.add_argument("--limiar", type=float, required=True)
    x.add_argument("--limite", type=int, help="só N documentos por conjunto")
    x.add_argument("--sem-upload", action="store_true")
    s = sub.add_parser("estatisticas")
    s.add_argument("--pasta", required=True)
    a = ap.parse_args()
    return {"treinar": cmd_treinar, "extrair": cmd_extrair, "estatisticas": cmd_estatisticas}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
