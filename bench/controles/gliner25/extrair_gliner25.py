# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gliner2[local]==2.0.0",
#   "gliner==0.2.29",
#   "torch==2.14.0",
#   "transformers==4.57.6",
#   "sentencepiece",
#   "protobuf",
#   "huggingface_hub",
# ]
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cu126" }
# ///
# -*- coding: utf-8 -*-
"""GLiNER 2.5 multi em zero-shot, no lugar do GLiNER multi v2.1 do benchmark.

Modelo: `fastino/gliner2.5-multi-v1`@2ca71aa (287M, mDeBERTa-v3-base, Apache-2.0), biblioteca
`gliner2==2.0.0`, carregado por `AutoExtractor` (arquitetura "boundary"). A revisão é baixada com
`snapshot_download` e o modelo carrega da pasta local: `from_pretrained(<repo>, revision=...)`
fixa a revisão dos pesos, mas o tokenizer da biblioteca carrega do id do repositório sem revisão.
Pesos em fp32 (o padrão, sem `quantize`), política de sobreposição padrão do checkpoint ("flat",
por rótulo), 1 GPU L4.

VARIANTE PRINCIPAL (`zs`), para comparar com a v2.1: os MESMOS quatro nomes de rótulo em
português, o MESMO limiar 0,5 e as MESMAS janelas de 200 palavras com passo 150 de
`bench/extrair_gliner.py`, com a mesma fusão entre janelas (na sobreposição fica o span de maior
score, qualquer rótulo). O contexto da 2.5 (max_len 4096) comporta a janela sem adaptação.
Diferença de execução declarada: as janelas de um documento vão num lote só
(`batch_extract_entities`), como o GLiNER fine-tunado fez; `--conferir-lote` mede num amostra se o
lote muda algum span em relação a janela por janela.

VARIANTE SECUNDÁRIA: no máximo três tentativas, fixadas aqui antes de qualquer saída da 2.5 e
rodadas só nas 305 (validação). A 2.5 aceita descrição por rótulo; as descrições saem das
definições e dos exemplos genéricos do prompt do Qwen3-8B do benchmark (`bench/extrair_qwen.py`),
escrito uma vez sem olhar os conjuntos avaliados.
    t1_desc_050  nomes + descrições, limiar 0,5
    t2_desc_030  nomes + descrições, limiar 0,3
    t3_desc_070  nomes + descrições, limiar 0,7
Critério de escolha (nas 305, pelo harness `bench.controles.avaliar`, só o conjunto `reais`):
maior F1 de extração cru (IoU >= 0,5, VAGA em JURIS); empate -> maior F1 exato; depois o limiar
mais perto de 0,5. A escolhida roda depois nos três conjuntos, noutro job. O limiar não pode ser
aplicado filtrando uma extração de limiar menor: a resolução "flat" escolhe o conjunto sem
sobreposição de maior score total, então cada limiar é uma passada.

REFERÊNCIA v2.1 (`--v21`): o laço exato de `bench/extrair_gliner.py` (gliner 0.2.29,
`urchade/gliner_multi-v2.1`@443d26d), para ter a v2.1 zero-shot nas 172 (o arquivo do benchmark
só cobre estresse + 305). Rodar também nas 305 confere a reprodução contra `saidas/bench/gliner.jsonl`.

Saída: um JSONL por variante no formato dos controles (1ª linha `meta`; depois
{conjunto, id, segundos, spans: [[inicio, fim, JURIS|LEI|VAGA, score]]}, offsets em codepoints do
campo `texto`), `segundos` medido com `torch.cuda.synchronize()` antes e depois. Sobe para
vinimlo/gama-goldenset em bench/saida/controles/gliner25/ e imprime `REVISAO <arquivo> <oid>`.

    # job A: principal nos três conjuntos + as três tentativas nas 305 + v2.1 nas 172 e 305
    hf jobs uv run --flavor l4x1 --timeout 30m --secrets HF_TOKEN \\
        bench/controles/gliner25/extrair_gliner25.py \\
        --variante zs=estresse,reais,novas --variante t1_desc_050=reais \\
        --variante t2_desc_030=reais --variante t3_desc_070=reais --v21 reais,novas --conferir-lote 20
    # job B: a tentativa escolhida nas 305, nos três conjuntos
    hf jobs uv run --flavor l4x1 --timeout 20m --secrets HF_TOKEN \\
        bench/controles/gliner25/extrair_gliner25.py --variante <escolhida>=estresse,reais,novas

Copiado de `bench/extrair_gliner.py` (o job só recebe este arquivo): ROTULOS, LIMIAR, PALAVRAS,
PASSO, `janelas` e o laço de `extrair` (este em `extrair_v21`, idêntico, e com a fusão isolada em
`fundir`). `bench/controles/gliner25/conferir.py` confere as cópias no container.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import time

# ------------------------------------------------------------------ copiado de bench/extrair_gliner.py
MODELO_V21 = "urchade/gliner_multi-v2.1"
REVISAO_V21 = "443d26d654e0324125a96bebd8e796c14ff2efe6"
ROTULOS = {
    "precedente judicial com número": "JURIS",
    "súmula ou tema de tribunal": "JURIS",
    "artigo de lei": "LEI",
    "referência a julgado sem número": "VAGA",
}
LIMIAR = 0.5
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


def extrair_v21(modelo, texto: str) -> list[list]:
    """`bench.extrair_gliner.extrair`, sem mudança."""
    achados = []
    for a, b in janelas(texto):
        for e in modelo.predict_entities(texto[a:b], list(ROTULOS), threshold=LIMIAR):
            achados.append((a + e["start"], a + e["end"], ROTULOS[e["label"]], float(e["score"])))
    return fundir(achados)
# ------------------------------------------------------------------ fim do trecho copiado

MODELO = "fastino/gliner2.5-multi-v1"
REVISAO = "2ca71aafb3446d9014e1c55c7ff51c9bc7209c47"
DADOS = "vinimlo/gama-goldenset"
REV_BENCH = "2fff5f670e13f77f339937cfe3da52ed0af9572d"    # entradas do benchmark
EXPERIMENTO = "gliner25"
LOTE_MAX = 32                                              # janelas por chamada

# Descrições por rótulo: definições e exemplos genéricos do prompt de `bench/extrair_qwen.py`.
DESCRICOES = {
    "precedente judicial com número": (
        "Precedente identificado por número de processo ou de recurso, com a classe e a UF no fim "
        "do número quando houver. Ex.: REsp nº 1.234.567/SP, AgInt no AREsp 123.456/RJ, processo "
        "nº 0001234-56.2020.8.26.0000."),
    "súmula ou tema de tribunal": (
        "Súmula ou tema identificado por número, com o tribunal. Ex.: Súmula 7 do STJ, Tema 1.046 "
        "da repercussão geral."),
    "artigo de lei": (
        "Dispositivo de lei inteiro, do artigo até o nome do diploma. Ex.: art. 5º, X, da "
        "Constituição Federal; artigo 927 do CPC; art. 14 da Lei nº 8.078/1990."),
    "referência a julgado sem número": (
        "Julgado citado sem número, identificado por tribunal, ano e relator, terminando no nome "
        "do relator. Ex.: precedente do STJ julgado em 2021 sob a relatoria de Nancy Andrighi. "
        "Não inclui referência genérica sem fonte, como a jurisprudência pacífica dos tribunais "
        "superiores."),
}
VARIANTES = {
    "zs": {"descricoes": False, "limiar": LIMIAR},
    "t1_desc_050": {"descricoes": True, "limiar": 0.5},
    "t2_desc_030": {"descricoes": True, "limiar": 0.3},
    "t3_desc_070": {"descricoes": True, "limiar": 0.7},
}
CONJUNTOS = ("estresse", "reais", "novas")


def esquema(descricoes: bool):
    """Rótulos na ordem de ROTULOS: lista de nomes, ou nome -> descrição."""
    return {r: DESCRICOES[r] for r in ROTULOS} if descricoes else list(ROTULOS)


class Extrator25:
    """O laço de `extrair` com a 2.5: janelas de 200 palavras, offsets somados ao início da
    janela, fusão pelo maior score. As janelas de um documento vão num lote (ou uma a uma)."""

    def __init__(self, modelo, descricoes: bool, limiar: float):
        self.modelo, self.rotulos, self.limiar = modelo, esquema(descricoes), limiar
        self.texto_diferente = 0          # e["text"] != texto[start:end].strip() (tem que ficar 0)

    def _janelas(self, pedacos: list[str], lote: bool) -> list[dict]:
        kw = dict(threshold=self.limiar, include_confidence=True, include_spans=True)
        if lote:
            out = []
            for k in range(0, len(pedacos), LOTE_MAX):
                bloco = pedacos[k:k + LOTE_MAX]
                out += self.modelo.batch_extract_entities(bloco, self.rotulos, batch_size=len(bloco), **kw)
            return out
        return [self.modelo.extract_entities(p, self.rotulos, **kw) for p in pedacos]

    def extrair(self, texto: str, lote: bool = True) -> list[list]:
        js = janelas(texto)
        achados = []
        for (a, b), r in zip(js, self._janelas([texto[a:b] for a, b in js], lote)):
            for nome, ents in (r.get("entities") or {}).items():
                for e in ents:
                    s, f = a + int(e["start"]), a + int(e["end"])
                    self.texto_diferente += texto[s:f].strip() != e["text"]
                    achados.append((s, f, ROTULOS[nome], float(e["confidence"])))
        return fundir(achados)


def ler_entradas(raiz: pathlib.Path) -> list[dict]:
    docs = []
    for arq in ("entrada_remota.jsonl", "entrada_novas.jsonl"):      # estresse + reais, novas
        docs += [json.loads(x) for x in (raiz / "bench" / arq).read_text(encoding="utf-8").splitlines()]
    return docs


def rodar(nome: str, extrair, docs: list[dict], meta: dict, destino: pathlib.Path) -> dict:
    """Extrai `docs` com `extrair(texto)`, grava o JSONL e devolve s/doc por conjunto."""
    import torch
    tempos = collections.defaultdict(list)
    n_spans = collections.Counter()
    with destino.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for k, d in enumerate(docs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            spans = extrair(d["texto"])
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            tempos[d["conjunto"]].append(dt)
            for s in spans:
                n_spans[(d["conjunto"], s[2])] += 1
            fh.write(json.dumps({"conjunto": d["conjunto"], "id": d["id"], "segundos": round(dt, 4),
                                 "spans": spans}, ensure_ascii=False) + "\n")
            if k % 200 == 0:
                print(f"{nome} {k}/{len(docs)}", flush=True)
    res = {"s_por_doc": {c: round(sum(v) / len(v), 4) for c, v in tempos.items()},
           "docs": {c: len(v) for c, v in tempos.items()},
           "s_total": round(sum(sum(v) for v in tempos.values()), 1),
           "spans": {f"{c}/{r}": n for (c, r), n in sorted(n_spans.items())}}
    print(nome, "EXTRACAO", json.dumps(res, ensure_ascii=False), flush=True)
    return res


def subir(api, arq: pathlib.Path, mensagem: str) -> str:
    info = api.upload_file(path_or_fileobj=str(arq), repo_id=DADOS, repo_type="dataset",
                           path_in_repo=f"bench/saida/controles/{EXPERIMENTO}/{arq.name}",
                           commit_message=mensagem)
    print("REVISAO", arq.name, info.oid, flush=True)
    return info.oid


def conjuntos_de(txt: str) -> list[str]:
    cs = [c for c in txt.split(",") if c]
    if not cs or any(c not in CONJUNTOS for c in cs):
        raise SystemExit(f"conjuntos inválidos: {txt}")
    return cs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variante", action="append", default=[],
                    help="nome=conjuntos, ex.: zs=estresse,reais,novas (nomes em VARIANTES)")
    ap.add_argument("--v21", default="", help="conjuntos da referência v2.1, ex.: reais,novas")
    ap.add_argument("--conferir-lote", type=int, default=0,
                    help="documentos em que se compara lote x janela a janela (variante zs)")
    ap.add_argument("--limite", type=int, help="só os N primeiros documentos de cada conjunto (fumaça)")
    ap.add_argument("--sem-upload", action="store_true")
    a = ap.parse_args()
    pedidos = []
    for v in a.variante:
        nome, _, cs = v.partition("=")
        if nome not in VARIANTES:
            raise SystemExit(f"variante desconhecida: {nome}")
        pedidos.append((nome, conjuntos_de(cs)))

    t_ini = time.time()
    import gliner2
    import torch
    import transformers
    from gliner2 import AutoExtractor
    from huggingface_hub import HfApi, snapshot_download

    torch.manual_seed(0)
    raiz_b = pathlib.Path(snapshot_download(DADOS, repo_type="dataset", revision=REV_BENCH, allow_patterns=[
        "bench/entrada_remota.jsonl", "bench/entrada_novas.jsonl"]))
    todos = ler_entradas(raiz_b)
    por_conj = collections.defaultdict(list)
    for d in todos:
        por_conj[d["conjunto"]].append(d)
    if a.limite:
        por_conj = {c: v[: a.limite] for c, v in por_conj.items()}
    print("documentos", {c: len(v) for c, v in por_conj.items()}, flush=True)

    pasta = snapshot_download(MODELO, revision=REVISAO)
    t0 = time.time()
    modelo = AutoExtractor.from_pretrained(pasta, map_location="cuda")
    modelo.eval()
    carga = round(time.time() - t0, 1)
    base = {"extrator": EXPERIMENTO, "modelo": f"{MODELO}@{REVISAO[:7]}",
            "revisao_completa": REVISAO, "classe": type(modelo).__name__,
            "arquitetura": getattr(modelo.config, "architecture", None),
            "parametros": sum(p.numel() for p in modelo.parameters()),
            "dtype": str(next(modelo.parameters()).dtype), "gliner2": gliner2.__version__,
            "transformers": transformers.__version__, "torch": torch.__version__,
            "dispositivo": torch.cuda.get_device_name(0), "carga_s": carga,
            "janelas": f"{PALAVRAS} palavras, passo {PASSO}", "lote_janelas": f"um documento por chamada, até {LOTE_MAX}",
            "overlap_policy": "padrão do checkpoint (flat, por rótulo)", "rotulos": ROTULOS,
            "entradas": f"{DADOS}@{REV_BENCH[:7]}"}
    print(json.dumps(base, ensure_ascii=False), flush=True)

    # Fumaça: uma frase com os três tipos, nos dois esquemas (sai no log).
    frase = ("Conforme o REsp nº 1.234.567/SP e a Súmula 7 do STJ, aplica-se o art. 927 do Código Civil, "
             "como decidiu o STJ em 2021 sob a relatoria de Nancy Andrighi.")
    for desc in (False, True):
        r = modelo.extract_entities(frase, esquema(desc), threshold=0.3, include_confidence=True, include_spans=True)
        print("FUMACA", "descricoes" if desc else "nomes", json.dumps(r, ensure_ascii=False, default=float), flush=True)

    # Tamanho da janela em subtokens do tokenizer do modelo (sem o esquema), para declarar o contexto.
    try:
        tok = modelo.processor.tokenizer
        tams = sorted(len(tok(d["texto"][x:y], add_special_tokens=False)["input_ids"])
                      for c in por_conj.values() for d in c for x, y in janelas(d["texto"]))
        base["subtokens_por_janela"] = {"janelas": len(tams), "mediana": tams[len(tams) // 2],
                                        "p99": tams[int(0.99 * (len(tams) - 1))], "max": tams[-1],
                                        "max_len_modelo": getattr(modelo.config, "max_len", None)}
    except Exception as exc:                                  # nunca derruba o job
        base["subtokens_por_janela"] = f"não medido: {exc!r}"
    print("SUBTOKENS", json.dumps(base["subtokens_por_janela"]), flush=True)

    # Aquecimento (fora da medida de tempo).
    Extrator25(modelo, False, LIMIAR).extrair(" ".join([frase] * 60))
    torch.cuda.synchronize()

    if a.conferir_lote:
        ext = Extrator25(modelo, False, LIMIAR)
        pool = [d for c in CONJUNTOS for d in por_conj.get(c, [])]
        amostra = pool[:: max(1, len(pool) // a.conferir_lote)][: a.conferir_lote]
        dif_spans = dif_score = 0
        max_dif = 0.0
        for d in amostra:
            x, y = ext.extrair(d["texto"], lote=True), ext.extrair(d["texto"], lote=False)
            dif_spans += [s[:3] for s in x] != [s[:3] for s in y]
            if [s[:3] for s in x] == [s[:3] for s in y]:
                m = max((abs(p[3] - q[3]) for p, q in zip(x, y)), default=0.0)
                dif_score += m > 0
                max_dif = max(max_dif, m)
        base["conferencia_lote"] = {"docs": len(amostra), "spans_diferentes": dif_spans,
                                    "so_score_diferente": dif_score, "max_dif_score": round(max_dif, 6)}
        print("CONFERENCIA_LOTE", json.dumps(base["conferencia_lote"]), flush=True)

    api = None if a.sem_upload else HfApi()
    resumo = {}
    for nome, cs in pedidos:
        cfg = VARIANTES[nome]
        ext = Extrator25(modelo, cfg["descricoes"], cfg["limiar"])
        meta = {**base, "variante": nome, "limiar_decisao": cfg["limiar"],
                "descricoes": DESCRICOES if cfg["descricoes"] else None, "conjuntos": cs}
        destino = pathlib.Path(f"/tmp/modelo_gliner25_{nome}.jsonl")
        docs = [d for c in cs for d in por_conj.get(c, [])]
        resumo[nome] = rodar(nome, ext.extrair, docs, meta, destino)
        resumo[nome]["texto_diferente"] = ext.texto_diferente
        print(nome, "texto_diferente", ext.texto_diferente, flush=True)
        if api:
            resumo[nome]["revisao"] = subir(api, destino, f"controles: spans do GLiNER 2.5 zero-shot ({nome})")

    if a.v21:
        import gliner
        from gliner import GLiNER
        cs = conjuntos_de(a.v21)
        m21 = GLiNER.from_pretrained(MODELO_V21, revision=REVISAO_V21).to("cuda").eval()
        meta = {"extrator": "gliner", "modelo": f"{MODELO_V21}@{REVISAO_V21[:7]}", "gliner": gliner.__version__,
                "transformers": transformers.__version__, "torch": torch.__version__,
                "dispositivo": torch.cuda.get_device_name(0), "limiar_decisao": LIMIAR,
                "janelas": f"{PALAVRAS} palavras, passo {PASSO}", "rotulos": ROTULOS, "conjuntos": cs,
                "nota": "o laço de bench/extrair_gliner.py; referência para as 172 e conferência nas 305"}
        with torch.inference_mode():
            extrair_v21(m21, " ".join([frase] * 60))
            destino = pathlib.Path("/tmp/modelo_gliner21_ref.jsonl")
            docs = [d for c in cs for d in por_conj.get(c, [])]
            resumo["v21_ref"] = rodar("v21_ref", lambda t: extrair_v21(m21, t), docs, meta, destino)
        if api:
            resumo["v21_ref"]["revisao"] = subir(api, destino, "controles: spans do GLiNER v2.1 zero-shot "
                                                               "(referência nas 172 e conferência nas 305)")
    print("RESUMO", json.dumps(resumo, ensure_ascii=False), flush=True)
    print(f"FIM {round((time.time() - t_ini) / 60, 1)} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
