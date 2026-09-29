# -*- coding: utf-8 -*-
"""H5: auditoria dos trechos do Qwen3-8B que não alinharam ao texto (diagnóstico, nada é corrigido).

Protocolo fixado antes de ver os trechos: `bench/controles/offline/protocolo_auditoria_qwen.json`.

    preparar  confere a rerodada contra saidas/bench/qwen.jsonl, monta a população de trechos não
              alinhados, sorteia 120 (semente 20260929) e calcula as pistas de cada um (sem decidir) ->
              saidas/bench/controles/offline/amostra_qwen.jsonl
    compilar  junta as categorias por código (formato, ocorrência) com as do revisor
              (bench/controles/offline/rotulos_qwen.json) -> bench/controles/offline/auditoria_qwen.json

    docker compose run --rm gama python -m bench.controles.offline.auditoria_qwen preparar
    docker compose run --rm gama python -m bench.controles.offline.auditoria_qwen compilar

Quem decide se um item alinhou é a própria `bench.extrair_qwen.alinhar` (importada): o item i ficou
de fora se `alinhar(texto, citacoes[:i+1])` perde um a mais que `alinhar(texto, citacoes[:i])`.
Um item que alinha sozinho (`alinhar(texto, [item])`) mas não na sequência existe no texto e caiu
pela política de ocorrência; um que não alinha nem sozinho não existe literalmente no texto.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import pathlib
import random
import re
import unicodedata

from bench import extrair_qwen as Q
from bench.controles import avaliar as A

APP = pathlib.Path("/app")
AQUI = APP / "bench" / "controles" / "offline"
SAIDA = APP / "saidas" / "bench" / "controles" / "offline"
HUB = APP / "saidas" / "bench" / "controles" / "_hub" / "bench" / "saida" / "controles" / "offline"
PROTOCOLO = json.loads((AQUI / "protocolo_auditoria_qwen.json").read_text(encoding="utf-8"))
TAMANHO, SEMENTE = PROTOCOLO["amostra"]["tamanho"], PROTOCOLO["amostra"]["semente"]
CATEGORIAS = [c["id"] for c in PROTOCOLO["categorias"]]
EXEMPLOS_PROMPT = re.findall(r'"([^"]+)"', Q.INSTRUCAO.split("Bordas:")[0])
OCR = str.maketrans({"l": "1", "i": "1", "o": "0", "s": "5", "g": "9", "b": "8", "z": "2", "t": "7"})


def _ler(arq: pathlib.Path) -> tuple[dict, dict]:
    meta, linhas = {}, {}
    for x in arq.read_text(encoding="utf-8").splitlines():
        r = json.loads(x)
        if "meta" in r:
            meta = r["meta"]
        else:
            linhas[(r["conjunto"], r["id"])] = r
    return meta, linhas


def _normalizar(s: str, ocr: bool = False) -> tuple[str, list[int]]:
    """Sem acento, casefold, º°->o ª->a, só letras e dígitos; devolve o mapa para os offsets."""
    out, mapa = [], []
    for i, ch in enumerate(s):
        ch = {"º": "o", "°": "o", "ª": "a"}.get(ch, ch)
        for c in unicodedata.normalize("NFKD", ch):
            if unicodedata.combining(c) or not c.isalnum():
                continue
            c = c.casefold()
            out.append(c.translate(OCR) if ocr else c)
            mapa.append(i)
    return "".join(out), mapa


def _achar_normalizado(texto: str, trecho: str) -> dict | None:
    for ocr in (False, True):
        nt, mapa = _normalizar(texto, ocr)
        nq, _ = _normalizar(trecho, ocr)
        if len(nq) < 3:
            return None
        k = nt.find(nq)
        if k >= 0:
            return {"nivel": "normalizado+ocr" if ocr else "normalizado", "inicio": mapa[k],
                    "fim": mapa[k + len(nq) - 1] + 1}
    return None


def _aproximado(texto: str, trecho: str) -> dict | None:
    """Melhor janela do texto normalizado por difflib (razão), com offsets originais."""
    nt, mapa = _normalizar(texto)
    nq, _ = _normalizar(trecho)
    if len(nq) < 3 or not nt:
        return None
    melhor = (0.0, 0, 0)
    for largura in {len(nq), int(len(nq) * 0.8), int(len(nq) * 1.25)}:
        largura = max(3, min(largura, len(nt)))
        sm = difflib.SequenceMatcher(None, "", nq, autojunk=False)   # b fixo (o trecho), a = janela
        for k in range(0, len(nt) - largura + 1):
            sm.set_seq1(nt[k:k + largura])
            if sm.real_quick_ratio() <= melhor[0] or sm.quick_ratio() <= melhor[0]:
                continue
            r = sm.ratio()
            if r > melhor[0]:
                melhor = (r, k, k + largura)
    r, a, b = melhor
    if b <= a:
        return None
    return {"razao": round(r, 3), "inicio": mapa[a], "fim": mapa[b - 1] + 1}


def _ouro_que_cruza(ouro: list, a: int, b: int) -> list:
    return [list(g) for g in ouro if g[0] < b and a < g[1]]


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- preparar

def reproducao(rerun: dict, publicado: dict) -> dict:
    iguais = [k for k in publicado if k in rerun and rerun[k]["spans"] == publicado[k]["spans"]
              and rerun[k]["nao_alinhados"] == publicado[k]["nao_alinhados"]]
    difs = [k for k in publicado if k not in iguais]
    return {"documentos": len(publicado), "identicos": len(iguais), "diferentes": len(difs),
            "diferentes_ids": [f"{c}/{d}" for c, d in difs][:50],
            "nao_alinhados_publicado": sum(r["nao_alinhados"] for r in publicado.values()),
            "nao_alinhados_rerodada": sum(r["nao_alinhados"] for r in rerun.values()),
            "nao_alinhados_em_docs_identicos": sum(publicado[k]["nao_alinhados"] for k in iguais),
            "_iguais": set(iguais)}


def populacao(bruto: dict, textos: dict, docs_validos: set) -> list[dict]:
    """Trechos não alinhados, na ordem da entrada e da lista 'citacoes'."""
    itens = []
    for chave, r in bruto.items():                 # bruto já está na ordem da entrada
        if chave not in docs_validos:
            continue
        texto, cits = textos[chave], r["citacoes"] or []
        antes = 0
        for i, c in enumerate(cits):
            _, perdidos = Q.alinhar(texto, cits[: i + 1])
            if perdidos == antes:
                continue
            antes = perdidos
            trecho, tipo = str(c.get("trecho", "")), str(c.get("tipo", "")).upper()
            if tipo not in Q.TIPOS or not trecho.split():
                causa = "formato_invalido"
            else:
                sozinho, _ = Q.alinhar(texto, [c])
                causa = "ocorrencia_errada_alinhamento" if sozinho else "sem_casamento_literal"
            itens.append({"conjunto": chave[0], "id": chave[1], "posicao": i, "item": c, "causa_codigo": causa})
    return itens


def pistas(it: dict, texto: str, ouro: list, cits: list, spans_alinhados: list) -> dict:
    c = it["item"]
    trecho = str(c.get("trecho", ""))
    p = {"trecho": trecho, "tipo": c.get("tipo"), "no_prompt": trecho.strip() in EXEMPLOS_PROMPT,
         "reticencias": bool(re.search(r"\.\.\.|…", trecho))}
    if it["causa_codigo"] == "ocorrencia_errada_alinhamento":
        rx = re.compile(r"\s+".join(map(re.escape, trecho.split())))
        ocorr = [[m.start(), m.end()] for m in rx.finditer(texto)]
        p["ocorrencias_no_texto"] = ocorr
        p["spans_que_bloqueiam"] = [s for s in spans_alinhados for o in ocorr if s[0] < o[1] and o[0] < s[1]]
        p["repete_item_anterior"] = any(str(x.get("trecho", "")) == trecho for x in cits[: it["posicao"]])
        a, b = ocorr[0]
    else:
        n = _achar_normalizado(texto, trecho) if trecho.strip() else None
        f = _aproximado(texto, trecho) if trecho.strip() and not n else None
        p["normalizado"] = n
        p["aproximado"] = f
        loc = n or (f if f and f["razao"] >= 0.6 else None)
        a, b = (loc["inicio"], loc["fim"]) if loc else (None, None)
    if a is not None:
        p["passagem"] = [a, b]
        p["texto_passagem"] = texto[a:b]
        p["contexto"] = texto[max(0, a - 80):a] + "⟦" + texto[a:b] + "⟧" + texto[b:b + 80]
        p["ouro_que_cruza"] = [[g[0], g[1], g[2], texto[g[0]:g[1]]] for g in _ouro_que_cruza(ouro, a, b)]
    p["sugestao"] = sugerir(it, p)
    return p


def sugerir(it: dict, p: dict) -> str:
    if it["causa_codigo"] in ("formato_invalido", "ocorrencia_errada_alinhamento"):
        return it["causa_codigo"]
    if p["no_prompt"]:
        return "inexistente"
    if "passagem" not in p:
        return "inexistente"
    if p["reticencias"]:
        return "truncado"
    return "reescrita_normalizada" if p["ouro_que_cruza"] else "nao_e_citacao"


def preparar() -> int:
    meta_b, bruto = _ler(HUB / "qwen_bruto.jsonl")
    meta_r, rerun = _ler(HUB / "qwen_rerun.jsonl")
    _, publicado = _ler(A.BENCH / "qwen.jsonl")
    rep = reproducao(rerun, publicado)
    iguais = rep.pop("_iguais")
    # a própria rerodada tem que ser coerente: o registro bruto reproduz o arquivo de spans do main()
    coerente = all(bruto[k]["spans_alinhados"] == rerun[k]["spans"] and bruto[k]["nao_alinhados"] == rerun[k]["nao_alinhados"]
                   for k in rerun)
    textos, ouros = {}, {}
    for c in ("estresse", "reais"):
        t, o = A.textos_e_ouro(c)
        textos.update({(c, d): x for d, x in t.items()})
        ouros.update({(c, d): x for d, x in o.items()})
    pop = populacao(bruto, textos, iguais)
    rng = random.Random(SEMENTE)
    amostra_idx = sorted(rng.sample(range(len(pop)), TAMANHO))
    SAIDA.mkdir(parents=True, exist_ok=True)
    causas_pop = collections.Counter(x["causa_codigo"] for x in pop)
    with (SAIDA / "amostra_qwen.jsonl").open("w", encoding="utf-8") as fh:
        for n, k in enumerate(amostra_idx):
            it = pop[k]
            chave = (it["conjunto"], it["id"])
            r = dict(it, n=n + 1, indice_populacao=k,
                     **pistas(it, textos[chave], ouros[chave], bruto[chave]["citacoes"], bruto[chave]["spans_alinhados"]))
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    resumo = {"reproducao": rep, "rerodada_coerente_com_registro_bruto": coerente, "populacao": len(pop),
              "populacao_por_conjunto": dict(collections.Counter(x["conjunto"] for x in pop)),
              "populacao_por_causa_codigo": dict(causas_pop),
              "populacao_copias_literais_de_exemplo_do_prompt": sum(
                  str(x["item"].get("trecho", "")).strip() in EXEMPLOS_PROMPT for x in pop),
              "populacao_com_reticencias": sum(bool(re.search(r"\.\.\.|…", str(x["item"].get("trecho", "")))) for x in pop),
              "populacao_formato_invalido": [x for x in pop if x["causa_codigo"] == "formato_invalido"],
              "json_invalidos_rerodada": sum(not r["json_ok"] for r in bruto.values()),
              "tokens_saida_max": max(r.get("tokens_saida_reencode", 0) for r in bruto.values()),
              "meta_bruto": meta_b, "meta_rerun": meta_r,
              "arquivos": {"qwen_bruto.jsonl": _sha(HUB / "qwen_bruto.jsonl"), "qwen_rerun.jsonl": _sha(HUB / "qwen_rerun.jsonl"),
                           "qwen.jsonl (publicado)": _sha(A.BENCH / "qwen.jsonl")}}
    (SAIDA / "populacao_qwen.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in resumo.items() if k not in ("meta_bruto", "meta_rerun")}, ensure_ascii=False,
                     indent=1))
    return 0


# ---------------------------------------------------------------- compilar

SUBTIPOS = [  # (subtipo, palavras da nota) — exploratório, não faz parte do protocolo
    ("molde_do_prompt_ou_ambiguo", ("molde do exemplo", "AMBÍGUO")),
    ("caixa", ("caixa", "minúsculas", "maiúsculas")),
    ("quebra_ou_espaco_dentro_do_token", ("quebra", "espaço dentro", "hífen")),
    ("ocr_ou_acento", ("OCR", "acento")),
    ("forma_da_citacao", ("abreviad", "expandid", "contraído", "omitido", "desmembrado", "aspas", "vírgula", "reescrito como", "->")),
]


def _subtipo(nota: str) -> str:
    return next((r for r, chaves in SUBTIPOS if any(c in nota for c in chaves)), "outro")


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """IC95 de Wilson para a fração na população (sem correção de população finita: conservador)."""
    if not n:
        return [0.0, 0.0]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return [round(max(0.0, c - m), 4), round(min(1.0, c + m), 4)]


def compilar() -> int:
    amostra = [json.loads(x) for x in (SAIDA / "amostra_qwen.jsonl").read_text(encoding="utf-8").splitlines()]
    resumo = json.loads((SAIDA / "populacao_qwen.json").read_text(encoding="utf-8"))
    rotulos = json.loads((AQUI / "rotulos_qwen.json").read_text(encoding="utf-8"))["rotulos"]
    itens = []
    for it in amostra:
        chave = str(it["n"])
        if it["causa_codigo"] in ("formato_invalido", "ocorrencia_errada_alinhamento"):
            cat, nota = it["causa_codigo"], rotulos.get(chave, [None, ""])[1] or "decidido por código"
            if chave in rotulos and rotulos[chave][0] != cat:
                raise SystemExit(f"item {chave}: rótulo do revisor contradiz a categoria por código")
        else:
            if chave not in rotulos:
                raise SystemExit(f"item {chave} sem rótulo do revisor")
            cat, nota = rotulos[chave]
        if cat not in CATEGORIAS:
            raise SystemExit(f"item {chave}: categoria {cat} fora do protocolo")
        itens.append({"n": it["n"], "conjunto": it["conjunto"], "id": it["id"], "posicao": it["posicao"],
                      "trecho": it["trecho"], "tipo": it["tipo"], "categoria": cat, "sugestao_automatica": it["sugestao"],
                      "texto_do_documento": it.get("texto_passagem"), "ouro_que_cruza": it.get("ouro_que_cruza"),
                      "nota": nota})
    cont = collections.Counter(x["categoria"] for x in itens)
    por_conj = {c: dict(collections.Counter(x["categoria"] for x in itens if x["conjunto"] == c)) for c in ("estresse", "reais")}
    exemplos = {}
    for cat in CATEGORIAS:
        xs = [x for x in itens if x["categoria"] == cat]
        exemplos[cat] = [{k: x[k] for k in ("n", "conjunto", "id", "trecho", "tipo", "texto_do_documento", "nota")}
                         for x in xs[:2]]
        if not xs and cat == "formato_invalido" and resumo["populacao_formato_invalido"]:
            x = resumo["populacao_formato_invalido"][0]
            exemplos[cat] = [{"n": None, "conjunto": x["conjunto"], "id": x["id"], "trecho": x["item"].get("trecho"),
                              "tipo": x["item"].get("tipo"), "texto_do_documento": None,
                              "nota": "fora da amostra: o único item de formato inválido da população (tipo fora de "
                                      "JURIS/LEI/VAGA); decidido por código"}]
        if not xs and cat == "truncado":
            exemplos[cat] = [{"nota": "nenhum na amostra e nenhum trecho com reticências na população. Esperado: o "
                                      "padrão de alinhar() não exige fronteira de palavra, então um prefixo literal "
                                      "alinha sempre; e nenhuma resposta chegou ao teto de 1.536 tokens (máximo "
                                      f"{resumo['tokens_saida_max']}, reencodado) nem teve JSON inválido"}]
    subtipos = collections.Counter(_subtipo(x["nota"]) for x in itens if x["categoria"] == "reescrita_normalizada")
    n = len(itens)
    reconhecidas = cont["reescrita_normalizada"] + cont["truncado"] + cont["ocorrencia_errada_alinhamento"]
    out = {
        "experimento": "offline/H5",
        "pergunta": PROTOCOLO["pergunta"],
        "protocolo": "bench/controles/offline/protocolo_auditoria_qwen.json (definido antes de ver os trechos)",
        "categorias": PROTOCOLO["categorias"],
        "rerodada": resumo,
        "amostra": {"tamanho": n, "semente": SEMENTE, "populacao": resumo["populacao"]},
        "contagem": {c: cont.get(c, 0) for c in CATEGORIAS},
        "fracao": {c: round(cont.get(c, 0) / n, 4) for c in CATEGORIAS},
        "fracao_ic95_wilson": {c: _wilson(cont.get(c, 0), n) for c in CATEGORIAS},
        "projecao_para_a_populacao": {c: round(cont.get(c, 0) / n * resumo["populacao"]) for c in CATEGORIAS},
        "contagem_por_conjunto": por_conj,
        "leitura": {"interface (o modelo achou uma citação que existe no texto: reescrita, truncada ou "
                    "descartada pela política de ocorrência)": reconhecidas,
                    "reconhecimento (não é citação ou não existe no texto)": cont["nao_e_citacao"] + cont["inexistente"],
                    "formato inválido": cont["formato_invalido"]},
        "concordancia_sugestao_x_final": round(sum(x["categoria"] == x["sugestao_automatica"] for x in itens) / n, 4),
        "exploratorio_subtipos_da_reescrita": {
            "contagem": dict(subtipos.most_common()),
            "como": "definido depois de rotular (exploratório): a primeira regra que casa na nota do revisor, na ordem "
                    + " > ".join(r for r, _ in SUBTIPOS)},
        "exemplos": exemplos,
        "itens": itens,
        "nota": "Diagnóstico: nenhuma previsão foi corrigida e o score publicado do Qwen não muda. Os trechos que falham "
                "continuam sendo erro do sistema avaliado (adaptador incluído).",
    }
    (AQUI / "auditoria_qwen.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("contagem", "contagem_por_conjunto", "leitura", "concordancia_sugestao_x_final")},
                     ensure_ascii=False, indent=1))
    print("->", AQUI / "auditoria_qwen.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=["preparar", "compilar"])
    a = ap.parse_args()
    return preparar() if a.acao == "preparar" else compilar()


if __name__ == "__main__":
    raise SystemExit(main())
