# -*- coding: utf-8 -*-
"""Bancos de frases do gerador da organização, extraídos do dev set.

Achado de 22/09 (ver wiki/experimentos): os 26 documentos da organização saem de
um gerador por moldes — 62% das frases sem citação se repetem entre documentos e
as 192 citações usam só 83 frases carregadoras. Estimativa Chao1: ~119
carregadoras e ~438 frases de enchimento no banco inteiro deles. Este módulo
desmonta cada documento nas peças desse gerador:

    preâmbulo (cabeçalho + título + abertura)      termina no primeiro "\\n\\n\\n"
    parágrafo de introdução
    (título de seção, parágrafo) × k               "I — DA CONTROVÉRSIA"
    fecho                                          pedido, "Termos em que...", local e data

Cada parágrafo vira uma sequência de frases: NARRATIVA (fatos do caso, no início
do parágrafo), ENCHIMENTO (argumentação genérica, sem citação) ou CARREGADORA
(frase com uma citação, que vira molde com o slot {CIT} e o artigo em concordância).

Números que são distratores (CNJ do cabeçalho, OAB, fls., datas) viram slots e são
sorteados de novo em cada documento gerado — o modelo não pode decorar distrator.
"""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import re

_SECAO = re.compile(r"^[IVX]+\s+[—–-]\s+\S")
_FRASE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ\"“])")

# Artigo/contração antes da citação -> forma masculina canônica do slot.
# O molde guarda "{o}" e a montagem escolhe o/a, no/na, do/da, ao/à, pelo/pela.
_ARTIGOS = {
    "o": "o", "a": "o", "no": "no", "na": "no", "do": "do", "da": "do",
    "ao": "ao", "à": "ao", "pelo": "pelo", "pela": "pelo",
    "os": "o", "as": "o", "nos": "no", "nas": "no",
}
FEMININO = {"o": "a", "no": "na", "do": "da", "ao": "à", "pelo": "pela"}


def _gabarito(dados: pathlib.Path) -> dict:
    g = collections.defaultdict(list)
    for r in csv.DictReader(open(dados / "goldenset_offsets.csv", encoding="utf-8-sig")):
        g[r["documento_id"]].append((int(r["inicio"]), int(r["fim"]), r["tipo"], r["classificacao"]))
    return g


def rotulo_de(tipo: str, classe: str) -> str:
    if classe == "incompleta":
        return "VAGA"
    return "LEI" if tipo == "lei" else "JURIS"


# ------------------------------------------------------------------ slots

_MESES = ("janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|"
          "novembro|dezembro")
_SLOTS = [
    (re.compile(r"\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b"), "{CNJ}"),
    (re.compile(r"\b\d{1,2}º? de (?:" + _MESES + r") de \d{4}\b"), "{DATA}"),
    (re.compile(r"\bfls?\.\s*\d+(?:/\d+)?"), "{FLS}"),
    (re.compile(r"\bOAB/[A-Z]{2}\s*\d+"), "{OAB}"),
    (re.compile(r"\b(?:19|20)\d{2}\.\d{5,8}\b"), "{PROTOCOLO}"),
]


def com_slots(texto: str) -> str:
    for rx, slot in _SLOTS:
        texto = rx.sub(slot, texto)
    return texto


# ------------------------------------------------------------------ desmonte

def _frases(paragrafo: str) -> list[str]:
    plano = re.sub(r"\s+", " ", paragrafo).strip()
    return [f for f in _FRASE.split(plano) if f]


def _carregadora(frase: str) -> tuple[str, str] | None:
    """'Vale invocar o {CIT}, de clareza...' -> ('Vale invocar {o} {CIT}, de ...', 'o')."""
    m = re.search(r"\b(\w+)\s+\{CIT\}", frase)
    if not m or m.group(1).lower() not in _ARTIGOS:
        return None
    art = _ARTIGOS[m.group(1).lower()]
    maiusc = m.group(1)[0].isupper()
    slot = "{" + ("O" if maiusc else "") + art + "}"
    return frase[:m.start(1)] + slot + " " + frase[m.end(1):].lstrip(), art


def desmontar(texto: str, spans: list) -> dict:
    """Um documento da organização -> peças do gerador."""
    # Troca cada citação por um marcador que sobrevive à quebra em frases.
    marcado = texto
    rotulos = []
    for a, b, tipo, classe in sorted(spans, reverse=True):
        marcado = marcado[:a] + "{CIT}" + marcado[b:]
        rotulos.insert(0, rotulo_de(tipo, classe))
    corte = marcado.find("\n\n\n")
    preambulo, corpo = marcado[:corte], marcado[corte + 3:]
    blocos = [b.strip("\n") for b in re.split(r"\n\s*\n", corpo) if b.strip()]
    secoes, paragrafos, fecho = [], [], []
    i = 0
    paragrafos.append(blocos[0])                  # introdução
    i = 1
    while i < len(blocos) and _SECAO.match(blocos[i]):
        secoes.append(blocos[i])
        paragrafos.append(blocos[i + 1] if i + 1 < len(blocos) else "")
        i += 2
    fecho = blocos[i:]
    return {"preambulo": preambulo, "secoes": secoes, "paragrafos": paragrafos,
            "fecho": fecho, "rotulos": rotulos, "termina_com_quebra": texto.endswith("\n")}


def _materia(texto: str) -> str:
    t = texto.lower()
    pontos = {
        "militar": t.count("militar"),
        "eleitoral": t.count("eleitora") + t.count("eleição") + t.count("eleições"),
        "trabalhista": t.count("reclamant") + t.count("reclamad") + t.count("trabalh"),
        "penal": t.count("denúncia") + t.count("acusado") + t.count("pena ") + t.count("réu"),
    }
    melhor = max(pontos, key=pontos.get)
    return melhor if pontos[melhor] >= 2 else "civel"


def _rx_carregadora(molde: str) -> re.Pattern:
    """Molde -> regex que casa a mesma frase com QUALQUER coisa no slot."""
    partes = re.split(r"(\{O?\w+\}\s*\{CIT\})", molde)
    rx = ""
    for p in partes:
        if re.fullmatch(r"\{O?\w+\}\s*\{CIT\}", p):
            rx += r"(?:[OoAaNnDdPp]\w{0,4}|[Àà])\s+(?P<ref>.+?)"
        else:
            rx += re.escape(p)
    return re.compile("^" + rx + "$")


def _similar(a: str, b: str) -> bool:
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio() > 0.9


def extrair(dados: pathlib.Path, excluir: set | None = None) -> dict:
    """Varre o dev set e devolve os bancos (JSON-serializáveis).

    Três tipos de frase sem citação, separados por evidência medida em 22/09:
    * CARREGADORA COM REFERÊNCIA GENÉRICA — "veja-se o verbete sumular aplicável
      à espécie": mesmo molde das citações, slot preenchido sem fonte. São as
      "vagas sem fonte" que a organização tirou do gabarito: negativos difíceis.
    * NARRATIVA — fatos do caso: vista em UMA matéria só.
    * ENCHIMENTO — argumentação genérica: vista em 2+ matérias.
    Cada entrada guarda os níveis de origem: entrada só vista no nível 2 pode
    carregar ruído de OCR e só entra em documento de nível 2.
    """
    excluir = excluir or set()
    gab = _gabarito(dados)
    generos, carregadoras, sequencias = [], [], []
    frases = collections.defaultdict(lambda: {"docs": set(), "mats": set(), "niveis": set(),
                                              "inicio": 0})
    for arq in sorted((dados / "txt").glob("*.txt")):
        doc = arq.stem
        if doc in excluir:
            continue
        nivel = 2 if "_n2_" in doc else 1
        texto = arq.read_text(encoding="utf-8")
        peca = desmontar(texto, gab[doc])
        materia = _materia(texto)
        generos.append({"doc": doc, "materia": materia, "nivel": nivel,
                        "preambulo": com_slots(peca["preambulo"]),
                        "secoes": peca["secoes"],
                        "fecho": [com_slots(b) for b in peca["fecho"]]})
        k = 0
        for j, par in enumerate(peca["paragrafos"]):
            if j > 0:
                sequencias.append((materia, nivel, [com_slots(x) for x in _frases(par)]))
            for i, fr in enumerate(_frases(par)):
                n = fr.count("{CIT}")
                if n:
                    rot = peca["rotulos"][k]
                    k += n
                    c = _carregadora(fr) if n == 1 else None
                    if c:
                        carregadoras.append({"molde": com_slots(c[0]), "rotulo": rot,
                                             "doc": doc, "nivel": nivel})
                    continue
                f = frases[com_slots(fr)]
                f["docs"].add(doc)
                f["mats"].add(materia)
                f["niveis"].add(nivel)
                f["inicio"] += i == 0 and j > 0

    moldes = {}
    for c in carregadoras:
        moldes.setdefault(c["molde"], (_rx_carregadora(c["molde"]), c["rotulo"]))
    genericas, narrativas, enchimento = [], [], []
    for fr, f in frases.items():
        ent = {"texto": fr, "niveis": sorted(f["niveis"]), "peso": len(f["docs"])}
        casou = next(((m, rot) for m, rot in ((rx.match(fr), rot) for rx, rot in moldes.values())
                      if m), None)
        if casou:
            # O artigo original diz o gênero da referência ("a jurisprudência").
            art = re.search(r"(\w+)\s+" + re.escape(casou[0].group("ref")), fr)
            fem = bool(art) and art.group(1).lower() in ("a", "na", "da", "à", "pela", "as", "nas", "das")
            genericas.append({**ent, "ref": casou[0].group("ref"), "genero": "f" if fem else "m",
                              "rotulo": "LEI" if casou[1] == "LEI" else "JURIS"})
            # O molde da frase genérica também é carregadora: a organização sorteia o
            # molde e depois o conteúdo (citação ou referência sem fonte).
            if art:
                c = _carregadora(fr[:art.start(1)] + art.group(1) + " {CIT}" +
                                 fr[art.end(0):])
                if c:
                    carregadoras.append({"molde": c[0], "rotulo": "LEI" if casou[1] == "LEI" else "JURIS",
                                         "doc": "generica", "nivel": min(f["niveis"])})
        elif len(f["mats"]) == 1:
            # Frase de uma matéria só é fato do caso ("O parecer conclusivo opinou
            # pela desaprovação" é eleitoral): nunca entra em documento de outra.
            narrativas.append({**ent, "materia": next(iter(f["mats"])), "abre": f["inicio"] > 0})
        else:
            enchimento.append(ent)

    # Variante ruidosa (só nível 2) de uma frase limpa já presente: descarta.
    def _dedup(lista):
        limpas = [e["texto"] for e in lista if 1 in e["niveis"]]
        return [e for e in lista if 1 in e["niveis"]
                or not any(_similar(e["texto"], l) for l in limpas)]
    # Blocos narrativos: a corrida inicial de frases narrativas de cada parágrafo,
    # NA ORDEM do original — o classificador adversarial denunciava a ordem trocada
    # ("reajustável ajuste": fim de uma frase + começo da seguinte).
    narr_txt = {n["texto"] for n in narrativas}
    blocos, vistos = [], set()
    for materia, nivel, seq in sequencias:
        run = []
        for fr in seq:
            if fr in narr_txt:
                run.append(fr)
            else:
                break
        if run and tuple(run) not in vistos:
            vistos.add(tuple(run))
            blocos.append({"materia": materia, "frases": run, "niveis": [nivel]})
    return {
        "generos": generos,
        "blocos_narrativos": blocos,
        "narrativas": _dedup(narrativas),
        "enchimento": _dedup(enchimento),
        "genericas": _dedup(genericas),
        "carregadoras": carregadoras,
    }


def dobras_do_dev(dados: pathlib.Path, k: int = 2, semente: int = 0) -> list[set]:
    """Partição fixa dos documentos da org em k dobras (a mesma da validação adversarial)."""
    import random
    docs = sorted(p.stem for p in (dados / "txt").glob("*.txt"))
    random.Random(semente).shuffle(docs)
    return [set(docs[d::k]) for d in range(k)]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default="/app/dados")
    ap.add_argument("--saida", default="/app/corpus/bancos/org.json")
    ap.add_argument("--excluir", default="", help="documentos fora dos bancos (ablação honesta)")
    a = ap.parse_args()
    b = extrair(pathlib.Path(a.dados), set(filter(None, a.excluir.split(","))))
    out = pathlib.Path(a.saida)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")
    rot = collections.Counter(c["rotulo"] for c in b["carregadoras"])
    print(f"generos {len(b['generos'])} | narrativas {len(b['narrativas'])} | "
          f"enchimento {len(b['enchimento'])} | genericas {len(b['genericas'])} | "
          f"carregadoras {len(b['carregadoras'])} "
          f"({len({c['molde'] for c in b['carregadoras']})} únicas) {dict(rot)} -> {out}")


if __name__ == "__main__":
    main()
