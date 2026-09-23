# -*- coding: utf-8 -*-
"""Montagem de um documento sintético no molde da organização.

Reproduz a estrutura medida no dev set em 22/09:

    preâmbulo            cabeçalho + título + abertura, termina em "\\n\\n\\n"
    introdução           enchimento + 1 carregadora
    seção × k            "I — DA CONTROVÉRSIA" + parágrafo:
                         narrativa (fatos, da matéria do documento) seguida de
                         enchimento e carregadoras intercalados
    fecho                pedido, fórmula, local e data

e a quebra de linha EXATA da organização: a linha termina logo depois da palavra
que a leva a >= 91 caracteres (60/60 parágrafos de prosa do nível 1 reproduzidos
byte a byte). A quebra só troca espaço por "\\n", então os offsets das citações,
calculados antes, continuam valendo — o rótulo é exato por construção.
"""
from __future__ import annotations

import collections
import csv
import pathlib
import random
import re
from dataclasses import dataclass, field

from gama.formas import forma
from gama.normalizar import chave_processo

from . import render as R
from .bancos import FEMININO
from .fichas import Dispositivo, Ficha, Sumula

# ------------------------------------------------------------------ mistura

# Tribunal das citações pela matéria do documento — medido no dev set em 22/09:
# acórdão real vem do tribunal da própria matéria em 79/82 (STF quase nunca);
# referência vaga vem do STF em ~60% ("Rcl de 2025, Rel. Min. ..."), senão do
# tribunal da matéria. Peso pequeno nos demais: robustez, não imitação cega.
TRIBUNAIS_REAL = {
    "trabalhista": {"TST": 10, "STF": 1, "STJ": 0.3},
    "eleitoral": {"TSE": 10, "STF": 0.5, "STJ": 0.3},
    "militar": {"STM": 10, "STF": 0.5, "STJ": 0.3},
    "penal": {"STJ": 10, "STF": 0.5, "STM": 0.3},
    "civel": {"STJ": 10, "STF": 0.5, "TST": 0.2},
}
TRIBUNAIS_VAGA = {
    "trabalhista": {"STF": 4, "TST": 2, "STJ": 0.3},
    "eleitoral": {"STF": 3, "TSE": 3, "STJ": 0.3},
    "militar": {"STF": 3, "STM": 4, "STJ": 0.3},
    "penal": {"STF": 4, "STJ": 3, "STM": 0.3},
    "civel": {"STF": 4, "STJ": 3, "TST": 0.2},
}
LEIS_DA_MATERIA = {
    "trabalhista": {"5452": 6, "CF": 3, "13105": 1},
    "eleitoral": {"4737": 4, "LC64": 4, "CF": 2},
    "militar": {"1001": 5, "3689": 2, "CF": 3},
    "penal": {"3689": 5, "CF": 3, "1001": 1},
    "civel": {"13105": 4, "10406": 3, "8078": 3, "CF": 2},
}


def mistura_do_dev(dados: pathlib.Path) -> dict:
    """Frequência de cada tipo de citação por nível e nº de citações por documento."""
    tipos = {1: collections.Counter(), 2: collections.Counter()}
    por_doc = collections.Counter()
    for r in csv.DictReader(open(dados / "goldenset_offsets.csv", encoding="utf-8-sig")):
        n = int(r["nivel"])
        por_doc[r["documento_id"]] += 1
        fm = forma(r["trecho"])
        c, t = r["classificacao"], r["tipo"]
        if c == "incompleta":
            k = "vaga"
        elif t == "lei":
            k = "lei" if c == "real" else "lei_inv"
        elif fm == "sumula":
            k = "sumula" if c == "real" else "sumula_inv"
        elif fm == "tema":
            k = "tema"
        else:
            k = "acordao" if c == "real" else "proc_inv"
        tipos[n][k] += 1
    contagens = {1: [], 2: []}
    for doc, q in por_doc.items():
        contagens[2 if "_n2_" in doc else 1].append(q)
    return {"tipos": tipos, "contagens": contagens}


# Fração de cada banco que vem da expansão por LLM (o resto, do banco da org).
# Ancorada na massa não vista estimada pelo Chao1 (~19% das carregadoras, ~33% do
# enchimento), com folga para diversidade: o cego usa frases que não vimos.
FRACAO_LLM = {"carregadoras": 0.30, "enchimento": 0.40, "genericas": 0.40,
              "blocos_narrativos": 0.40, "generos": 0.35}


def _origem(e: dict) -> str:
    return "llm" if str(e.get("origem", e.get("doc", ""))).startswith("llm") else "org"


# ------------------------------------------------------------------ slots

_MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto",
          "setembro", "outubro", "novembro", "dezembro"]


def _slot(nome: str, rng: random.Random) -> str:
    if nome == "CNJ":
        return (f"{rng.randint(0, 9999999):07d}-{rng.randint(0, 99):02d}.{rng.randint(2015, 2025)}."
                f"{rng.randint(1, 8)}.{rng.randint(1, 27):02d}.{rng.randint(0, 9999):04d}")
    if nome == "DATA":
        # Medido: 26/26 documentos da org usam UMA data só (preâmbulo, narrativa e
        # fecho), sempre entre 2019 e 2025.
        return f"{rng.randint(1, 28)} de {rng.choice(_MESES)} de {rng.randint(2019, 2025)}"
    if nome == "FLS":
        a = rng.randint(10, 900)
        return f"fls. {a}/{a + rng.randint(1, 400)}"
    if nome == "OAB":
        return f"OAB/{rng.choice(['SP', 'RJ', 'MG', 'RS', 'PR', 'BA', 'PE', 'CE', 'DF', 'GO'])} {rng.randint(10000, 499999)}"
    if nome == "PROTOCOLO":
        return f"{rng.randint(2018, 2025)}.{rng.randint(100000, 9999999)}"
    raise KeyError(nome)


def preencher(texto: str, rng: random.Random, data: str | None = None) -> str:
    def f(m):
        if m.group(1) == "DATA" and data:
            return data
        return _slot(m.group(1), rng)
    return re.sub(r"\{(CNJ|DATA|FLS|OAB|PROTOCOLO)\}", f, texto)


# ------------------------------------------------------------------ quebra de linha

def quebrar(paragrafo: str, limite: int) -> str:
    """Quebra da organização: termina a linha logo após a palavra que a leva a >= limite.

    Só troca espaço por "\\n" — len() e offsets não mudam.
    """
    out = list(paragrafo)
    col = 0
    for m in re.finditer(r"(\S+)(\s?)", paragrafo):
        col += len(m.group(1))
        if col >= limite and m.group(2):
            out[m.start(2)] = "\n"
            col = 0
        else:
            col += len(m.group(2))
    return "".join(out)


def _limite(rng: random.Random) -> int:
    """Limite por parágrafo, pela distribuição medida no nível 1 (menor limite que
    reproduz cada parágrafo): 91 em 29/60, 90 em 19/60, 86-89 no resto."""
    r = rng.random()
    return 91 if r < 0.5 else 90 if r < 0.82 else rng.randint(86, 89)


# ------------------------------------------------------------------ documento

@dataclass
class Documento:
    texto: str
    spans: list = field(default_factory=list)     # [(inicio, fim, Cit)]
    meta: dict = field(default_factory=dict)


@dataclass
class _Item:
    cit: R.Cit | None                 # None = referência genérica (sem rótulo)
    ref: dict | None = None


class Montador:
    def __init__(self, bancos: dict, fichas: list, sumulas: list, disps: list,
                 mistura: dict, idx):
        self.b = bancos
        self.idx = idx
        self.mistura = mistura
        self.sumulas: list[Sumula] = sumulas
        self.disps: dict[str, list[Dispositivo]] = collections.defaultdict(list)
        for d in disps:
            self.disps[d.lei].append(d)
        # Só fichas citáveis sem ambiguidade: a mesma chave com a mesma cadeia em
        # duas fichas não tem desempate possível — a organização não as cita.
        grupos = collections.defaultdict(list)
        for f in fichas:
            grupos[(chave_processo(f.numero), f.cadeia)].append(f)
        self.fichas: dict[str, list[Ficha]] = collections.defaultdict(list)
        for (_, cad), fs in grupos.items():
            f = fs[0]
            # TSE sem CNJ (só o número antigo) não tem forma de citação no dev set.
            if len(fs) == 1 and cad and (f.tribunal != "TSE" or "-" in f.numero):
                self.fichas[f.tribunal].append(f)
        self.vagas: dict[str, list[Ficha]] = collections.defaultdict(list)
        for f in fichas:
            if f.relator and f.ano:
                self.vagas[f.tribunal].append(f)

    # -------------------------------------------------------- escolhas

    def _pool(self, chave: str, nivel: int, rng: random.Random | None = None) -> list:
        """Entradas do banco utilizáveis neste nível (nível 1 não recebe texto ruidoso).

        Com `rng`, sorteia antes a ORIGEM (org ou LLM) pela FRACAO_LLM e devolve só
        aquela metade — assim 600 frases de LLM não afogam as 18 da organização.
        """
        itens = [e for e in self.b.get(chave, [])
                 if nivel == 2 or (1 in e.get("niveis", [1]) and e.get("nivel", 1) == 1)]
        if rng is None or chave not in FRACAO_LLM:
            return itens
        llm = [e for e in itens if _origem(e) == "llm"]
        org = [e for e in itens if _origem(e) == "org"]
        if not llm or not org:
            return itens
        return llm if rng.random() < FRACAO_LLM[chave] else org

    def _tribunal(self, materia: str, rng: random.Random, tabela: dict) -> str:
        pesos = tabela.get(materia, tabela["civel"])
        return rng.choices(list(pesos), weights=list(pesos.values()))[0]

    def _sem_candidato(self, cit: R.Cit) -> bool:
        """Inventada precisa ter ZERO candidatos no acervo — garantia da organização."""
        from gama.normalizar import nucleo_numerico
        n = nucleo_numerico(cit.texto)
        return not n or not self.idx.candidatos_processo(n)

    def citacao(self, tipo: str, materia: str, nivel: int, rng: random.Random) -> R.Cit:
        if tipo == "acordao":
            trib = self._tribunal(materia, rng, TRIBUNAIS_REAL)
            return R.real_acordao(rng.choice(self.fichas[trib]), rng, nivel)
        if tipo == "vaga":
            trib = self._tribunal(materia, rng, TRIBUNAIS_VAGA)
            return R.vaga(rng.choice(self.vagas[trib]), rng, nivel)
        if tipo == "sumula":
            return R.real_sumula(rng.choice(self.sumulas), rng, nivel)
        if tipo == "sumula_inv":
            return R.inventada_sumula(rng, nivel)
        if tipo == "tema":
            return R.inventado_tema(rng, nivel)
        if tipo == "lei":
            pesos = LEIS_DA_MATERIA.get(materia, LEIS_DA_MATERIA["civel"])
            lei = rng.choices(list(pesos), weights=list(pesos.values()))[0]
            return R.real_lei(rng.choice(self.disps[lei]), rng, nivel)
        if tipo == "lei_inv":
            return R.inventada_lei(rng, nivel)
        for _ in range(50):
            c = R.inventada_processo(rng, nivel)
            if self._sem_candidato(c):
                return c
        raise RuntimeError("não achei número inventado livre")

    def _tipos(self, nivel: int, rng: random.Random) -> list[str]:
        cont = self.mistura["contagens"][nivel]
        n = max(3, rng.choice(cont) + rng.randint(-1, 1))
        freq = self.mistura["tipos"][nivel]
        # Suavização: nenhum tipo some por acaso da amostra de 13 documentos.
        chaves = ["acordao", "vaga", "sumula", "sumula_inv", "tema", "lei", "lei_inv", "proc_inv"]
        pesos = [freq.get(k, 0) + 1.5 for k in chaves]
        return rng.choices(chaves, weights=pesos, k=n)

    # -------------------------------------------------------- frases

    def _carregadora(self, rotulo: str, usadas: set, nivel: int, rng: random.Random) -> str:
        alvo = "LEI" if rotulo == "LEI" else ("JURIS", "VAGA")
        pool = [c for c in self._pool("carregadoras", nivel, rng)
                if (c["rotulo"] == alvo if isinstance(alvo, str) else c["rotulo"] in alvo)]
        livres = [c for c in pool if c["molde"] not in usadas] or pool
        c = rng.choice(livres)
        usadas.add(c["molde"])
        return c["molde"]

    @staticmethod
    def _artigo(molde: str, genero: str) -> tuple[str, str, str]:
        """Molde -> (antes, depois) com o artigo concordando com o gênero da citação."""
        m = re.search(r"\{(O?)(\w+)\}\s*\{CIT\}", molde)
        art = m.group(2)
        if genero == "f":
            art = FEMININO.get(art, art)
        if m.group(1):
            art = art[0].upper() + art[1:]
        return molde[:m.start()] + art + " ", molde[m.end():]

    def _paragrafo(self, itens: list, narrativa: list, usadas: set, nivel: int,
                   rng: random.Random, abre_com_enchimento: bool) -> tuple[str, list]:
        """-> (texto do parágrafo sem quebra, [(ini, fim, Cit)] relativos ao parágrafo)."""
        def um_enchimento() -> str:
            pool = self._pool("enchimento", nivel, rng)
            return rng.choices(pool, weights=[e.get("peso", 1) for e in pool])[0]["texto"]
        frases, spans = [], []
        pos = 0

        def add(fr: str):
            nonlocal pos
            if frases:
                pos += 1
            frases.append(fr)
            pos += len(fr)

        for fr in narrativa:
            add(preencher(fr, rng, self._data))
        if abre_com_enchimento:
            add(um_enchimento())
        for it in itens:
            for _ in range(rng.choices([0, 1, 2], weights=[4, 4, 1])[0]):
                add(um_enchimento())
            rot = it.cit.rotulo if it.cit else it.ref["rotulo"]
            antes, depois = self._artigo(self._carregadora(rot, usadas, nivel, rng),
                                         it.cit.genero if it.cit else it.ref["genero"])
            antes, depois = preencher(antes, rng, self._data), preencher(depois, rng, self._data)
            corpo = it.cit.texto if it.cit else it.ref["ref"]
            ini = pos + (1 if frases else 0) + len(antes)
            add(antes + corpo + depois)
            if it.cit:
                spans.append((ini, ini + len(corpo), it.cit))
        for _ in range(rng.choices([0, 1, 2], weights=[3, 4, 2])[0]):
            add(um_enchimento())
        return " ".join(frases), spans

    # -------------------------------------------------------- documento

    def montar(self, rng: random.Random, nivel: int) -> Documento:
        g = rng.choice(self._pool("generos", nivel, rng))
        materia = g["materia"]
        usadas: set = set()
        self._data = _slot("DATA", rng)

        itens = [_Item(self.citacao(t, materia, nivel, rng)) for t in self._tipos(nivel, rng)]
        for _ in range(rng.choices([0, 1, 2], weights=[3, 5, 2])[0]):
            itens.append(_Item(None, rng.choice(self._pool("genericas", nivel, rng))))
        rng.shuffle(itens)

        secoes = g["secoes"]
        # Introdução leva 1 item; o resto se espalha pelas seções.
        grupos = [[] for _ in range(len(secoes) + 1)]
        grupos[0].append(itens[0])
        for it in itens[1:]:
            grupos[rng.randint(1, len(secoes))].append(it)

        blocos = [b for b in self._pool("blocos_narrativos", nivel, rng) if b["materia"] == materia] or \
            [b for b in self._pool("blocos_narrativos", nivel) if b["materia"] == materia] or \
            self._pool("blocos_narrativos", nivel)

        partes: list[str] = [preencher(g["preambulo"], rng, self._data), "\n\n\n"]
        spans: list = []

        def emite(par: str, sp: list):
            base = sum(len(p) for p in partes)
            partes.append(quebrar(par, _limite(rng)))
            for a, b, c in sp:
                spans.append((base + a, base + b, c))

        par, sp = self._paragrafo(grupos[0], [], usadas, nivel, rng, abre_com_enchimento=True)
        emite(par, sp)
        # Um bloco narrativo por seção, na ordem original, sem repetir no documento.
        ordem = rng.sample(blocos, len(blocos))
        for s_i, (titulo, grupo) in enumerate(zip(secoes, grupos[1:])):
            partes.append("\n\n" + titulo + "\n\n")
            narrativa = list(ordem[s_i % len(ordem)]["frases"])
            par, sp = self._paragrafo(grupo, narrativa, usadas, nivel, rng, abre_com_enchimento=False)
            emite(par, sp)
        for bloco in g["fecho"]:
            partes.append("\n\n")
            # O fecho vem quebrado do original; reflui antes de quebrar de novo.
            emite(re.sub(r"\s*\n\s*", " ", preencher(bloco, rng, self._data)), [])
        texto = "".join(partes)
        return Documento(texto, spans, {"genero": g["doc"], "materia": materia, "nivel": nivel})
