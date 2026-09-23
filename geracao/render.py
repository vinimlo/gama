# -*- coding: utf-8 -*-
"""Forma de superfície das citações — espelho do gerador da organização.

Cada função recebe a matéria-prima (ficha, súmula, dispositivo ou nada, no caso
das inventadas) e devolve uma `Cit`: o texto do span, o rótulo e o gênero
gramatical (para o artigo do carregador concordar: "o REsp", "a Rcl").

As variantes vêm do catálogo das 192 citações do dev set (ver
`wiki/conceitos/convencoes-de-borda.md`), com duas camadas:

* nível 1: formas limpas — sigla ou extenso, `nº` opcional, UF sempre com `/`;
* nível 2: abreviações e separadores alternativos que a organização documentou
  ("Rec. Esp.", "AgRg", "RCL", "n°", "No", " - SP", "(SC)", CNJ sem pontos).
  O ruído de OCR e as quebras de linha vêm depois, de `gama.ruido`.

Tudo aqui respeita as convenções de borda: sem artigo ou preposição dentro do
span, sem pontuação final, número clássico sempre com UF.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .fichas import UFS, Dispositivo, Ficha, Sumula

NBSP = " "


@dataclass(frozen=True)
class Cit:
    texto: str
    tipo: str              # jurisprudencia | lei
    classe: str            # real | inventada | incompleta
    id_canonico: str | None
    genero: str            # m | f
    rotulo: str            # JURIS | LEI | VAGA  (o que o extrator aprende)
    fonte: str             # de que gerador saiu (auditoria)


# ------------------------------------------------------------------ classes

# sigla canônica -> (formas nível 1, formas extras nível 2, gênero, plural)
CLASSES = {
    "REsp":  (["REsp", "Recurso Especial"], ["Rec. Esp.", "RESP", "R.Esp."], "m", False),
    "AREsp": (["AREsp", "Agravo em Recurso Especial"], ["AgREsp", "ARESP", "A.REsp", "Ag. em REsp"], "m", False),
    "AgInt": (["AgInt", "Agravo Interno"], ["Ag. Int.", "AGINT", "Ag.Int."], "m", False),
    "AgR":   (["AgRg", "AgR", "Agravo Regimental"], ["AG.REG.", "Ag. Reg.", "AGR", "AgRG"], "m", False),
    "ED":    (["EDcl", "ED", "Embargos de Declaração"], ["EDs", "Emb. Decl.", "EMB.DECL."], "m", True),
    "EDv":   (["EDv", "Embargos de Divergência"], ["EREsp", "Emb. Div."], "m", True),
    "E":     (["E", "Embargos"], ["EMB"], "m", True),
    "RHC":   (["RHC", "Recurso em Habeas Corpus"], ["Rec. em HC", "R.H.C."], "m", False),
    "HC":    (["HC", "Habeas Corpus"], ["H.C.", "Hab. Corpus"], "m", False),
    "RMS":   (["RMS", "Recurso em Mandado de Segurança"], ["Rec. em MS"], "m", False),
    "MS":    (["MS", "Mandado de Segurança"], ["Mand. Seg."], "m", False),
    "Rcl":   (["Rcl", "Reclamação"], ["RCL", "Recl."], "f", False),
    "RE":    (["RE", "Recurso Extraordinário"], ["RE.", "Rec. Ext."], "m", False),
    "ARE":   (["ARE", "Recurso Extraordinário com Agravo"], ["A.R.E."], "m", False),
    "AI":    (["AI", "Agravo de Instrumento"], ["Ag. Inst.", "A.I."], "m", False),
    "SLS":   (["SLS", "Suspensão de Liminar e de Sentença"], ["S.L.S."], "f", False),
    "APL":   (["APL", "Apelação"], ["Apelação Criminal", "Ap.", "APELAÇÃO"], "f", False),
    "RSE":   (["RSE", "Recurso em Sentido Estrito"], ["Rec. Sent. Estr.", "R.S.E."], "m", False),
    "REspe": (["REspe", "Recurso Especial Eleitoral"], ["REspEl", "RESPE", "REspe."], "m", False),
    "AREspe": (["AREspe", "Agravo em Recurso Especial Eleitoral"], ["AREspEl", "ARESPE"], "m", False),
    "Rp":    (["Rp", "Representação"], ["Repr."], "f", False),
    "RR":    (["RR", "Recurso de Revista"], ["Rec. Revista"], "m", False),
    "ARR":   (["ARR", "Agravo em Recurso de Revista"], ["AgRR"], "m", False),
    "AIRR":  (["AIRR", "Agravo de Instrumento em Recurso de Revista"], ["AgAIRR"], "m", False),
    "RO":    (["RO", "Recurso Ordinário"], ["Rec. Ord."], "m", False),
    "AR":    (["AR", "Ação Rescisória"], ["Aç. Resc."], "f", False),
    "ADI":   (["ADI", "Ação Direta de Inconstitucionalidade"], ["A.D.I."], "f", False),
    "Ag":    (["Ag", "Agravo"], ["AG"], "m", False),
}

_ORDINAL = {"2": ("Segundo", "Segundos"), "3": ("Terceiro", "Terceiros"), "4": ("Quarto", "Quartos")}


def _base(sigla: str) -> tuple[str, str]:
    m = re.fullmatch(r"([A-Za-z]+?)(\d?)", sigla)
    return (m.group(1), m.group(2)) if m else (sigla, "")


def _nome(sigla: str, rng: random.Random, nivel: int, extenso: bool) -> tuple[str, str, bool]:
    base, ordem = _base(sigla)
    n1, n2, gen, plural = CLASSES.get(base, ([base], [], "m", False))
    if extenso:
        nome = n1[-1]
    else:
        opcoes = n1[:-1] or n1
        if nivel == 2 and rng.random() < 0.45:
            opcoes = opcoes + n2
        nome = rng.choice(opcoes)
    if ordem in _ORDINAL:
        nome = f"{_ORDINAL[ordem][0]} {nome}"
    return nome, gen, plural


def _conectivo(gen: str, plural: bool) -> str:
    return {("m", False): "no", ("f", False): "na", ("m", True): "nos", ("f", True): "nas"}[(gen, plural)]


def _cadeia_texto(cadeia: tuple, rng: random.Random, nivel: int) -> tuple[str, str, bool]:
    """('AgInt', 'AREsp') -> 'AgInt no AREsp' | 'Agravo Interno no Agravo em Recurso Especial'.

    Devolve também se a ÚLTIMA classe saiu por extenso: no dev set, classe por
    extenso vem sempre com "nº" ("Recurso Especial nº"); sigla, em ~40%.
    """
    extenso = rng.random() < 0.28
    partes = []
    ultimo_ext = False
    for s in cadeia:
        ext = extenso and rng.random() < 0.9
        nome, gen, plural = _nome(s, rng, nivel, ext)
        partes.append((nome, gen, plural))
        ultimo_ext = ext
    txt = partes[0][0]
    for nome, gen, plural in partes[1:]:
        txt += f" {_conectivo(gen, plural)} {nome}"
    return txt, partes[0][1], ultimo_ext


def _marca(ultimo_ext: bool, rng: random.Random, nivel: int) -> str:
    if nivel == 1:
        return "nº " if ultimo_ext or rng.random() < 0.4 else ""
    m = _marca_n(rng, nivel)
    return m if m or not ultimo_ext else "nº "


# ------------------------------------------------------------------ número

def _marca_n(rng: random.Random, nivel: int) -> str:
    """O 'nº' entre classe e número (ou nada)."""
    if nivel == 1:
        return rng.choice(["nº ", "nº ", "", "", "nº "])
    return rng.choice(["nº ", "", "n° ", "n. ", "No ", "Nº ", "N° ", "nº" + NBSP, "n°" + NBSP,
                       "No" + NBSP, "Nº  ", "n.  "])


def _pontos(num: str, rng: random.Random, nivel: int) -> str:
    """Número clássico com ou sem separador de milhar."""
    digitos = num.replace(".", "")
    if rng.random() < 0.45:
        return digitos
    g = []
    while len(digitos) > 3:
        g.insert(0, digitos[-3:])
        digitos = digitos[:-3]
    g.insert(0, digitos)
    sep = "." if nivel == 1 or rng.random() < 0.8 else " "
    return sep.join(g)


def _uf(uf: str, rng: random.Random, nivel: int) -> str:
    if nivel == 1:
        return "/" + uf
    return rng.choice(["/" + uf, "/" + uf, " - " + uf, "-" + uf, "/ " + uf, " (" + uf + ")",
                       " – " + uf, " (" + uf + ")"])


def _cnj(num: str, rng: random.Random, nivel: int) -> str:
    """CNJ; no nível 2, às vezes sem os pontos ('0600316-4920206160182')."""
    if nivel == 2 and rng.random() < 0.18:
        a, b = num.split("-", 1)
        return a + "-" + b.replace(".", "")
    return num


# ------------------------------------------------------------------ real: acórdão

def _prefixo_tst(cadeia: tuple, rng: random.Random, nivel: int) -> str:
    siglas = "-".join(_base(s)[0] if _base(s)[0] in CLASSES else s for s in cadeia)
    if nivel == 2 and rng.random() < 0.3:
        siglas = re.sub(r"-", lambda _: rng.choice(["-", " - ", "- "]), siglas)
    return siglas


def real_acordao(f: Ficha, rng: random.Random, nivel: int) -> Cit:
    cad = f.cadeia
    if f.tribunal == "TST":
        num = _cnj(f.numero, rng, nivel)
        siglas = _prefixo_tst(cad, rng, nivel)
        estilo = rng.random()
        if estilo < 0.35:
            txt = f"TST-{siglas}-{num}"
        elif estilo < 0.55:
            txt = f"{rng.choice(['processo', 'Processo'])} {_marca_n(rng, nivel)}TST-{siglas}-{num}"
        elif estilo < 0.85:
            txt = f"{siglas}-{num}"
        else:
            nome, _, ext = _cadeia_texto(cad, rng, nivel)
            txt = f"{nome} {_marca(ext, rng, nivel)}{num}"
        return Cit(txt, "jurisprudencia", "real", f.id, "m", "JURIS", "real:tst")

    if f.tribunal == "TSE":
        num = _cnj(f.numero, rng, nivel) if "-" in f.numero else _pontos(f.numero, rng, nivel)
        if len(cad) >= 2 and rng.random() < 0.4:
            # 'AgR-REspe 378-82...' / 'ED no AgR-REspe ...'
            sig = [_nome(s, rng, nivel, False)[0] for s in cad]
            txt = " no ".join(sig[:-2] + [f"{sig[-2]}-{sig[-1]}"])
            ext = False
        else:
            txt, _, ext = _cadeia_texto(cad, rng, nivel)
        gen = CLASSES.get(_base(cad[0])[0], ([], [], "m", False))[2]
        return Cit(f"{txt} {_marca(ext, rng, nivel)}{num}", "jurisprudencia", "real", f.id, gen,
                   "JURIS", "real:tse")

    # STF / STJ / STM
    if f.tribunal == "STF" and len(cad) >= 2 and rng.random() < 0.35:
        cad = cad[-1:]                  # "Rcl nº 68.244/SP" para ficha AgR na Rcl
    txt, gen, ext = _cadeia_texto(cad, rng, nivel)
    if "-" in f.numero:
        num = _cnj(f.numero, rng, nivel)
        uf = f.uf or rng.choice(UFS)
        cauda = _uf(uf, rng, nivel) if rng.random() < 0.6 else ""
    else:
        num = _pontos(f.numero, rng, nivel)
        cauda = _uf(f.uf or rng.choice(UFS), rng, nivel)
    return Cit(f"{txt} {_marca(ext, rng, nivel)}{num}{cauda}", "jurisprudencia", "real", f.id, gen,
               "JURIS", f"real:{f.tribunal.lower()}")


# ------------------------------------------------------------------ súmula e tema

def _sumula_texto(numero: str, tribunal: str | None, vinculante: bool,
                  rng: random.Random, nivel: int) -> str:
    # Nunca "Enunciado da Súmula": no gabarito o span começa em "Súmula".
    cab = rng.choice(["Súmula", "Súmula", "Súmula", "Súm.", "SÚMULA"])
    if vinculante:
        base = f"{cab} Vinculante {numero}"
        if rng.random() < 0.3:
            base += " do STF"
        return base
    if tribunal is None:
        return f"{cab} {numero}"
    if rng.random() < 0.15:
        return f"{cab} {numero}/{tribunal}"
    return f"{cab} {numero} do {tribunal}"


def real_sumula(s: Sumula, rng: random.Random, nivel: int) -> Cit:
    txt = _sumula_texto(s.numero, s.tribunal, s.vinculante, rng, nivel)
    return Cit(txt, "jurisprudencia", "real", s.id, "f", "JURIS", "real:sumula")


_MAX_SUMULA = {"STF": 736, "STJ": 680, "TST": 465, "TSE": 75, "STM": 20}


def inventada_sumula(rng: random.Random, nivel: int) -> Cit:
    if rng.random() < 0.2:
        n = str(rng.randint(60, 199))
        return Cit(_sumula_texto(n, None, True, rng, nivel), "jurisprudencia", "inventada",
                   None, "f", "JURIS", "inv:sv")
    trib = rng.choice(["STF", "STF", "STJ", "TSE", "TST", "TSE"])
    n = str(rng.randint(_MAX_SUMULA[trib] + 5, 999))
    return Cit(_sumula_texto(n, trib, False, rng, nivel), "jurisprudencia", "inventada",
               None, "f", "JURIS", "inv:sumula")


def inventado_tema(rng: random.Random, nivel: int) -> Cit:
    # Nenhum Tema existe no acervo: qualquer número é `inventada`. A faixa cobre os
    # temas reais (1–1.400) e os acima deles — o v1 só via 1.400–2.999 e parava em
    # "Terna 725" (sonda m->rn).
    n = rng.randint(1, 2999)
    num = f"{n // 1000}.{n % 1000:03d}" if rng.random() < 0.6 else str(n)
    cauda = rng.choice(["da repercussão geral", "da repercussão geral", "do STF",
                        "de repercussão geral"])
    return Cit(f"Tema {num} {cauda}", "jurisprudencia", "inventada", None, "m", "JURIS", "inv:tema")


# ------------------------------------------------------------------ inventada: processo

def _numero_repetido(rng: random.Random) -> str:
    """'88.178': par de dígitos repetido na frente — o molde das Rcl inventadas do dev."""
    d = rng.randint(1, 9)
    return f"{d}{d}{rng.randint(0, 999):03d}"


def _cnj_stm(rng: random.Random) -> str:
    return (f"7{rng.randint(100000, 999999)}-{rng.randint(10, 99)}."
            f"{rng.randint(2015, 2026)}.7.00.0000")


def _cnj_tribunal(trib: str, rng: random.Random) -> str:
    j = {"TSE": "6", "TST": "5", "STM": "7"}[trib]
    seq = rng.choice([f"{rng.randint(1, 99999)}", f"0{rng.randint(600000, 609999)}"])
    return (f"{seq}-{rng.randint(10, 99)}.{rng.randint(2008, 2025)}.{j}."
            f"{rng.randint(0, 27):02d}.{rng.randint(0, 9999):04d}")


# (sigla da cadeia, gerador do número, usa UF?) com pesos do dev set
_INVENTADAS = [
    (("Rcl",), _numero_repetido, True, 30),
    (("RHC",), lambda r: _numero_repetido(r) if r.random() < 0.6 else str(r.randint(300000, 999999)), True, 6),
    (("REsp",), lambda r: str(r.randint(2400000, 9999999)), True, 6),
    (("RE",), lambda r: str(r.randint(2000000, 9999999)), True, 5),
    (("AREsp",), lambda r: str(r.choice([r.randint(2300000, 9999999), r.randint(50000, 999999)])), True, 5),
    (("AgInt",), _cnj_stm, True, 2),
    (("APL",), _cnj_stm, True, 2),
    (("RSE",), _cnj_stm, True, 2),
    (("AgInt", "REsp"), lambda r: str(r.randint(2400000, 9999999)), True, 2),
    (("HC",), lambda r: str(r.randint(900000, 9999999)), True, 2),
    (("MS",), lambda r: _numero_repetido(r), True, 1),
    (("ARE",), lambda r: str(r.randint(1600000, 9999999)), True, 1),
    (("REspe",), lambda r: _cnj_tribunal("TSE", r), False, 1),
    (("RR",), lambda r: _cnj_tribunal("TST", r), False, 1),
]


def inventada_processo(rng: random.Random, nivel: int) -> Cit:
    cad, gera, com_uf, _ = rng.choices(_INVENTADAS, weights=[x[3] for x in _INVENTADAS])[0]
    num = gera(rng)
    txt, gen, ext = _cadeia_texto(cad, rng, nivel)
    if "-" in num:
        num_txt = _cnj(num, rng, nivel)
        cauda = _uf(rng.choice(UFS), rng, nivel) if com_uf and rng.random() < 0.6 else ""
        if cad[0] in ("REspe",):
            cauda = ""
        if cad[0] == "RR" and rng.random() < 0.6:
            return Cit(f"TST-RR-{num_txt}" if rng.random() < 0.5 else f"RR-{num_txt}",
                       "jurisprudencia", "inventada", None, "m", "JURIS", "inv:tst")
    else:
        num_txt = _pontos(num, rng, nivel)
        cauda = _uf(rng.choice(UFS), rng, nivel)
    return Cit(f"{txt} {_marca(ext, rng, nivel)}{num_txt}{cauda}", "jurisprudencia", "inventada",
               None, gen, "JURIS", "inv:processo")


# ------------------------------------------------------------------ vaga (incompleta)

_VAGA_MOLDES = [
    # (molde, gênero) — os cinco moldes do dev set + variações próximas
    ("julgado do {T} proferido em {A} pela relatoria de {R}", "m", 9),
    ("acórdão do {T} julgado em {A} sob relatoria de {R}", "m", 6),
    ("precedente do {T} de {A}, da relatoria de {R}", "m", 7),
    ("{C} do {T}, de {A}, Rel. Min. {R}", "?", 7),
    ("{S} de {A}, Rel. Min. {R}", "?", 5),
    ("acórdão do {T} de {A}, Rel. Min. {R}", "m", 1),
    ("precedente do {T}, julgado em {A} sob a relatoria de {R}", "m", 1),
]


def vaga(f: Ficha, rng: random.Random, nivel: int) -> Cit:
    molde, gen, _ = rng.choices(_VAGA_MOLDES, weights=[x[2] for x in _VAGA_MOLDES])[0]
    base = f.cadeia[-1] if f.cadeia else "Rcl"
    nome_ext, g_ext, _ = _nome(base, rng, nivel, True)
    nome_sig, g_sig, _ = _nome(base, rng, nivel, False)
    rel = f.relator
    r = rng.random()
    if r < 0.25:
        rel = rel.upper()
    elif r < 0.5:
        rel = " ".join(p if p.lower() in ("de", "da", "do", "dos", "das", "e") else p.capitalize()
                       for p in rel.lower().split())
    txt = molde.format(T=f.tribunal, A=f.ano or rng.randint(2015, 2025), R=rel,
                       C=nome_ext, S=nome_sig)
    if gen == "?":
        gen = g_ext if "{C}" in molde else g_sig
    return Cit(txt, "jurisprudencia", "incompleta", None, gen, "VAGA", "vaga")


# ------------------------------------------------------------------ lei

# chave do índice -> formas de nomear o diploma (gênero do artigo: "do CPC", "da CLT")
DIPLOMAS = {
    "13105": [("do CPC", 3), ("do Código de Processo Civil", 3), ("da Lei nº 13.105/2015", 2)],
    "10406": [("do Código Civil", 4), ("do CC", 1), ("da Lei nº 10.406/2002", 1)],
    "5452":  [("da CLT", 4), ("da Consolidação das Leis do Trabalho", 3)],
    "3689":  [("do Código de Processo Penal", 3), ("do CPP", 3)],
    "1001":  [("do Código Penal Militar", 4), ("do CPM", 2)],
    "8078":  [("do Código de Defesa do Consumidor", 4), ("do CDC", 3), ("da Lei nº 8.078/1990", 1)],
    "4737":  [("do Código Eleitoral", 5), ("da Lei nº 4.737/1965", 1)],
    "CF":    [("da Constituição Federal", 5), ("da Constituição da República", 2), ("da CF", 2),
              ("da CF/88", 1), ("da Carta Magna", 1)],
    "LC64":  [("da Lei Complementar nº 64/1990", 4), ("da LC nº 64/1990", 1), ("da LC 64/90", 1)],
}

# diplomas que só aparecem com artigo inventado, e o maior artigo real de cada um
_MAX_ARTIGO = {"CF": 250, "8078": 119, "13105": 1072, "5452": 922, "4737": 383, "LC64": 28,
               "9504": 107, "13467": 6, "10406": 2046, "3689": 811, "1001": 410}
DIPLOMAS_SO_INVENTADOS = {
    "9504": [("da Lei nº 9.504/1997", 4), ("da Lei das Eleições", 1)],
    "13467": [("da Lei nº 13.467/2017", 4)],
}


def _diploma(chave: str, rng: random.Random) -> str:
    tabela = DIPLOMAS.get(chave) or DIPLOMAS_SO_INVENTADOS[chave]
    return rng.choices([t for t, _ in tabela], weights=[w for _, w in tabela])[0]


def _art(rng: random.Random, nivel: int) -> str:
    return rng.choice(["art. ", "art. ", "art. ", "artigo ", "art "] +
                      (["art." + NBSP, "Art. "] if nivel == 2 else []))


def _num_artigo(n: int) -> str:
    if n < 10:
        return f"{n}º"
    return f"{n // 1000}.{n % 1000:03d}" if n >= 1000 else str(n)


def real_lei(d: Dispositivo, rng: random.Random, nivel: int) -> Cit:
    txt = _art(rng, nivel) + d.artigo
    r = rng.random()
    if d.incisos and r < 0.35:
        txt += f", {rng.choice(d.incisos)}"
        if d.alineas and rng.random() < 0.25:
            txt += f", '{rng.choice(d.alineas)}'"
        txt += ","
    elif d.paragrafos and r < 0.45:
        txt += f", § {rng.choice(d.paragrafos)},"
    txt += " " + _diploma(d.lei, rng)
    return Cit(txt, "lei", "real", d.id, "m", "LEI", "real:lei")


def inventada_lei(rng: random.Random, nivel: int) -> Cit:
    chave = rng.choice(["CF", "CF", "CF", "8078", "13105", "5452", "4737", "LC64", "9504",
                        "13467", "10406", "3689"])
    n = _MAX_ARTIGO[chave] + rng.randint(8, 120)
    txt = _art(rng, nivel) + _num_artigo(n) + " " + _diploma(chave, rng)
    return Cit(txt, "lei", "inventada", None, "m", "LEI", "inv:lei")
