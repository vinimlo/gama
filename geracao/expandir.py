# -*- coding: utf-8 -*-
"""Expansão dos bancos da organização por LLMs de pesos abertos.

Por quê: o Chao1 estima que ~19% das carregadoras e ~33% do enchimento do banco da
organização não aparecem no dev set — o conjunto cego vai usá-los. A validação
adversarial honesta (bancos de metade do dev contra a outra metade) mostrou isso
como a principal pista que separa nossos documentos dos deles. Os LLMs escrevem
variantes no mesmo registro; nada que eles escrevem carrega citação (a régua
rejeita), então o rótulo continua exato por construção.

    docker compose run --rm lab python -m geracao.expandir --saida /app/corpus/bancos/llm.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import random
import re
import unicodedata

from gama.extrair import extrair

from . import prompts as P
from .llm import MODELOS, chat
from .montar import preencher, quebrar

_SLOT_ART = re.compile(r"\{(O|o|no|do|ao|pelo)\}\s*\{CIT\}")
_PALAVRAS_DE_CITACAO = re.compile(
    r"\b(art\.|artigo|lei\b|súmula|sumula|resp\b|rcl\b|nº|n\.º|stf|stj|tst|tse|stm|"
    r"relator|relatoria|rel\.|min\.)", re.I)


def _norm(s: str) -> str:
    d = unicodedata.normalize("NFD", s.lower())
    d = "".join(c for c in d if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z{} ]", "", d).strip()


def _limpa_de_citacao(texto: str) -> bool:
    """Régua não acha nada e não há dígito fora dos slots."""
    rng = random.Random(0)
    cheio = preencher(texto, rng)
    if re.search(r"\d", re.sub(r"\{(CNJ|DATA|FLS|OAB|PROTOCOLO)\}", "", texto)):
        return False
    return not extrair(cheio)


_CHAVES = ("moldes", "frases", "refs", "blocos", "pecas", "texto")   # "texto": reais.anotar


def _json(resposta: str) -> dict:
    """Objeto JSON da resposta com uma das chaves esperadas.

    Tolera cerca ```json (Kimi) e raciocínio em texto ANTES do JSON (o GLM-5.3
    ignora think=false e escreve "The user wants..." com chaves {o} no meio):
    tenta decodificar a partir de cada "{" até achar um objeto com chave esperada.
    """
    txt = re.sub(r"```(?:json)?", "", resposta)
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", txt):
        try:
            obj, _ = dec.raw_decode(txt[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and any(k in obj for k in _CHAVES):
            return obj
    return {}


# ------------------------------------------------------------------ validadores

def val_carregadora(m: str) -> bool:
    if not isinstance(m, str) or m.count("{CIT}") != 1 or not _SLOT_ART.search(m):
        return False
    if re.search(r"\{(?!O\}|o\}|no\}|do\}|ao\}|pelo\}|CIT\})", m):
        return False
    n = len(m.split())
    neutro = _SLOT_ART.sub("o precedente invocado", m)
    return 6 <= n <= 36 and m.rstrip().endswith(".") and _limpa_de_citacao(neutro) \
        and not _PALAVRAS_DE_CITACAO.search(neutro)


def val_enchimento(f: str) -> bool:
    return isinstance(f, str) and 5 <= len(f.split()) <= 26 and f.rstrip().endswith(".") \
        and "{" not in f and _limpa_de_citacao(f) and not _PALAVRAS_DE_CITACAO.search(f)


def val_generica(r: dict) -> bool:
    if not isinstance(r, dict) or r.get("genero") not in ("m", "f") or r.get("tipo") not in ("JURIS", "LEI"):
        return False
    ref = r.get("ref", "")
    if not isinstance(ref, str) or not 2 <= len(ref.split()) <= 12 or re.search(r"\d|\{", ref):
        return False
    if re.match(r"(?i)(o|a|os|as)\s", ref) or re.search(r"(?i)\brel\.|relat|min\.", ref):
        return False
    return _limpa_de_citacao(f"Cita-se o {ref}, no ponto.")


def val_bloco(b) -> bool:
    if not isinstance(b, list) or not 2 <= len(b) <= 3 or not all(isinstance(f, str) for f in b):
        return False
    txt = " ".join(b)
    if re.search(r"\{(?!DATA\}|FLS\})", txt) or not 20 <= len(txt.split()) <= 100:
        return False
    return _limpa_de_citacao(txt) and not _PALAVRAS_DE_CITACAO.search(txt)


def val_peca(p: dict) -> bool:
    if not isinstance(p, dict):
        return False
    try:
        cab, tit, abe = p["cabecalho"], p["titulo"], p["abertura"]
        sec, fec = p["secoes"], p["fecho"]
    except KeyError:
        return False
    if not (isinstance(sec, list) and 3 <= len(sec) <= 5 and all(re.match(r"^[IVX]+\s+—\s+\S", s) for s in sec)):
        return False
    if not (isinstance(fec, list) and 2 <= len(fec) <= 3 and "{DATA}" in fec[-1]):
        return False
    corpo = " ".join([tit, abe] + fec)
    # Ordinal no cabeçalho ("3ª VARA", "1ª TURMA") é realista e serve de distrator.
    cab_sem = re.sub(r"\d{1,2}\s*[ªº]", "", re.sub(r"\{\w+\}", "", cab))
    return _limpa_de_citacao(corpo) and not re.search(r"\d", cab_sem)


# ------------------------------------------------------------------ tarefas

def _exemplos(itens: list, k: int, rng: random.Random, fmt=lambda x: x) -> str:
    return "\n".join(f"- {fmt(x)}" for x in rng.sample(itens, min(k, len(itens))))


def _peca_exemplo(g: dict) -> str:
    pre = g["preambulo"]
    return json.dumps({"preambulo": pre[:700], "secoes": g["secoes"], "fecho": g["fecho"]},
                      ensure_ascii=False)


def tarefas(org: dict, escala: int) -> list[dict]:
    t = []
    modelos = list(MODELOS)
    molde_j = sorted({c["molde"] for c in org["carregadoras"] if c["rotulo"] != "LEI"})
    molde_l = sorted({c["molde"] for c in org["carregadoras"] if c["rotulo"] == "LEI"})
    ench = [e["texto"] for e in org["enchimento"] if 1 in e["niveis"]]
    gen = sorted({g["ref"] for g in org["genericas"] if 1 in g["niveis"]})
    for mod in modelos:
        for s in range(3 * escala):
            r = random.Random(f"cj{mod}{s}")
            t.append({"tipo": "carregadoras", "rotulo": "JURIS", "modelo": mod, "semente": s,
                      "prompt": P.CARREGADORAS.format(alvo=P.ALVO["JURIS"], n=25,
                                                      exemplos=_exemplos(molde_j, 14, r))})
        for s in range(escala):
            r = random.Random(f"cl{mod}{s}")
            t.append({"tipo": "carregadoras", "rotulo": "LEI", "modelo": mod, "semente": s,
                      "prompt": P.CARREGADORAS.format(alvo=P.ALVO["LEI"], n=25,
                                                      exemplos=_exemplos(molde_l, 10, r))})
        for s in range(3 * escala):
            r = random.Random(f"en{mod}{s}")
            t.append({"tipo": "enchimento", "modelo": mod, "semente": s,
                      "prompt": P.ENCHIMENTO.format(n=40, exemplos=_exemplos(ench, 14, r))})
        for s in range(escala):
            r = random.Random(f"ge{mod}{s}")
            t.append({"tipo": "genericas", "modelo": mod, "semente": s,
                      "prompt": P.GENERICAS.format(n=30, exemplos=_exemplos(gen, 12, r))})
    for i, (materia, tipos) in enumerate(P.TIPOS_DE_PECA.items()):
        blocos = [b["frases"] for b in org["blocos_narrativos"] if b["materia"] == materia and 1 in b["niveis"]] \
            or [b["frases"] for b in org["blocos_narrativos"]]
        for j, mod in enumerate(modelos):
            for s in range(escala):
                r = random.Random(f"na{materia}{mod}{s}")
                t.append({"tipo": "blocos_narrativos", "materia": materia, "modelo": mod, "semente": s,
                          "prompt": P.NARRATIVAS.format(
                              n=8, materia=materia, descricao=P.DESCRICAO_MATERIA[materia],
                              exemplos=_exemplos(blocos, 4, r, lambda b: " ".join(b)))})
        gen_org = [g for g in org["generos"] if g["materia"] == materia and g["nivel"] == 1] or \
            [g for g in org["generos"] if g["nivel"] == 1]
        for k, tipo in enumerate(tipos):
            mod = modelos[(i + k) % len(modelos)]
            r = random.Random(f"pc{materia}{tipo}")
            t.append({"tipo": "generos", "materia": materia, "modelo": mod, "semente": 0,
                      "prompt": P.PECAS.format(n=2, tipo=tipo, materia=materia,
                                               descricao=P.DESCRICAO_MATERIA[materia],
                                               exemplos="\n".join(_peca_exemplo(g) for g in r.sample(gen_org, min(2, len(gen_org)))))})
    return t


def executar(t: dict) -> tuple[dict, list]:
    msgs = [{"role": "system", "content": P.SISTEMA}, {"role": "user", "content": t["prompt"]}]
    d = _json(chat(t["modelo"], msgs, temperatura=0.9, semente=t["semente"]))
    orig = f"llm:{t['modelo']}"
    tipo = t["tipo"]
    if tipo == "carregadoras":
        return t, [{"molde": m.strip(), "rotulo": t["rotulo"], "doc": orig, "nivel": 1}
                   for m in d.get("moldes", []) if val_carregadora(m)]
    if tipo == "enchimento":
        return t, [{"texto": f.strip(), "niveis": [1, 2], "peso": 1, "origem": orig}
                   for f in d.get("frases", []) if val_enchimento(f)]
    if tipo == "genericas":
        return t, [{"texto": "", "ref": r["ref"].strip(), "genero": r["genero"], "rotulo": r["tipo"],
                    "niveis": [1, 2], "origem": orig} for r in d.get("refs", []) if val_generica(r)]
    if tipo == "blocos_narrativos":
        return t, [{"materia": t["materia"], "frases": [f.strip() for f in b], "niveis": [1, 2], "origem": orig}
                   for b in d.get("blocos", []) if val_bloco(b)]
    if tipo == "generos":
        out = []
        for i, p in enumerate(d.get("pecas", [])):
            if not val_peca(p):
                continue
            pre = f"{p['cabecalho'].strip()}\n\n{p['titulo'].strip()}\n\n{quebrar(p['abertura'].strip(), 91)}"
            out.append({"doc": f"{orig}:{t['materia']}:{i}", "materia": t["materia"], "nivel": 1,
                        "preambulo": pre, "secoes": [s.strip() for s in p["secoes"]],
                        "fecho": [f.strip() for f in p["fecho"]], "origem": orig})
        return t, out
    return t, []


def _chave(tipo: str, item: dict) -> str:
    if tipo == "carregadoras":
        return _norm(item["molde"])
    if tipo == "enchimento":
        return _norm(item["texto"])
    if tipo == "genericas":
        return _norm(item["ref"])
    if tipo == "blocos_narrativos":
        return _norm(" ".join(item["frases"]))
    return _norm(item["preambulo"][:200])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", default="/app/corpus/bancos/org.json")
    ap.add_argument("--saida", default="/app/corpus/bancos/llm.json")
    ap.add_argument("--escala", type=int, default=2)
    ap.add_argument("--paralelo", type=int, default=6)
    a = ap.parse_args()
    org = json.loads(pathlib.Path(a.org).read_text(encoding="utf-8"))
    ts = tarefas(org, a.escala)
    # Intercala modelos: um modelo lento não bloqueia a fila dos outros.
    random.Random(0).shuffle(ts)
    vistos = {k: {_chave(k, x) for x in org.get(k, []) if k != "genericas" or x.get("ref")}
              for k in ("carregadoras", "enchimento", "genericas", "blocos_narrativos", "generos")}
    banco = {k: [] for k in vistos}
    stats = {}
    print(f"{len(ts)} chamadas", flush=True)
    with cf.ThreadPoolExecutor(a.paralelo) as ex:
        futs = [ex.submit(executar, t) for t in ts]
        for n, fu in enumerate(cf.as_completed(futs), 1):
            try:
                t, itens = fu.result()
            except Exception as e:                    # noqa: BLE001 — registra e segue
                print("falhou:", e, flush=True)
                continue
            k = t["tipo"]
            novos = 0
            for it in itens:
                c = _chave(k, it)
                if c and c not in vistos[k]:
                    vistos[k].add(c)
                    banco[k].append(it)
                    novos += 1
            s = stats.setdefault(f"{k}:{t['modelo']}", [0, 0])
            s[0] += novos
            s[1] += 1
            if n % 10 == 0:
                print(f"{n}/{len(ts)} " + " ".join(f"{k}={len(v)}" for k, v in banco.items()), flush=True)
    # ordem estável (determinismo do gerador a partir do banco)
    for k in banco:
        banco[k].sort(key=lambda x: _chave(k, x))
    pathlib.Path(a.saida).write_text(json.dumps(banco, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: len(v) for k, v in banco.items()}), json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
