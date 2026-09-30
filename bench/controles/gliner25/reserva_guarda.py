# -*- coding: utf-8 -*-
"""Reserva da guarda com o GLiNER 2.5 (experimento 3a), nas 305 e nas 172.

Refaz `saidas/analises/reserva_da_guarda.py` (que não é alterado) com os spans já gravados: a
guarda de produção (`gama.extratores.guarda.Guarda`, limiar 0,95 intacto) sobre o Gama v1.3,
trocando só o reserva, isto é, quem entra onde o Gama hesita. Nenhum job novo.

Leitura, aparar, F1 de extração e bootstrap pelo harness validado (`bench.controles.avaliar`):
    Gama v1.3   `saidas/bench/gama.jsonl` nas 305 e `saidas/bench/modelo_base12.jsonl` nas 172
                (as referências do harness)
    régua       as gravações da régua do harness (reserva de produção = Gama v1.3 com guarda)
    reservas    os spans gravados de cada GLiNER, aparados como o harness apara (o score do
                formato curto vira a confiança, que a guarda ignora no reserva)
    oráculo     o gabarito como reserva (`span_de`, como no script original)

F1: mesmo tipo, IoU >= 0,5, casamento 1 para 1, VAGA em JURIS (`bench.pontuar.extracao`).
Bootstrap pareado por ementa, 2.000 reamostras, semente 0 (`avaliar.bootstrap`): contra a régua
(o Gama v1.3 com guarda, `avaliar.referencia`), contra "nenhum" e entre a 2.5 e a v2.1.

Conferência: nas 305, os reservas que o script original também tem (nenhum, régua, GLiNER v2.1
zero-shot, oráculo) são relidos pelo caminho dele (`bench.hipoteses.carregar`) e as listas finais
de spans por documento têm de ser iguais às deste script.

    docker compose run --rm -T -e PYTHONDONTWRITEBYTECODE=1 gama \\
        python -m bench.controles.gliner25.reserva_guarda
Saída: `saidas/bench/controles/gliner25/reserva/reserva_guarda.json`.
"""
from __future__ import annotations

import json
import pathlib

from gama.extratores.guarda import Guarda
from gama.formas import DetectorDeForma

from bench import pontuar
from bench.controles import avaliar as av
from bench.controles.verificador import nucleo as nu

G25 = av.SAIDA / "gliner25"
DESTINO = G25 / "reserva" / "reserva_guarda.json"
HUB25 = G25 / "_hub" / "bench" / "saida" / "controles" / "gliner25"
ARQ = {
    "gliner21_zs": {"reais": [av.BENCH / "gliner.jsonl"],                       # gravação do benchmark
                    "novas": [HUB25 / "modelo_gliner21_ref.jsonl"]},            # rodada no job A (exp. 1)
    "gliner25_zs": [HUB25 / "modelo_gliner25_zs.jsonl"],
    "gliner25_desc050": [G25 / "_hub_b" / "bench" / "saida" / "controles" / "gliner25"
                         / "modelo_gliner25_t1_desc_050.jsonl"],
    "gliner21_ft": [av.SAIDA / "_hub" / "bench" / "saida" / "controles" / "gliner_ft" / "modelo_gliner_ft.jsonl"],
    "gliner25_ft": [G25 / "ft" / "_hub" / "bench" / "saida" / "controles" / "gliner25" / "ft"
                    / "modelo_gliner25_ft.jsonl"],
}
ORDEM = ["nenhum", "regua", "gliner21_zs", "gliner25_zs", "gliner25_desc050", "gliner21_ft", "gliner25_ft",
         "oraculo"]
DESCRICAO = {
    "nenhum": "sem reserva: o span fraco só sai",
    "regua": "régua (produção; = Gama v1.3 com guarda)",
    "gliner21_zs": "GLiNER multi v2.1 zero-shot (urchade/gliner_multi-v2.1@443d26d)",
    "gliner25_zs": "GLiNER 2.5 zero-shot, principal (nomes, limiar 0,5)",
    "gliner25_desc050": "GLiNER 2.5 zero-shot, variante escolhida no exp. 1 (descrições, limiar 0,5)",
    "gliner21_ft": "GLiNER v2.1 fine-tunado no final_v3",
    "gliner25_ft": "GLiNER 2.5 fine-tunado no final_v3 (época 2, limiar 0,50)",
    "oraculo": "gabarito como reserva (teto)",
}
CONTRASTES = [("gliner25_zs", "gliner21_zs"), ("gliner25_desc050", "gliner21_zs"), ("gliner25_ft", "gliner21_ft"),
              ("gliner25_ft", "gliner25_zs")]


def _rel(p) -> str:
    return str(p).replace("/app/", "", 1)


def reservas(c: str, textos: dict, gold: dict) -> tuple[dict, dict]:
    """({nome: {doc: [Span]}}, {nome: arquivos}) do conjunto."""
    out, fontes = {"nenhum": {d: [] for d in textos}, "regua": av.regua(c)}, {"nenhum": [], "regua": [_rel(av.REGUA[c])]}
    for nome, arqs in ARQ.items():
        arqs = arqs[c] if isinstance(arqs, dict) else arqs
        _, linhas = av.ler(arqs)
        sp = av.spans(linhas, c)
        if sp is None:
            raise SystemExit(f"{nome}: conjunto {c} incompleto em {arqs}")
        out[nome], fontes[nome] = sp, [_rel(a) for a in arqs]
    out["oraculo"] = {d: [DetectorDeForma().span(textos[d], a, b, t, None) for a, b, t in gold[d]] for d in textos}
    fontes["oraculo"] = ["gabarito do conjunto"]
    return {k: out[k] for k in ORDEM}, fontes


def _chave(ss) -> list:
    return [(s.inicio, s.fim, s.tipo, s.forma) for s in ss]


def conjunto(c: str) -> dict:
    textos, gold = av.textos_e_ouro(c)
    gama = av.spans(pontuar._ler_spans(av.REFERENCIA[c])[1], c)
    base = {d: Guarda().aplicar(gama[d], []) for d in textos}                     # fortes (0,95) sem reserva
    g1 = {d: Guarda(confianca_minima=0.0).aplicar(gama[d], []) for d in textos}   # depois da regra 1
    fracos = sum(len(g1[d]) - len(base[d]) for d in textos)
    res, fontes = reservas(c, textos, gold)
    finais = {n: {d: Guarda().aplicar(gama[d], r[d]) for d in textos} for n, r in res.items()}
    pd = {n: av.por_doc(c, f) for n, f in finais.items()}
    ref = av.referencia(c)
    assert pd["regua"] == ref, "reserva régua != referência do harness (Gama v1.3 com guarda)"
    cru = av.extracao(c, gama)
    f1 = {n: av.extracao(c, f)["f1"] for n, f in finais.items()}
    linhas = {}
    for n, f in finais.items():
        e = av.extracao(c, f)
        ids_base = {d: {id(s) for s in base[d]} for d in textos}
        entraram = [(d, s) for d in textos for s in f[d] if id(s) not in ids_base[d]]
        casam = sum(nu.casa(s, gold[d], True) for d, s in entraram)
        gap = f1["oraculo"] - f1["nenhum"]
        linhas[n] = {
            "descricao": DESCRICAO[n], "fontes": fontes[n],
            "spans_no_reserva": sum(len(v) for v in res[n].values()),
            "f1": e["f1"], "f1_exato": round(e["f1_exato"], 4), "p": e["p"], "r": e["r"],
            "tp": e["tp"], "fp": e["fp"], "fn": e["fn"],
            "por_tipo": {t: {k: v[k] for k in ("tp", "fp", "fn", "f1")} for t, v in e["por_tipo"].items()},
            "spans_do_reserva_que_entraram": len(entraram),
            "entraram_que_casam_ouro": casam, "entraram_sem_par_no_ouro": len(entraram) - casam,
            "docs_diferentes_da_producao": sum(_chave(f[d]) != _chave(finais["regua"][d]) for d in textos),
            "fracao_do_ganho_do_oraculo": round((f1[n] - f1["nenhum"]) / gap, 4) if gap else None,
            "vs_gama_v13_guarda": av.bootstrap(pd[n], ref),
            "vs_nenhum": av.bootstrap(pd[n], pd["nenhum"]),
        }
    contrastes = {f"{a} - {b}": av.bootstrap(pd[a], pd[b]) for a, b in CONTRASTES}
    return {"documentos": len(textos), "gama_v13": _rel(av.REFERENCIA[c]),
            "gama_v13_cru_f1": cru["f1"], "spans_fracos_do_gama": fracos,
            "spans_fortes_do_gama": sum(len(v) for v in base.values()),
            "reservas": linhas, "contrastes": contrastes, "_finais": finais}


def conferir_original(finais: dict) -> dict:
    """Nas 305, os reservas do script original relidos pelo caminho dele dão os mesmos spans finais."""
    from bench.hipoteses import carregar
    textos, gold, gama = carregar("gama", "reais")
    orig = {"regua": carregar("regua", "reais")[2], "gliner21_zs": carregar("gliner", "reais")[2],
            "nenhum": {d: [] for d in textos},
            "oraculo": {d: [DetectorDeForma().span(textos[d], a, b, t, None) for a, b, t in gold[d]] for d in textos}}
    out = {}
    for n, r in orig.items():
        o = {d: Guarda().aplicar(gama[d], r.get(d, [])) for d in textos}
        out[n] = {"docs_diferentes": sum(_chave(o[d]) != _chave(finais[n][d]) for d in textos),
                  "f1_original": pontuar.extracao(gold, {d: [(s.inicio, s.fim, pontuar.rotulo(s)) for s in ss]
                                                         for d, ss in o.items()}, True)["f1"]}
    out["ok"] = all(v["docs_diferentes"] == 0 for k, v in out.items() if k != "ok")
    return out


def main() -> int:
    out = {"experimento": "reserva da guarda com o GLiNER 2.5 (3a)", "codigo": av.impressao(),
           "guarda": "gama.extratores.guarda.Guarda, CONFIANCA_MINIMA 0,95 (produção), VAGA_COLADA 2",
           "metrica": "F1 de extração, mesmo tipo, IoU >= 0,5, 1 para 1, VAGA em JURIS",
           "bootstrap": "pareado por ementa, 2.000 reamostras, semente 0 (avaliar.bootstrap)",
           "conjuntos": {}}
    finais = {}
    for c in ("reais", "novas"):
        r = conjunto(c)
        finais[c] = r.pop("_finais")
        out["conjuntos"][c] = r
        print(f"== {c} ({r['documentos']} docs)  Gama v1.3 cru F1 {r['gama_v13_cru_f1']:.4f}  "
              f"fracos {r['spans_fracos_do_gama']}  fortes {r['spans_fortes_do_gama']}", flush=True)
        for n, x in r["reservas"].items():
            b = x["vs_gama_v13_guarda"]
            print(f"  {n:17s} F1 {x['f1']:.4f}  entraram {x['spans_do_reserva_que_entraram']:3d} "
                  f"(casam {x['entraram_que_casam_ouro']:3d})  vs régua {b['dif']:+.4f} {b['ic95']}  "
                  f"ganho/oráculo {x['fracao_do_ganho_do_oraculo']}", flush=True)
        for k, b in r["contrastes"].items():
            print(f"  {k:32s} {b['dif']:+.4f} {b['ic95']}", flush=True)
    out["conferencia_original_305"] = conferir_original(finais["reais"])
    print("conferência com o script original (305):", out["conferencia_original_305"], flush=True)
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    DESTINO.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", DESTINO)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
