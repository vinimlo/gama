# -*- coding: utf-8 -*-
"""Checagem no dev (26 documentos da organização): só aqui, em CPU, no container. Nada do dev sobe.

1. Gama v1.3 (`neural-cru`, /models) e régua sobre os textos do dev, aparados como o harness apara
   (`verificador_treinado.dados.nucleo.aparados`); candidatos pelo mesmo construtor dos dados de treino
   (`candidatos_doc`, mesmo estado).
2. Score de cada candidato com o verificador treinado, em CPU: camadas em fp32 e a tabela de embeddings
   em bf16 (o container tem 4 GB; bf16 puro em CPU é lento demais), a mesma entrada de `job.codificar`.
3. Políticas A, B e C com o tau escolhido nas 305 (lido do relatório do CLI) pela função do verificador
   (`nucleo.politica`); produção = `nucleo.politica_v0` com 0,95. Conta documentos mudados, F1 de
   extração (VAGA separada, como no estresse) e a nota oficial pelo caminho de produção
   (`bench.pontuar.oficial`, JSON por documento só no disco, nesta pasta).

    docker compose run --rm gama python -m bench.controles.verificador_treinado.decision_1_0_eos_0_8b.dev \\
        --pasta saidas/bench/controles/verificador_treinado/decision_1_0_eos_0_8b/_modelo/<subpasta>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from bench import conjuntos, pontuar
from bench.controles.verificador import nucleo as vn
from bench.controles.verificador_treinado.dados import nucleo as dn

from . import job as J
from .politicas import SAIDA

DEV = SAIDA / "dev"


def _linear_fp32(self, x):
    import torch
    return torch.nn.functional.linear(x, self.weight.float(), None if self.bias is None else self.bias.float())


def modelo_cpu(pasta: str):
    """Pesos guardados em bf16 (cabe no container) e conta em fp32: cada Linear sobe o peso para fp32 na
    hora (conversão exata), os parâmetros pequenos (normas, conv, A_log, dt_bias) viram fp32 e a saída
    da tabela de embeddings também."""
    import types

    import torch
    from transformers import AutoModelForSequenceClassification
    torch.set_num_threads(4)
    m = AutoModelForSequenceClassification.from_pretrained(pasta, dtype=torch.bfloat16).eval()
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear):
            mod.forward = types.MethodType(_linear_fp32, mod)
        elif mod is not m.model.embed_tokens:
            for p in mod.parameters(recurse=False):
                p.data = p.data.float()
            for nome, b in mod.named_buffers(recurse=False):
                if b.is_floating_point():
                    setattr(mod, nome, b.float())
    m.model.embed_tokens.register_forward_hook(lambda mod, i, o: o.float())
    return m


def extrair(textos: dict) -> dict:
    from gama.extratores import CatalogoDeExtratores
    out = {}
    catalogo = CatalogoDeExtratores("/models")
    for nome, ext in (("gama", catalogo.carregar("neural-cru")), ("regua", catalogo.carregar("regua"))):
        linhas = {}
        for d, t in textos.items():
            linhas[d] = [[s.inicio, s.fim, s.tipo, s.forma, s.digitos, s.confianca] for s in ext.extrair(t)]
        out[nome] = linhas
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", required=True, help="modelo salvo (save_pretrained + tokenizer)")
    ap.add_argument("--relatorio", default=str(SAIDA / f"{J.NOME}.json"))
    ap.add_argument("--limite", type=int, help="só N candidatos (ensaio de tempo; sem políticas)")
    a = ap.parse_args()
    import torch
    t_ini = time.perf_counter()
    textos, gold = conjuntos.carregar("dev")
    brutos = extrair(textos)
    cands, docs = [], {}
    for d, t in textos.items():
        G = dn.aparados(t, brutos["gama"][d])
        R = dn.aparados(t, brutos["regua"][d])
        regs = dn.candidatos_doc("dev", d, t, G, R)
        cands += regs
        docs[d] = ([vn.Cand(r["cid"], r["origem"], r["_span"]) for r in regs if r["origem"] == "gama"],
                   [vn.Cand(r["cid"], r["origem"], r["_span"]) for r in regs if r["origem"] == "regua"])
    print(f"dev: {len(textos)} documentos, {len(cands)} candidatos "
          f"({sum(r['origem'] == 'gama' for r in cands)} Gama)", flush=True)
    if a.limite:
        cands = cands[:a.limite]
    tok = J.tokenizer(a.pasta, None)
    modelo = modelo_cpu(a.pasta)
    ids, cortados = J.codificar(tok, cands)
    t0 = time.perf_counter()
    p = J.pontuar(modelo, ids, tok.pad_token_id, lote=8)
    seg = time.perf_counter() - t0
    score = {r["cid"]: s for r, s in zip(cands, p)}
    out = {"documentos": len(textos), "candidatos": len(cands), "cortados": cortados,
           "entradas_distintas": len(set(map(tuple, ids))), "segundos_cpu": round(seg, 1),
           "ms_por_candidato_cpu": round(1000 * seg / len(cands), 1),
           "cpu": "container, 4 vCPUs, camadas fp32 + embeddings bf16", "modelo": a.pasta}
    if a.limite:
        print(out, flush=True)
        return 0
    pares = [(score[r["cid"]], int(vn.casa(r["_span"], gold[r["doc"]], False))) for r in cands]
    out["auroc"] = {"candidatos": len(pares), "positivos": sum(y for _, y in pares), "auroc": vn.auroc(pares),
                    "rotulo": "casa com o gabarito do dev (mesmo tipo, IoU >= 0,5, VAGA separada)"}
    for origem in ("gama", "regua"):
        pp = [(s, y) for (s, y), r in zip(pares, cands) if r["origem"] == origem]
        out["auroc"][origem] = {"n": len(pp), "positivos": sum(y for _, y in pp), "auroc": vn.auroc(pp)}
    fortes = [score[r["cid"]] for r in cands if r["forte"]]
    out["fortes"] = {"n": len(fortes), "abaixo_de_0_5": sum(s < 0.5 for s in fortes),
                     "min": round(min(fortes), 4) if fortes else None}
    rel = json.loads(pathlib.Path(a.relatorio).read_text(encoding="utf-8"))
    taus = {pol: r["tau"] for pol, r in rel["politicas"].items()}
    prod = {d: vn.politica_v0(G, R, vn.FORTE) for d, (G, R) in docs.items()}
    from gama.indice import Indice
    idx = Indice.do_banco(pontuar.DB)
    antes = pontuar.SAIDA
    pontuar.SAIDA = DEV
    try:
        def medir(rotulo: str, finais: dict) -> dict:
            ex = pontuar.extracao(gold, {d: [(s.inicio, s.fim, pontuar.rotulo(s)) for s in ss] for d, ss in finais.items()},
                                  False)
            return {"docs_mudados_vs_producao": sum(vn._chave(finais[d]) != vn._chave(prod[d]) for d in finais),
                    "extracao": ex, "oficial": pontuar.oficial("dev", rotulo, textos, finais, idx)}
        out["producao"] = medir("producao_guarda@0.95", prod)
        out["politicas"] = {}
        for pol, tau in taus.items():
            finais = {d: vn.politica(pol, G, R, score, tau) for d, (G, R) in docs.items()}
            r = {"tau": tau, **medir(f"{J.NOME}__{pol}@{tau:.2f}", finais)}
            r["docs_mudados_por_tau"] = {f"{t:.2f}": sum(vn._chave(vn.politica(pol, G, R, score, t)) != vn._chave(prod[d])
                                                        for d, (G, R) in docs.items()) for t in vn.GRADE}
            out["politicas"][pol] = r
            print(pol, tau, r["docs_mudados_vs_producao"], r["oficial"].get("final"), flush=True)
    finally:
        pontuar.SAIDA = antes
    out["segundos_total"] = round(time.perf_counter() - t_ini, 1)
    DEV.mkdir(parents=True, exist_ok=True)
    destino = SAIDA / "dev.json"
    destino.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", destino, json.dumps({k: v for k, v in out.items() if k != "politicas"}, ensure_ascii=False)[:1500],
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
