# -*- coding: utf-8 -*-
"""Avaliação local do verificador "gliner_decide" (container, CPU). O modelo roda só no HF Jobs
(`job.py`): a gliner2 não está na imagem, então aqui só entram os arquivos gravados.

    docker compose run --rm gama python -m bench.controles.verificador_treinado.gliner_decide <acao> [...]

    escolher --bruto B              a escolha da formulação zero-shot SÓ nas 305: `cmd_escolher` do CLI de
                                    `bench.controles.verificador`, sem alteração (maior AUROC sobre todos os
                                    candidatos das 305; empate -> a primeira na ordem declarada) -> escolha.json
    rodar --variante V --scores S   políticas A, B e C pelo `cmd_rodar` do mesmo CLI, sem alteração (tau nas
                                    305, medido nas 172 e no estresse), com o nome gliner_decide_<V> (zs|ft).
                                    A única diferença é a pasta: `nucleo.SAIDA` aponta para
                                    saidas/bench/controles/verificador_treinado/gliner_decide/.
    refazer                         `cmd_refazer` do mesmo CLI, na mesma pasta: refaz cada relatório do zero a
                                    partir dos arquivos gravados e repassa os spans finais pelo harness.
    tabela                          `cmd_tabela` -> resumo.json.
    auroc --variante V --validacao F
                                    AUROC da validação interna (scores do job, F) e das 305, 172 e estresse
                                    (do relatório gliner_decide_<V>.json) -> auroc_<V>.json

O dev (26 documentos, só local) não é medido: a checagem exige o modelo no container, e a gliner2 não
está na imagem (não se instala nada no container).
"""
from __future__ import annotations

import argparse
import json
import pathlib

from bench.controles import avaliar
from bench.controles.verificador import __main__ as vcli
from bench.controles.verificador import nucleo as vn

SAIDA = avaliar.SAIDA / "verificador_treinado" / "gliner_decide"


def _redirecionar() -> None:
    """Toda escrita do CLI do verificador (relatório, spans, JSON oficiais) vai para esta pasta."""
    SAIDA.mkdir(parents=True, exist_ok=True)
    vn.SAIDA = SAIDA


def _escrever(destino: pathlib.Path, obj) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", destino, flush=True)


def cmd_escolher(a) -> None:
    _redirecionar()
    vcli.cmd_escolher(argparse.Namespace(candidatos=str(vn.CANDIDATOS), bruto=a.bruto))


def cmd_rodar(a) -> None:
    _redirecionar()
    ns = argparse.Namespace(candidatos=str(vn.CANDIDATOS), nome=f"gliner_decide_{a.variante}", scores=a.scores,
                            campo="score", v0=False, politicas="A,B,C", sem_oficial=False, saida=None)
    vcli.cmd_rodar(ns)


def cmd_refazer(a) -> None:
    _redirecionar()
    vcli.cmd_refazer(argparse.Namespace())


def cmd_tabela(a) -> None:
    _redirecionar()
    vcli.cmd_tabela(argparse.Namespace())


def cmd_auroc(a) -> None:
    rel = json.loads((SAIDA / f"gliner_decide_{a.variante}.json").read_text(encoding="utf-8"))
    linhas = [json.loads(x) for x in pathlib.Path(a.validacao).read_text(encoding="utf-8").splitlines()]
    val = [r for r in linhas if "meta" not in r]
    out = {"validacao_interna": {"candidatos": len(val), "positivos": sum(r["rotulo"] for r in val),
                                 "auroc": vn.auroc([(r["score"], r["rotulo"]) for r in val])}}
    for c, nome in (("reais", "305"), ("novas", "172"), ("estresse", "estresse")):
        out[nome] = rel["auroc"][c]
    meta_val = next(r["meta"] for r in linhas if "meta" in r)
    out["validacao_interna"]["por_subconjunto_job"] = meta_val.get("metricas")
    out["arquivos"] = {"validacao": {"arquivo": a.validacao, "sha256": vn.sha256(a.validacao)},
                       "relatorio": f"gliner_decide_{a.variante}.json"}
    print(json.dumps({k: (v if k != "validacao_interna" else {kk: vv for kk, vv in v.items()
                                                                if kk != "por_subconjunto_job"})
                      for k, v in out.items()}, ensure_ascii=False), flush=True)
    _escrever(SAIDA / f"auroc_{a.variante}.json", out)


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.controles.verificador_treinado.gliner_decide",
                                 description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="acao", required=True)
    p = sub.add_parser("escolher")
    p.add_argument("--bruto", required=True)
    p = sub.add_parser("rodar")
    p.add_argument("--variante", required=True, choices=("zs", "ft"))
    p.add_argument("--scores", required=True)
    sub.add_parser("refazer")
    sub.add_parser("tabela")
    p = sub.add_parser("auroc")
    p.add_argument("--variante", required=True, choices=("zs", "ft"))
    p.add_argument("--validacao", required=True)
    a = ap.parse_args()
    {"escolher": cmd_escolher, "rodar": cmd_rodar, "refazer": cmd_refazer, "tabela": cmd_tabela,
     "auroc": cmd_auroc}[a.acao](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
