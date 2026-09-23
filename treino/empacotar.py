# -*- coding: utf-8 -*-
"""Monta o pacote de dados de treino e (opcionalmente) publica no Hub, privado.

    docker compose run --rm lab python -m treino.empacotar \\
        --goldenset /app/corpus/goldenset/v1 --saida /app/corpus/pacote/v1 \\
        --repo vinimlo/gama-goldenset

Estrutura (o que o treinar.py espera):
    codigo/bio.py      cópia do módulo de rótulos da inferência — treino e
                       inferência usam o MESMO código, fixado pela revisão
    goldenset/...      txt/, goldenset_offsets.csv, meta.jsonl, relatorio.json

Os dados da organização (dados/) NUNCA entram no pacote: redistribuição não
autorizada. O goldenset sintético contém frases de molde extraídas do dev set,
por isso o repositório é privado.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil

RAIZ = pathlib.Path(__file__).resolve().parent.parent


def montar(goldenset: pathlib.Path, saida: pathlib.Path, limite: int | None = None,
           nome: str = "goldenset") -> None:
    (saida / "codigo").mkdir(parents=True, exist_ok=True)
    shutil.copy(RAIZ / "src" / "gama" / "extratores" / "bio.py", saida / "codigo" / "bio.py")
    # Pacote inteiro (sem o acervo, que é dado da organização): o ensaio no hardware-alvo
    # roda o extrator de verdade, não uma cópia.
    destino_pkg = saida / "codigo" / "gama"
    if destino_pkg.exists():
        shutil.rmtree(destino_pkg)
    shutil.copytree(RAIZ / "src" / "gama", destino_pkg,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    destino = saida / nome
    if destino.exists():
        shutil.rmtree(destino)
    (destino / "txt").mkdir(parents=True)
    metas = [json.loads(x) for x in (goldenset / "meta.jsonl").read_text(encoding="utf-8").splitlines()]
    if limite:
        metas = metas[:limite]
    ids = {m["documento_id"] for m in metas}
    for d in sorted(ids):
        shutil.copy(goldenset / "txt" / f"{d}.txt", destino / "txt" / f"{d}.txt")
    # csv, não splitlines: o trecho pode ter "\n" dentro (campo entre aspas).
    with open(goldenset / "goldenset_offsets.csv", encoding="utf-8", newline="") as fe, \
            open(destino / "goldenset_offsets.csv", "w", encoding="utf-8", newline="") as fs:
        r, w = csv.reader(fe), csv.writer(fs)
        w.writerow(next(r))
        w.writerows(ln for ln in r if ln[1] in ids)
    (destino / "meta.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in metas),
                                        encoding="utf-8")
    rel = goldenset / "relatorio.json"
    if rel.exists():
        shutil.copy(rel, destino / "relatorio.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldenset", required=True,
                    help="pasta, ou lista nome=pasta,nome=pasta (vira subpasta do pacote)")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--limite", type=int, default=None)
    ap.add_argument("--repo", default=None, help="dataset privado no Hub")
    a = ap.parse_args()
    saida = pathlib.Path(a.saida)
    if saida.exists():
        shutil.rmtree(saida)
    for item in a.goldenset.split(","):
        nome, _, pasta = item.rpartition("=")
        montar(pathlib.Path(pasta), saida, a.limite, nome or "goldenset")
    print("pacote em", saida)
    if a.repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
        info = api.upload_folder(folder_path=str(saida), repo_id=a.repo, repo_type="dataset",
                                 commit_message=f"pacote {saida.name}")
        print("REVISAO", info.oid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
