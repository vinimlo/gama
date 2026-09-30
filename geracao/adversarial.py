# -*- coding: utf-8 -*-
"""Validação adversarial: o goldenset sintético é distinguível dos documentos da organização?

Classificador TF-IDF (palavras 1-2 + caracteres 3-5) + regressão logística tenta
separar documentos nossos dos 26 da organização, em validação cruzada repetida.
AUC ~0,5 = indistinguível. Alvo: AUC <= 0,70. Com 26 positivos a AUC é
ruidosa — reportamos a média e o desvio entre repetições, e as features que mais
denunciam cada lado (é delas que sai a correção do gerador).

    docker compose run --rm lab python -m geracao.adversarial --sint /app/corpus/goldenset/v0
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline, make_union


def _modelo():
    feats = make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                        max_features=60000),
    )
    return make_pipeline(feats, LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="liblinear"))


def avaliar(org: list[str], sint: list[str], repeticoes: int = 5, semente: int = 0) -> dict:
    X = org + sint
    y = np.array([1] * len(org) + [0] * len(sint))
    aucs = []
    for r in range(repeticoes):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=semente + r)
        p = np.zeros(len(X))
        for tr, te in skf.split(X, y):
            m = _modelo().fit([X[i] for i in tr], y[tr])
            p[te] = m.predict_proba([X[i] for i in te])[:, 1]
        aucs.append(roc_auc_score(y, p))
    m = _modelo().fit(X, y)
    uniao, lr = m.steps[0][1], m.steps[1][1]
    nomes = np.concatenate([t.get_feature_names_out() for _, t in uniao.transformer_list])
    coef = lr.coef_[0]
    ordem = np.argsort(coef)
    return {"auc_media": float(np.mean(aucs)), "auc_dp": float(np.std(aucs)), "aucs": aucs,
            "denuncia_org": [(nomes[i], round(float(coef[i]), 2)) for i in ordem[::-1][:25]],
            "denuncia_sint": [(nomes[i], round(float(coef[i]), 2)) for i in ordem[:25]]}


def honesto(dados: pathlib.Path, db: str, dobras: int = 2, por_dobra: int = 130,
            semente: int = 0, extras: list | None = None) -> dict:
    """AUC sem vazamento: os bancos de cada dobra excluem os documentos da org avaliados.

    Com bancos do dev inteiro, o documento da org deixado de fora na validação
    cruzada tem frases que só ele tinha — e elas aparecem nos sintéticos rotulados
    como "sintético". O classificador aprende isso ao contrário e a AUC cai abaixo de
    0,5 por vazamento, não por qualidade.
    """
    from gama.indice import Indice
    from . import fichas as F
    from .bancos import extrair
    from .gerar import gerar_um
    from .montar import Montador, mistura_do_dev

    idx = Indice.do_banco(db)
    fichas, sumulas, disps = F.carregar(db)
    mistura = mistura_do_dev(dados)
    docs = sorted((dados / "txt").glob("*.txt"))
    random.Random(semente).shuffle(docs)
    resultados = []
    for d in range(dobras):
        fora = docs[d::dobras]
        bancos = extrair(dados, excluir={p.stem for p in fora})
        for extra in extras or []:                       # expansões por LLM
            for k, v in json.loads(pathlib.Path(extra).read_text(encoding="utf-8")).items():
                bancos.setdefault(k, []).extend(v)
        m = Montador(bancos, fichas, sumulas, disps, mistura, idx)
        sint = []
        i = 0
        while len(sint) < por_dobra:
            nivel = 1 if i % 2 == 0 else 2
            doc, laudo = gerar_um(m, i, 1000 + d, nivel, idx)
            i += 1
            if not laudo.fatal:
                sint.append(doc.texto)
        r = avaliar([p.read_text(encoding="utf-8") for p in fora], sint, semente=semente)
        resultados.append(r)
        print(f"dobra {d}: AUC {r['auc_media']:.3f} ± {r['auc_dp']:.3f} ({len(fora)} org × {len(sint)})")
    aucs = [r["auc_media"] for r in resultados]
    return {"auc_media": float(np.mean(aucs)), "dobras": aucs,
            "denuncia_org": resultados[0]["denuncia_org"], "denuncia_sint": resultados[0]["denuncia_sint"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", default="/app/dados/txt")
    ap.add_argument("--sint", help="pasta de um goldenset já gerado (medida com vazamento)")
    ap.add_argument("--honesto", action="store_true", help="bancos sem os docs avaliados")
    ap.add_argument("--db", default="/app/dados/desafio1_bracis.db")
    ap.add_argument("--extras", default="", help="bancos extras (LLM), separados por vírgula")
    ap.add_argument("--amostra", type=int, default=260, help="documentos sintéticos (10x a org)")
    a = ap.parse_args()
    if a.honesto:
        r = honesto(pathlib.Path(a.org).parent, a.db,
                    extras=[x for x in a.extras.split(",") if x])
        print(f"AUC honesta {r['auc_media']:.3f}  dobras {[round(x, 3) for x in r['dobras']]}")
        print("denuncia ORG :", r["denuncia_org"])
        print("denuncia SINT:", r["denuncia_sint"])
        return 0
    org = [p.read_text(encoding="utf-8") for p in sorted(pathlib.Path(a.org).glob("*.txt"))]
    todos = sorted((pathlib.Path(a.sint) / "txt").glob("*.txt"))
    random.Random(0).shuffle(todos)
    sint = [p.read_text(encoding="utf-8") for p in todos[: a.amostra]]
    r = avaliar(org, sint)
    print(f"AUC {r['auc_media']:.3f} ± {r['auc_dp']:.3f}  ({len(org)} org × {len(sint)} sintéticos)")
    print("denuncia ORG :", r["denuncia_org"])
    print("denuncia SINT:", r["denuncia_sint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
