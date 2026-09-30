# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "laya==0.3.22",
#   "torch==2.14.0",
#   "transformers==5.17.0",
#   "safetensors",
#   "huggingface_hub",
#   "numpy",
# ]
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cu126" }
# ///
# -*- coding: utf-8 -*-
"""Score do Laya fine-tunado (laya_ft) para cada candidato do verificador, no HF Jobs (L4).

Carrega os pesos gravados em vinimlo/gama-exp-verif-laya_ft (revisão dada) pelo carregador público
`laya.Agent` e dá a cada um dos 13.664 candidatos de `candidatos.jsonl` (vinimlo/gama-goldenset,
bench/saida/controles/verificador/, revisão e sha256 conferidos) o score P(A) da pergunta f3 com a
temperatura do checkpoint, sem arredondar: `Agent._encode_state` + `collate_items` +
`Agent._forward` e a mesma conta de `Agent._decode_answers` (a saída pública arredonda para 4
casas; confere-se que ela bate com o score arredondado). Precisão: fp32 (autocast desligado),
para o score ser o mesmo que a cópia mínima dá em CPU no container. Também pontua a validação
interna (validacao.jsonl dos dados, revisão fixa) para a AUROC.

Tempo: (1) fp32 em lotes de 64 estados ordenados por tamanho, total / candidatos; (2) uma
chamada `Agent.predict` por estado, com o runtime padrão do Laya (bf16 na L4), nos primeiros
`--latencia` estados, como o zero-shot mediu (ms por estado), e a diferença para o fp32.

Saídas: bench/saida/controles/verificador_treinado/laya_ft.jsonl em vinimlo/gama-goldenset
(linha 1 meta; depois {"cid", "score"}) e avaliacao/validacao_scores.jsonl no repositório do modelo.

    hf jobs uv run --flavor l4x1 --timeout 30m --secrets HF_TOKEN \\
        bench/controles/verificador_treinado/laya_ft/inferir_job.py --modelo-revisao <oid>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import time

MODELO = "vinimlo/gama-exp-verif-laya_ft"
GOLDEN = "vinimlo/gama-goldenset"
CANDIDATOS = "bench/saida/controles/verificador/candidatos.jsonl"
CANDIDATOS_REVISAO = "de07c938a68cc43779f7226e5194c49807b497f2"
CANDIDATOS_SHA = "c3745f168dd45df4f8914092df779bdbe00112995c14a69bbe9683d33c48ed6e"
DADOS = "vinimlo/gama-exp-verificador-dados"
DADOS_REVISAO = "24eede0c83a8530b2d6531e5000d7a0abf66f7f6"
VALIDACAO_SHA = "3fa89d9c824fd1b3103cf325935991880a3df2ab70607d0ad44f8fac847e305d"
SAIDA = "bench/saida/controles/verificador_treinado/laya_ft.jsonl"
LOTE = 64


def _sha(caminho) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo-revisao", required=True)
    ap.add_argument("--latencia", type=int, default=1000, help="estados medidos com uma chamada cada")
    ap.add_argument("--limite", type=int, help="ensaio: só os N primeiros estados")
    ap.add_argument("--sem-upload", action="store_true")
    a = ap.parse_args()

    os.environ.setdefault("USE_TF", "0")
    import numpy as np
    import torch
    import transformers
    import laya
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from laya.common import QTYPES, collate_items, temp_bucket

    pasta = snapshot_download(MODELO, revision=a.modelo_revisao, allow_patterns=[
        "rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*", "treino.json"])
    treino = json.loads(pathlib.Path(pasta, "treino.json").read_text(encoding="utf-8"))
    pergunta = treino["pergunta"]
    arq_c = pathlib.Path(hf_hub_download(GOLDEN, CANDIDATOS, repo_type="dataset", revision=CANDIDATOS_REVISAO))
    arq_v = pathlib.Path(hf_hub_download(DADOS, "validacao.jsonl", repo_type="dataset", revision=DADOS_REVISAO))
    for arq, esperado in ((arq_c, CANDIDATOS_SHA), (arq_v, VALIDACAO_SHA)):
        if _sha(arq) != esperado:
            raise SystemExit(f"{arq.name}: sha256 diferente de {esperado}")
    cands, meta_c = [], {}
    for linha in arq_c.read_text(encoding="utf-8").splitlines():
        r = json.loads(linha)
        if "meta" in r:
            meta_c = r["meta"]
        else:
            cands.append(r)
    valid = [json.loads(x) for x in arq_v.read_text(encoding="utf-8").splitlines() if x.strip()]
    estados_c = list(dict.fromkeys(r["estado"] for r in cands))[: a.limite]
    estados_v = [e for e in dict.fromkeys(r["estado"] for r in valid) if e not in set(estados_c)][: a.limite]
    estados = estados_c + estados_v
    print(f"candidatos {len(cands)} ({len(estados_c)} estados distintos) validação {len(valid)} "
          f"(+{len(estados_v)} estados)", flush=True)

    t0 = time.perf_counter()
    ag = laya.Agent(pasta, device="cuda")
    carga = time.perf_counter() - t0
    dtype_padrao, amp_padrao = ag.dtype, ag.amp_enabled
    ids = ["f3"]
    ag._check_question("f3", pergunta)
    internal = {"f3": ag._to_internal(pergunta)}
    k_opc = 2
    t_escala = ag.temperature_by_options.get(temp_bucket(QTYPES["choice"], k_opc), ag.temperature[QTYPES["choice"]])
    print(f"carga {carga:.1f}s dtype padrão {dtype_padrao} temperatura {t_escala}", flush=True)

    # ------------------------------------------------------------ fp32, em lotes
    ag.amp_enabled = False

    @torch.no_grad()                                  # como em Agent.predict_batch
    def pontuar(lista: list) -> tuple[dict, dict]:
        t0 = time.perf_counter()
        cod = [ag._encode_state(e, ids, internal) for e in lista]
        t_cod = time.perf_counter() - t0
        ordem = sorted(range(len(lista)), key=lambda i: len(cod[i][0]["ids"]))
        out = {}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for k in range(0, len(ordem), LOTE):
            idx = ordem[k:k + LOTE]
            b = collate_items([cod[i] for i in idx], ag.tok.pad_token_id)
            logits, _ = ag._forward(b)
            for j, i in enumerate(idx):
                z = logits[j, :k_opc].astype(np.float64) / t_escala
                p = np.exp(z - z.max())
                out[lista[i]] = float(p[0] / p.sum())
        torch.cuda.synchronize()
        return out, {"estados": len(lista), "segundos": round(time.perf_counter() - t0, 3),
                     "segundos_tokenizacao": round(t_cod, 3),
                     "truncados": sum(bool(it[0]["state_stats"]["truncated"]) for it in cod)}

    sc, t_c = pontuar(estados_c)
    sc_v, t_v = pontuar(estados_v)
    sc.update(sc_v)
    trunc = t_c["truncados"] + t_v["truncados"]
    print(f"fp32: candidatos {t_c} validação {t_v}", flush=True)

    # saída pública (arredonda para 4 casas) num pedaço: tem de bater com o score arredondado
    amostra = estados[:512]
    pub = ag.predict_batch(amostra, {"f3": pergunta}, batch_size=LOTE)
    dif_pub = max(abs(r["answers"]["f3"]["probabilities"]["A"] - round(sc[e], 4)) for e, r in zip(amostra, pub))
    print(f"predict_batch público x score: diferença máxima {dif_pub:.2e}", flush=True)
    if dif_pub > 2e-4:
        raise SystemExit("o score não reproduz a saída pública do laya.Agent")

    # ------------------------------------------------------------ uma chamada por estado, runtime padrão (bf16)
    ag.amp_enabled, ag.dtype = amp_padrao, dtype_padrao
    lat = estados[: a.latencia]
    for e in lat[:20]:
        ag.predict(e, {"f3": pergunta})                  # aquecimento
    difs, t_lat = [], []
    for e in lat:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        r = ag.predict(e, {"f3": pergunta})
        t_lat.append(time.perf_counter() - t1)
        difs.append(r["answers"]["f3"]["probabilities"]["A"] - sc[e])
    t_lat.sort()
    grade = [round(0.05 * i, 2) for i in range(1, 20)]
    cruza = sum(any((sc[e] >= t) != (sc[e] + d >= t) for t in grade) for e, d in zip(lat, difs))
    latencia = {"estados": len(lat), "dtype": str(dtype_padrao), "ms_media": round(1000 * sum(t_lat) / len(t_lat), 2),
                "ms_mediana": round(1000 * t_lat[len(t_lat) // 2], 2), "ms_p95": round(1000 * t_lat[int(0.95 * len(t_lat))], 2),
                "dif_max_bf16_menos_fp32": round(max(map(abs, difs)), 5),
                "estados_que_mudam_de_lado_em_algum_tau_da_grade": cruza}
    print("latência", latencia, flush=True)

    meta = {"verificador": "laya_ft", "modelo": f"{MODELO}@{a.modelo_revisao}", "laya": laya.__version__,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "dispositivo": torch.cuda.get_device_name(0), "python": platform.python_version(),
            "pergunta": pergunta, "score": "P(A) da pergunta f3 com a temperatura do checkpoint, fp32, sem arredondar",
            "temperatura": t_escala, "lr_encoder": treino["escolhida"]["lr_encoder"],
            "candidatos": {"arquivo": CANDIDATOS, "repo": GOLDEN, "revisao": CANDIDATOS_REVISAO, "sha256": CANDIDATOS_SHA,
                           "meta": {k: v for k, v in meta_c.items() if k != "codigo"}},
            "estados": {"candidatos": len(estados_c), "so_validacao": len(estados_v)},
            "estados_truncados": trunc, "segundos_carga": round(carga, 2),
            "tempo": {"fp32_lote": {"lote": LOTE, "candidatos": t_c, "validacao": t_v,
                                    "ms_por_estado": round(1000 * t_c["segundos"] / max(1, len(estados_c)), 3),
                                    "ms_por_candidato": round(1000 * t_c["segundos"] / len(cands), 3),
                                    "nota": "forward em lote dos estados distintos dos 13.664 candidatos (um estado "
                                            "serve a todos os candidatos com as mesmas bordas); sem a tokenização"},
                      "uma_chamada_por_estado": latencia},
            "predict_batch_publico_dif_max": dif_pub}
    faltam = [r["cid"] for r in cands if r["estado"] not in sc]
    if a.limite:
        print("ensaio: sem gravação", {k: v for k, v in meta.items() if k in ("tempo", "estados")}, flush=True)
        return 0
    if faltam:
        raise SystemExit(f"{len(faltam)} candidatos sem score")
    destino = pathlib.Path("/tmp/laya_ft.jsonl")
    with destino.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for r in cands:
            fh.write(json.dumps({"cid": r["cid"], "score": sc[r["estado"]]}) + "\n")
    dv = pathlib.Path("/tmp/validacao_scores.jsonl")
    with dv.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": {**meta, "arquivo": "validacao.jsonl", "dados": f"{DADOS}@{DADOS_REVISAO}",
                                      "sha256": VALIDACAO_SHA}}, ensure_ascii=False) + "\n")
        for r in valid:
            fh.write(json.dumps({"cid": r["cid"], "score": sc[r["estado"]], "rotulo": r["rotulo"]}) + "\n")
    print("sha256 laya_ft.jsonl", _sha(destino), "validacao_scores.jsonl", _sha(dv), flush=True)
    if a.sem_upload:
        return 0
    api = HfApi()
    info = api.upload_file(path_or_fileobj=str(destino), path_in_repo=SAIDA, repo_id=GOLDEN, repo_type="dataset",
                           commit_message="verificador treinado: laya_ft.jsonl")
    print("REVISAO_SCORES", info.oid, flush=True)
    info = api.upload_file(path_or_fileobj=str(dv), path_in_repo="avaliacao/validacao_scores.jsonl", repo_id=MODELO,
                           repo_type="model", commit_message="validação interna: scores dos pesos gravados")
    print("REVISAO_MODELO_VALIDACAO", info.oid, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
