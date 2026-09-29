# -*- coding: utf-8 -*-
"""Antes de treinar: `treino/treinar.py` e o `ExtratorNeural` janelam o BERTimbau certo?

Um BERT só aceita 512 posições, então quase todo documento vira várias janelas. O que pode
quebrar em silêncio: tokens especiais ([CLS]/[SEP]) no lugar errado, janela com mais de 512
posições, rótulo deslocado em relação ao token, offsets que não voltam ao texto, e o recorte
dos logits de cada janela (`lg[pre:pre + (b - a)]`) errar por um. Nada aqui é reimplementado:
o teste chama `treinar.carregar`, `treinar.especiais`, `treinar.exemplos`, `treinar.prever`,
`ExtratorNeural.__init__`/`_logits` e `bio.rotular`/`decodificar` como estão.

1. Tokenizador: rápido (offsets), caixa e acento preservados, especiais descobertos.
2. Ida e volta `rotular -> decodificar` sem janela, no treino (final_v3 @31474b1), no
   estresse, nas 305 e nas 172, para o BERTimbau e para o mmBERT (teto de cada tokenizador).
3. Janelas do treino (`treinar.exemplos`, max_len 512): tamanho, [CLS] no início, [SEP] no
   fim, especiais com rótulo -100, corpo igual ao rótulo do documento, cobertura total.
4. Modelo perfeito falso (devolve o rótulo-ouro de cada token e rótulo errado nos especiais)
   passado pelas janelas de `ExtratorNeural._logits` e `treinar.prever`: o rótulo por token
   e os spans têm de sair iguais aos do documento inteiro sem janela.
5. BERT de verdade (pequeno, pesos aleatórios, config do BERTimbau com 512 posições) por
   `ExtratorNeural`: roda documento longo sem erro; 513 posições quebram (o teto importa).

    docker compose run --rm -e HF_HOME=/app/saidas/bench/controles/bertimbau/hf lab \\
        python -m bench.controles.bertimbau.checar_janelas
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import tempfile
import types

import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from gama.extratores import bio
from gama.extratores.neural import ExtratorNeural
from treino import treinar

from .. import avaliar

BERT = ("neuralmind/bert-base-portuguese-cased", "94d69c95f98f7d5b2a8700c420230ae10def0baa")
MMBERT = ("jhu-clsp/mmBERT-base", "c5955035435e2bf121cde7f3c8863ef52ff35d82")
AQUI = pathlib.Path("/app/saidas/bench/controles/bertimbau")
DADOS = ("vinimlo/gama-goldenset", "31474b1f7db9096c4ca4f2a4eae2e9b82852d7a7")   # o do treino do v1.2
DS = pathlib.Path("/app/corpus/goldenset/v3")    # cópia local; conferida arquivo a arquivo com o Hub
MAX_LEN = 512


def conferir_dados() -> dict:
    """A cópia local = `final_v3` do dataset na revisão do treino? Compara o blob git (ou o sha256
    do LFS) de cada arquivo com a listagem do Hub, sem baixar 6.000 arquivos (o Hub devolve 429)."""
    from huggingface_hub import HfApi
    remotos = {f.path[len("final_v3/"):]: f for f in HfApi().list_repo_tree(
        DADOS[0], path_in_repo="final_v3", recursive=True, revision=DADOS[1], repo_type="dataset", expand=True)
        if hasattr(f, "blob_id")}
    locais = {str(p.relative_to(DS)): p for p in DS.rglob("*") if p.is_file()}
    diferentes = []
    for nome, f in remotos.items():
        if nome not in locais:
            diferentes.append(("falta", nome))
            continue
        b = locais[nome].read_bytes()
        if f.lfs:
            ok = hashlib.sha256(b).hexdigest() == f.lfs.sha256
        else:
            ok = hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() == f.blob_id
        if not ok:
            diferentes.append(("conteudo", nome))
    usados = [n for n in locais if n.startswith("txt/") or n in ("goldenset_offsets.csv", "meta.jsonl")]
    return {"hub": len(remotos), "local": len(locais), "diferentes": diferentes[:20], "n_diferentes": len(diferentes),
            "locais_fora_do_hub": sorted(set(usados) - set(remotos))[:20]}


def _ida_e_volta(tok, texto: str, spans: list) -> tuple[list, list, list]:
    enc = tok(texto, return_offsets_mapping=True, add_special_tokens=False)
    offs = [tuple(o) for o in enc["offset_mapping"]]
    rot = bio.rotular(offs, spans)
    return enc["input_ids"], offs, rot


def ida_e_volta(tok, docs: dict) -> dict:
    """{doc: (texto, spans)} -> spans do ouro que não voltam idênticos, spans a mais."""
    total = faltam = sobram = unk_spans = unk_tokens = 0
    exemplos = []
    tokens = []
    unk_trechos = collections.Counter()
    for d, (texto, spans) in docs.items():
        ids, offs, rot = _ida_e_volta(tok, texto, spans)
        tokens.append(len(ids))
        # [UNK]: o modelo não lê o trecho (o BERTimbau não tem "º" depois de letra/dígito: "nº", "1º")
        unk = [(s, e) for i, (s, e) in zip(ids, offs) if i == tok.unk_token_id]
        unk_tokens += len(unk)
        for a, b, _ in spans:
            dentro = [(s, e) for s, e in unk if s < b and a < e]
            unk_spans += bool(dentro)
            unk_trechos.update(texto[s:e] for s, e in dentro)
        volta = set(bio.decodificar(offs, rot, texto))
        ouro = {tuple(s) for s in spans}
        total += len(ouro)
        f = ouro - volta
        faltam += len(f)
        sobram += len(volta - ouro)
        for a, b, t in sorted(f)[:1]:
            if len(exemplos) < 8:
                perto = [texto[x:y] for x, y, _ in volta if x < b and a < y]
                exemplos.append({"doc": d, "ouro": texto[a:b], "tipo": t, "voltou": perto})
    tokens.sort()
    return {"docs": len(docs), "spans": total, "nao_voltam": faltam, "a_mais": sobram, "exemplos": exemplos,
            "unk_tokens": unk_tokens, "spans_com_unk": unk_spans, "unk_mais_comuns": unk_trechos.most_common(8),
            "tokens_por_doc": {"mediana": tokens[len(tokens) // 2], "max": tokens[-1],
                               "media": round(sum(tokens) / len(tokens), 1)},
            "docs_acima_de_510": sum(n > MAX_LEN - 2 for n in tokens)}


class Falso:
    """Modelo perfeito: confere a janela recebida e devolve o rótulo-ouro de cada token do
    corpo. Nos especiais devolve B-JURIS com certeza, para um recorte errado aparecer."""

    def __init__(self, pre: list, suf: list, ids: list, rot: list, cob: list):
        self.fila = [(pre + ids[a:b] + suf, rot[a:b]) for a, b in cob]
        self.pre = len(pre)

    def __call__(self, input_ids, attention_mask):
        recebido = input_ids[0].tolist()
        esperado, rot = self.fila.pop(0)
        assert recebido == esperado, "janela diferente da esperada"
        assert len(recebido) <= MAX_LEN, len(recebido)
        assert attention_mask.shape == input_ids.shape
        lg = torch.full((len(recebido), len(bio.ROTULOS)), -20.0)
        lg[:, bio.ID["B-JURIS"]] = 20.0
        for p, r in enumerate(rot):
            lg[self.pre + p] = -20.0
            lg[self.pre + p, bio.ID["O"] if r == bio.IGNORAR else r] = 20.0
        return types.SimpleNamespace(logits=lg[None])


def _modelo_pequeno(pasta: pathlib.Path, tok) -> None:
    """BERT de 1 camada com a config do BERTimbau (512 posições), pesos aleatórios."""
    cfg = AutoConfig.from_pretrained(BERT[0], revision=BERT[1], num_labels=len(bio.ROTULOS),
                                     id2label=dict(enumerate(bio.ROTULOS)), label2id=bio.ID,
                                     num_hidden_layers=1, hidden_size=32, num_attention_heads=2,
                                     intermediate_size=64)
    torch.manual_seed(0)
    AutoModelForTokenClassification.from_config(cfg).save_pretrained(pasta)
    tok.save_pretrained(pasta)


def janelas_do_treino(tok, docs: dict) -> dict:
    pre, suf = treinar.especiais(tok)
    ex = treinar.exemplos(tok, docs, bio, MAX_LEN)
    ruins = collections.Counter()
    for e in ex:
        ids, lab = e["input_ids"], e["labels"]
        ruins["tamanho>512"] += len(ids) > MAX_LEN
        ruins["tamanho_labels"] += len(lab) != len(ids)
        ruins["cls_no_inicio"] += ids[:len(pre)] != pre
        ruins["sep_no_fim"] += ids[len(ids) - len(suf):] != suf
        ruins["especial_com_rotulo"] += any(x != bio.IGNORAR for x in lab[:len(pre)] + lab[len(lab) - len(suf):])
    # corpo = rótulo do documento; cobertura sem buraco (refeito por documento, na mesma ordem)
    k = 0
    corpo = MAX_LEN - len(pre) - len(suf)
    for d, (texto, spans) in docs.items():
        ids, offs, rot = _ida_e_volta(tok, texto, spans)
        cob = bio.janelas(len(ids), corpo, corpo // 2)
        vistos = set()
        for a, b in cob:
            e = ex[k]
            k += 1
            ruins["corpo_ids"] += e["input_ids"][len(pre):len(pre) + b - a] != ids[a:b]
            ruins["corpo_rotulos"] += e["labels"][len(pre):len(pre) + b - a] != rot[a:b]
            vistos.update(range(a, b))
        ruins["tokens_sem_janela"] += len(ids) - len(vistos)
    ruins["janelas_conferidas_de_ordem"] = k == len(ex)
    return {"janelas": len(ex), "corpo": corpo, "problemas": dict(ruins)}


def perfeito(tok, pasta: pathlib.Path, docs: dict, limite: int = 400) -> dict:
    """Passa o modelo perfeito pelas janelas da inferência e do treino nos docs mais longos."""
    ext = ExtratorNeural(pasta)                      # __init__ de verdade: pre/suf e teto de 512
    pre, suf = treinar.especiais(tok)
    out = {"max_len_extrator": ext.max_len, "pre_extrator": ext.pre, "suf_extrator": ext.suf,
           "pre_treino": pre, "suf_treino": suf}
    longos = sorted(docs, key=lambda d: -len(docs[d][0]))[:limite]
    # o tokenizador salvo com o modelo (o que a inferência carrega) tokeniza igual ao original
    out["tokenizador_salvo_diferente"] = sum(
        _ida_e_volta(ext.tok, t, [])[:2] != _ida_e_volta(tok, t, [])[:2] for t, _ in docs.values())
    diferentes_rot = diferentes_neural = diferentes_prever = janelas = 0
    corpo = ext.max_len - len(ext.pre) - len(ext.suf)
    for d in longos:
        texto, spans = docs[d]
        ids, offs, rot = _ida_e_volta(tok, texto, spans)
        cob = bio.janelas(len(ids), corpo, corpo // 2)
        janelas += len(cob)
        sem_janela = bio.decodificar(offs, [bio.ID["O"] if r == bio.IGNORAR else r for r in rot], texto)
        ext.modelo = Falso(ext.pre, ext.suf, ids, rot, cob)
        r_jan, conf = ext._logits(ids)
        esperado = [bio.ID["O"] if r == bio.IGNORAR else r for r in rot]
        diferentes_rot += r_jan != esperado or min(conf, default=1.0) < 0.999
        ext.modelo = Falso(ext.pre, ext.suf, ids, rot, cob)
        pelo_extrator = [(s.inicio, s.fim) for s in ext.extrair(texto)]
        diferentes_neural += pelo_extrator != [(a, b) for a, b, _ in sem_janela]
        pr = treinar.prever(Falso(pre, suf, ids, rot, cob), tok, texto, bio, MAX_LEN, "cpu")
        diferentes_prever += pr != sem_janela
    out.update({"docs": len(longos), "janelas": janelas, "rotulo_por_token_diferente": diferentes_rot,
                "spans_extrator_diferentes": diferentes_neural, "spans_prever_diferentes": diferentes_prever})
    # BERT de verdade (pesos aleatórios): o doc mais longo passa; 513 posições não
    ext = ExtratorNeural(pasta)
    texto = docs[longos[0]][0]
    out["bert_real_doc_longo_ok"] = isinstance(ext.extrair(texto), list)
    try:
        t = torch.ones((1, MAX_LEN + 1), dtype=torch.long)
        ext.modelo(input_ids=t, attention_mask=t)
        out["bert_real_513_quebra"] = False
    except Exception as e:                             # noqa: BLE001 - só registra o tipo
        out["bert_real_513_quebra"] = type(e).__name__
    return out


def main() -> int:
    res = {"bertimbau": {"repo": BERT[0], "revisao": BERT[1]}, "mmbert": {"repo": MMBERT[0], "revisao": MMBERT[1]}}
    tok = AutoTokenizer.from_pretrained(BERT[0], revision=BERT[1])
    tok_mm = AutoTokenizer.from_pretrained(MMBERT[0], revision=MMBERT[1])
    exemplo = "Súmula nº 7 do STJ; REsp 1.234.567/SP, Rel. Min. João Otávio; art. 927, § 1º, do CC."
    res["bertimbau"]["tokenizador"] = {
        "classe": type(tok).__name__, "rapido": tok.is_fast, "especiais": list(treinar.especiais(tok)),
        "cls_sep": [tok.cls_token_id, tok.sep_token_id],
        "tokens_exemplo": tok.convert_ids_to_tokens(tok(exemplo, add_special_tokens=False)["input_ids"]),
        "max_position_embeddings": AutoConfig.from_pretrained(BERT[0], revision=BERT[1]).max_position_embeddings}
    res["mmbert"]["tokenizador"] = {"classe": type(tok_mm).__name__, "rapido": tok_mm.is_fast,
                                    "especiais": list(treinar.especiais(tok_mm))}

    res["dados_conferidos"] = conferir_dados()
    print("dados", res["dados_conferidos"], flush=True)
    docs, split = treinar.carregar(DS)
    treino = {d: v for d, v in docs.items() if split.get(d, "treino") == "treino"}
    interno = {d: v for d, v in docs.items() if split.get(d) == "estresse"}
    res["dados"] = {"pasta": str(DS), "treino": len(treino), "split_interno": len(interno)}
    conjuntos = {"treino_final_v3": treino, "split_interno_final_v3": interno}
    for c in avaliar.CONJUNTOS:
        textos, ouro = avaliar.textos_e_ouro(c)
        conjuntos[c] = {d: (t, [tuple(s) for s in ouro.get(d, [])]) for d, t in textos.items()}
    for nome, tk in (("bertimbau", tok), ("mmbert", tok_mm)):
        res[nome]["ida_e_volta"] = {c: ida_e_volta(tk, ds) for c, ds in conjuntos.items()}
        print(nome, {c: (r["spans"], r["nao_voltam"], r["a_mais"], r["docs_acima_de_510"])
                     for c, r in res[nome]["ida_e_volta"].items()}, flush=True)

    res["bertimbau"]["janelas_treino_512"] = janelas_do_treino(tok, treino)
    res["mmbert"]["janelas_treino_1024"] = {"janelas": len(treinar.exemplos(tok_mm, treino, bio, 1024)),
                                            "nota": "o log do treino do v1.2 registra 9.688"}
    print("janelas", res["bertimbau"]["janelas_treino_512"], res["mmbert"]["janelas_treino_1024"], flush=True)

    with tempfile.TemporaryDirectory(dir=AQUI) as tmp:
        _modelo_pequeno(pathlib.Path(tmp), tok)
        res["bertimbau"]["modelo_perfeito"] = {
            c: perfeito(tok, pathlib.Path(tmp), conjuntos[c]) for c in ("treino_final_v3", "reais", "novas", "estresse")}
    print("perfeito", json.dumps(res["bertimbau"]["modelo_perfeito"], ensure_ascii=False), flush=True)

    destino = AQUI / "checar_janelas.json"
    destino.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", destino)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
