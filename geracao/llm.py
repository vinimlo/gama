# -*- coding: utf-8 -*-
"""Cliente mínimo do Ollama (modelos de pesos abertos na nuvem do Ollama).

Só para GERAR DADO de treino — nunca roda na solução avaliada. Licenças dos três
geradores verificadas em 22/09: nenhuma restringe
usar a saída para treino.

Cache em disco por hash da requisição: rodar de novo devolve a mesma resposta,
então o goldenset é reprodutível a partir do cache (que vai junto na publicação).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
import urllib.error
import urllib.request

MODELOS = {
    "deepseek": "deepseek-v4-pro:cloud",
    "glm": "glm-5.3:cloud",
    "kimi": "kimi-k3:cloud",
}
HOST = os.environ.get("OLLAMA_HOST_URL", "http://host.docker.internal:11434")
CACHE = pathlib.Path(os.environ.get("GAMA_LLM_CACHE", "/app/corpus/llm_cache"))


def chat(modelo: str, mensagens: list[dict], temperatura: float = 0.9, semente: int = 0,
         json_saida: bool = True, tentativas: int = 8, timeout: int = 600) -> str:
    # O GLM-5.3 ignora think=false e escreve o raciocínio DENTRO da resposta, que
    # estoura o teto antes do JSON. Com think=true o Ollama separa o raciocínio em
    # outro campo; o teto sobe para caber os dois.
    pensa = modelo == "glm"
    corpo = {
        "model": MODELOS.get(modelo, modelo),
        "messages": mensagens,
        "stream": False,
        "think": pensa,
        # Teto de saída: o GLM-5.3 em modo JSON a 0,9 entrou em laço de repetição
        # (28 mil tokens, 81 mil caracteres). A maior resposta legítima tem ~1.500.
        "options": {"temperature": temperatura, "seed": semente,
                    "num_predict": 16000 if pensa else 4096},
    }
    if json_saida:
        corpo["format"] = "json"
    chave = hashlib.sha256(json.dumps(corpo, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    arq = CACHE / f"{chave}.json"
    if arq.exists():
        return json.loads(arq.read_text(encoding="utf-8"))["resposta"]
    erro = None
    for t in range(tentativas):
        try:
            req = urllib.request.Request(f"{HOST}/api/chat", data=json.dumps(corpo).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            resposta = d["message"]["content"]
            CACHE.mkdir(parents=True, exist_ok=True)
            arq.write_text(json.dumps({"requisicao": corpo, "resposta": resposta,
                                       "eval_count": d.get("eval_count")}, ensure_ascii=False),
                           encoding="utf-8")
            return resposta
        except urllib.error.HTTPError as e:
            erro = e
            # 429 (limite de taxa do provedor): recuo exponencial, até 2 min por espera
            time.sleep(min(120, 10 * 2 ** t) if e.code == 429 else 5 * (t + 1))
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            erro = e
            time.sleep(5 * (t + 1))
    raise RuntimeError(f"{modelo}: {erro}")
