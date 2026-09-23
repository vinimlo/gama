# 2026-09-22 · Validação por dobras: modelo avaliado em documentos da org que nunca viu

Pergunta: um extrator treinado só em sintético (moldes remontados) acerta documentos
da organização cujas frases ele não viu? E a expansão por LLM ajuda ou atrapalha?

## Desenho

- Os 26 documentos do dev set em 2 dobras de 13 (partição fixa, `bancos.dobras_do_dev`).
- Para cada dobra d: bancos extraídos só das outras 13, 1.500 documentos gerados,
  mmBERT-base fine-tunado (HF Jobs, A10G, 3 épocas, max_len 1024, semente 13).
- Variantes: org (só bancos da org) × llm (org + expansão v1: DeepSeek e Kimi).
- Avaliação: pipeline inteiro (extrator → resolver → classificação) nos 13 documentos da
  dobra d, métrica oficial (`vendor/kaggle_metric.py` via harness).
- Dataset `vinimlo/gama-goldenset@93c7b5f`; modelos `vinimlo/gama-d{d}-{v}`.

## Resultado

| Dobra | Régua (ajustada a estes docs) | neural org | neural org+LLM | união org+LLM |
|---|---|---|---|---|
| 0 | 1,09857 | 1,07822 (N2 incompleta 0,917) | 1,09857 | 1,09857 |
| 1 | 1,09840 | (pendente) | 1,09840 | 1,09840 |

Segunda arquitetura (BERTimbau-base, 512 tokens com janelas), mesmo protocolo:

| Dobra | neural org | neural org+LLM |
|---|---|---|
| 0 | 1,06927 | 1,09857 |
| 1 | 1,07786 | 1,09840 |

O efeito da expansão por LLM se repete nas duas arquiteturas e nas duas dobras.
mmBERT e BERTimbau empatam no teto: o dev não os separa.

Estresse sintético (10% do goldenset, nunca treinado): F1 exato = 1,000 nos três tipos.

org+LLM: 26/26 documentos, 192/192 citações certas, F1 = 1,000 em toda classe, τ = 0,
sem o modelo ter visto nenhum desses documentos.

## Leitura

1. O gerador por moldes transfere. Treinar em documentos remontados dos moldes deles
   generaliza para documentos da organização com frases fora do banco.
2. A expansão por LLM ajuda o modelo (dobra 0: 1,078 → 1,099), mesmo tendo PIORADO a
   AUC adversarial (0,825 → 0,894). As duas métricas medem coisas diferentes: a AUC mede
   se o texto parece o deles; a nota nas dobras mede se o modelo extrai melhor. A
   variedade de contexto ensina o modelo a não depender de frase carregadora específica.
   Decisão: treino final com org + LLM.
3. União = neural nas duas dobras: a régua não acrescenta nada onde o modelo já acerta
   (e a nota dela nas dobras é otimista, por ter sido ajustada a esses documentos).
4. O dev set saturou como seletor: tudo empata em F1 = 1,0. Escolher entre candidatos
   (mmBERT × BERTimbau, neural × união) pede um estresse mais difícil.
5. O que falta para 1,10000 é calibração. O bônus 0,1·(1 − Brier) está em 0,0984–0,0987
   porque as confianças vêm da tabela fixa de `classificar.py` (0,90–0,95). Com acerto
   ~100%, o ótimo do Brier é confiança ~0,995. No topo do cego, onde várias equipes podem
   ter F1 = 1,0, essa diferença decide o pódio. Ver [D-006](../decisoes/D-006_calibracao-decide-o-topo.md).
