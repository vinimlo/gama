# 2026-09-22 · Estresse difícil: frases nunca treinadas e ruído forte

Por quê: o dev set saturou (todo candidato bom tira F1 = 1,0 nele), então não serve
mais para escolher. O risco real do cego é redação que não vimos.

## Conjunto

`corpus/goldenset/estresse_glm` (dataset `vinimlo/gama-goldenset@4877c6a`, subpasta
`estresse_glm`): 600 documentos, 4.447 citações. Bancos da org + frases só do GLM-5.3
(que nenhum modelo viu no treino, feito com DeepSeek e Kimi), 90% de frase LLM,
ruído de nível 2 normal. Nunca treinado.

## Extração (F1 por tipo; GPU, HF Jobs a10g-small, `treino/avaliar_extracao.py`)

| Extrator | IoU ≥ 0,5 JURIS / LEI / VAGA | Borda exata JURIS / LEI / VAGA | Erros (fp+fn, exato) |
|---|---|---|---|
| Gama final (`vinimlo/gama@2aba5d1`, 4.000 docs, bancos completos) | 1,000 / 1,000 / 1,000 | 0,9997 / 1,000 / 1,000 | 2 |
| mmBERT dobra 0, org+LLM | 1,000 / 1,000 / 0,9993 | 1,000 / 1,000 / 0,9993 | 1 |
| mmBERT dobra 1, org+LLM | 0,9998 / 1,000 / 0,9972 | 0,9995 / 1,000 / 0,9972 | 7 |
| BERTimbau dobra 0, org+LLM | 1,000 / 1,000 / 0,9986 | 1,000 / 1,000 / 0,9972 | 4 |
| BERTimbau dobra 1, org+LLM | 1,000 / 1,000 / 1,000 | 0,9997 / 1,000 / 0,9986 | 4 |
| mmBERT dobra 0, só org | 1,000 / 1,000 / 0,9979 | 0,9983 / 1,000 / 0,9965 | 15 |
| mmBERT dobra 1, só org | 0,9987 / 0,9966 / 0,9924 | 0,9953 / 0,9952 / 0,9924 | 46 |

A linha do Gama é o v1; o Gama v1.2 (`vinimlo/gama@ad06ffd`, ruído dirigido à
palavra-chave) zera os dois erros de borda: 0 em 4.447.

Com resolução, pela métrica oficial, numa amostra estratificada por nível (200 N1 + 200
N2, mais o dev): 100% dos pares casados certos
([D-006](../decisoes/D-006_calibracao-decide-o-topo.md)). Amostras deste conjunto são
sempre estratificadas: o corte por ordem alfabética pega só `syn_n1_*`.

Referência pela métrica oficial (resolução incluída): régua 0,848 (N1 1,046; N2 0,750,
com F1 inventada 0,663 e incompleta 0,514).

## Leitura

1. A expansão por LLM é o que dá robustez: modelos só-org erram 3 a 20 vezes mais
   em redação nova. Confirma a [validação por dobras](2026-09-22_validacao-por-dobras.md) por outro ângulo.
2. mmBERT e BERTimbau empatam. Sem motivo para trocar; o contexto de 8 mil tokens do
   mmBERT (documento numa passada, 0,077 s/doc na L4) é vantagem prática.
3. Treinar com o banco completo ajuda: o Gama final supera todos os modelos de dobra.
4. A régua é o que o desafio pune: ajustada ao dev, cai a 0,848 em redação nova.
