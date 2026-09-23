---
slug: baseline-deterministico
tipo: experimento
status: fechado
data: 2026-09-22
score: 1.09830
related: [indice-de-cabecalho, empate-resolve-para-real]
one_liner: "Pipeline 100% determinístico: 1,09830 no dev set, 0 erros em 192 citações, 10 ms/doc"
---

# Baseline determinístico (a régua)

Primeira rodada medida. Extração por regex + normalização de OCR + índice
canônico offline + árvore de decisão por cardinalidade. Sem modelo.

Hoje é a régua: a referência que o extrator neural precisa superar e o fallback do
pipeline quando os pesos não estão montados (`--extrator regua`).

## Resultado

| | F1 real | F1 inventada | F1 incompleta | macro-F1 | τ | bônus | score |
|---|---|---|---|---|---|---|---|
| Nível 1 (1×) | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 0,0000 | 0,0985 | 1,0985 |
| Nível 2 (2×) | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 0,0000 | 0,0982 | 1,0982 |
| Final | | | | | | | 1,09830 |

Recall de span: 192/192. Falsos positivos: 0. Custo: ~10 ms/documento.

## Trajetória (o que cada correção valeu)

| # | Mudança | Recall span | Score |
|---|---|---|---|
| 0 | primeira versão do extrator | 67,2% | |
| 1 | vagas antes de números; ano não é processo; CNJ relaxado; artigo com inciso | 89,1% | |
| 2 | *(regressão)* reescrita perdeu `nº` do vocabulário | 80,2% | |
| 3 | `nº`/`n°` normalizados; cauda do artigo com ponto no número da lei | 95,8% | |
| 4 | `_fronteira_sentenca` distingue ponto-de-abreviatura de ponto-final | 100% | 1,02742 |
| 5 | Constituição indexada; `§ 1º-A`; súmula com OCR; classe por extenso | 100% | 1,03057 |
| 6 | empate numerado resolve para `real` | 100% | 1,07456 |
| 7 | autorreferência do TST sempre buscada, com âncora em "autos de" | 100% | 1,09830 |

## O que ensinou

O erro de borda concentra-se num predicado. O salto de 80% → 100% de recall
veio de uma função de 12 linhas. O corte ingênuo de sentença em `". "` disparava
dentro de `AgRg no Rec. Esp. n. 1.522.200` e decepava o prefixo: o span era
achado mas com IoU < 0,5, contando ao mesmo tempo como recall perdido e falso
positivo, uma punição dupla. Distinguir abreviatura de fim de frase resolveu 8 citações.

A mesma armadilha três vezes. Aplicar a tabela de OCR letra→dígito ao trecho
inteiro converte o prefixo (`APL`, `AgInt`, `TST-ED-E-ED-RR`) em dígitos e gera
chave fantasma. Falha silenciosa: aparece só como "zero candidatos". Virou
invariante do código (`span.digitos`).

O TST escondeu 10 citações atrás de um preâmbulo. Os acórdãos trabalhistas
abrem com `A C Ó R D Ã O SbDI-1 GMJRP/…` e só citam o próprio número por volta do
caractere 1100 (às vezes 4300). Pior: citam a `Lei 13.015/2014` no preâmbulo, o
que gerava uma chave e impedia o fallback de autorreferência de rodar. Alargar a
janela cegamente pegaria números apenas citados; a âncora semântica
`"autos de … nº TST-…"` vai longe no texto sem perder precisão.

Raciocinar pela métrica antes de escolher a regra. Ver
[D-002](../decisoes/D-002_empate-resolve-para-real.md): sozinha, essa decisão valeu +0,044.

## Limitações (honestas)

- Este número é ajustado ao dev set. O extrator foi iterado contra estas 192
  citações. Não é estimativa de desempenho no conjunto cego.
- Sem validação cruzada. Com 26 documentos, qualquer split deixa amostra
  pequena demais; a avaliação real virá do conjunto cego.
- Robustez a formas de superfície não vistas: não medida aqui. Medida depois no
  [estresse difícil](2026-09-22_estresse-dificil.md): a régua cai para 0,848.
- 270 chaves de processo ambíguas no índice (>1 acórdão). Contornadas pela
  política de empate, não resolvidas.

## Submetido ao Kaggle (22/09/2026)

`submission.csv` enviado. O Kaggle pontuou 1.09830, idêntico ao harness
local até o último dígito. Isso fecha a validação do contrato inteiro:
offsets em codepoints, encoding, conversor oficial e métrica. A partir daqui o
harness local é fonte confiável e não é preciso gastar submissão para medir.

A faixa do leaderboard logo acima (1,09879–1,09999) também tinha zero erros e diferia
apenas pela confiança declarada, o que levou à
[D-006](../decisoes/D-006_calibracao-decide-o-topo.md).
