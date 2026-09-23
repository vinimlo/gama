# 2026-09-22 · Texto real: onde o Gama quebra fora do estilo da organização

Pergunta: o Gama, perfeito no estilo da organização, acha citações em ementas reais?

## Desenho (extração → organização → classificação → votação)

- Extração/organização: 2.454 ementas de `celsowm/jurisprudencias_br@9738075`
  (CC-BY-4.0; STF, STJ, TJRJ), amostra por reservatório por tribunal, proveniência por
  registro. LeNER-Br fora (licença "unknown").
- Classificação: Gama e régua extraem (GPU, HF Jobs). Em 1.919 ementas eles divergem.
  Amostra de 300 divergentes + 60 concordantes anotada por DeepSeek-V4-Pro e Kimi-K3
  reescrevendo com marcadores; anotação aceita só se o texto sem marcador for idêntico
  ao original (55 de 360 descartadas). Rótulo prata = os dois concordam (936 spans).
- Votação: divergências entre os LLMs resolvidas em camadas. Primeiro o critério escrito,
  depois um terceiro voto (GLM-5.3, maioria 2 de 3), os critérios de convenção (diploma
  antes do artigo, qualificador de Tema, lista de súmulas) e, por fim, 3 convenções
  votadas por padrão (lista com classe compartilhada = uma citação por número;
  qualificador entre classe e número entra; nome por extenso + sigla começa no
  extenso). Ambíguos: 81 → 51 → 23 → 1.
  Resultado: `corpus/reais/ouro_real.jsonl`, 305 ementas, 1.085 citações adjudicadas.

Gotcha: o parser de JSON reaproveitado da expansão só aceitava as chaves dela e descartou
as 759 respostas (todas válidas) como "desalinhadas". O cache tornou o reprocesso grátis.

## Resultado (IoU ≥ 0,5 contra a prata)

Medido com o Gama v1 (`vinimlo/gama@2aba5d1`). O v1.2 foi medido de novo no
[benchmark contra extratores sem treino](2026-09-23_bench-extratores-crus.md), contra o
ouro adjudicado: o recall de precedente sobe para 0,76, mas a precisão cai para 0,35,
porque fora do molde ele marca fragmentos soltos ("Rel", "2011", "DJe 19/08/2019").

| Extrator | Precedente P / R | Dispositivo de lei P / R |
|---|---|---|
| Gama | 0,47 / 0,49 | 0,64 / 0,92 |
| Régua | 0,66 / 0,82 | 0,63 / 0,61 |
| União (Gama + régua onde o Gama não marcou) | 0,53 / 0,76 | 0,63 / 0,93 |

## Onde o Gama erra

- Não acha (170): citação em lista depois de "DJe" ("…; STJ, REsp 1.999.624/PR, Rel.
  Min."), entre parênteses ("(ARE 639.228, Rel. Min. …)"), em caixa-alta da ementa ("RESP
  1.749.206/MG", "SÚMULAS NS. 282 E 356"), Tema sem "da repercussão geral".
- Borda quebrada (52): cadeia de sufixos do STF ("RE nº 740.591-AgR-terceiro-ED-ED/RO").
- Falso positivo (139): "DJe de 4.6.2012", "Portaria n. 191/2006", "EC 20/1998",
  artigo sem diploma.

## Leitura

O Gama especializou no estilo da organização, que é o alvo declarado do cego ("mesmo
formato"); por isso isto não contradiz a [validação por dobras](2026-09-22_validacao-por-dobras.md) nem o estresse. Mas é a medida de risco caso o
cego traga redação fora do molde. A união é a proteção barata: no estilo da
organização empata com o neural (dobras, estresse N1); em texto real sobe o recall de
precedente de 0,49 para 0,76. No estresse com ruído (N2), porém, a união custa 61 falsos
positivos; por isso ela é opção e não padrão ([D-007](../decisoes/D-007_sem-ensemble.md)).
