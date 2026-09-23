# D-006: Confiança = probabilidade de acerto medida por balde

Data: 22/09/2026 · Status: aceita

## Contexto

O campo `confianca` entra no score pelo bônus de calibração,
`score = s · (1 + 0,10·(1 − Brier))`. Com acurácia `p` e confiança `c` em todos os pares
casados, o Brier esperado é `p(1−c)² + (1−p)c²`, mínimo em `c = p`.

| Estratégia | Brier | Bônus `0,1·(1−brier)` |
|---|---|---|
| saturar, `c = 1` | `1 − p` | `0,1·p` |
| calibrar uniforme, `c = p` | `p(1−p)` | `0,1·(1 − p + p²)` |
| discriminação perfeita (1 nos acertos, 0 nos erros) | `0` | `0,1` |

Em termos absolutos a calibração vale pouco: a distância entre saturar e calibrar é
`0,1·(1−p)²`, 0,00025 com p = 0,95. Quem decide o placar é o F1.

Só que o extrator acerta 192/192 citações em documentos da organização que nunca viu
([validação por dobras](../experimentos/2026-09-22_validacao-por-dobras.md)). Se o cego sai
do mesmo gerador, várias equipes podem chegar a F1 = 1,0, e aí a ordem do ranking sai do
bônus: 1,0984 com confiança fixa em 0,90–0,95 contra 1,1000 com confiança perto de 1 em
cima de acerto.

## Decisão

A confiança de cada citação é a probabilidade de acerto medida do seu balde
(via × classe × faixa de confiança do modelo), estimada em dados rotulados que o modelo
não treinou, com prior Beta(1,1) e teto 0,995.

- Nunca 1,0: confiança alta em cima de erro é o pior caso do Brier.
- Balde com erro medido (empate de candidatos, span de baixa confiança do modelo, forma
  rara) mantém confiança baixa. Balde sem dado fica no prior conservador.
- `avaliacao/calibrar.py` mede; a tabela fica em `src/gama/calibracao.json` e nunca é
  escrita à mão.

## Resultado

Amostra estratificada por nível (dev + 200 documentos N1 + 200 N2 do
[estresse difícil](../experimentos/2026-09-22_estresse-dificil.md)), Gama v1.2:
3.165 pares casados, 100% de acerto em todo balde medido, inclusive sob o ruído de N2.
Tabela: 0,991–0,995. Único balde abaixo: empate de candidatos (`processo|real_ambiguo`,
1 de 2), que fica em 0,5.

Efeito no dev: bônus 0,0985 → 0,09999; score 1,09848 → 1,10000.
