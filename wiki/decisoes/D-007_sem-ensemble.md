# D-007: Sem ensemble; a união com a régua fica como opção

Data: 22/09/2026 · Status: aceita

## Critério

Um ensemble só entra se ganhar do melhor candidato isolado nos dois conjuntos: no dev e no
estresse difícil.

## Medida

| Conjunto | Gama sozinho |
|---|---|
| Dev (métrica oficial, calibrado) | 1,10000, o teto prático |
| Estresse difícil, métrica oficial com resolução (dev + 200 N1 + 200 N2) | 100% dos 3.165 pares casados |
| Estresse difícil, extração (600 docs, N1 + N2) | F1 IoU ≥ 0,5 = 1,000 nos três tipos; zero erros de borda em 4.447 spans |

A união com a régua empata com o neural nas dobras: a régua não acrescenta onde o modelo
acerta. mmBERT e BERTimbau empatam no estresse.

Não há ganho mensurável possível sobre um candidato no teto. O ensemble só traria custo:
dois modelos para reproduzir offline, mais tempo por documento, mais superfície de bug.

## E a união neural + régua?

Em texto real, fora do molde da organização, o Gama fica atrás da régua (F1 de extração
0,605 contra 0,663, no [benchmark](../experimentos/2026-09-23_bench-extratores-crus.md)), e
com o v1 a união (Gama + régua onde o Gama não marcou) recuperava boa parte do recall de
precedente ([texto real](../experimentos/2026-09-22_texto-real.md)). Mas no estresse difícil:

| IoU ≥ 0,5 | N1 | N2 (ruído) |
|---|---|---|
| Gama (neural) | 1,000, zero erros | 1,000, zero erros |
| União | 1,000 | 0,979, 61 falsos positivos da régua |

N2 pesa o dobro no score e o cego é anunciado como "mesmo formato". O ganho da união é
num risco hipotético (texto fora do molde); a perda é medida (ruído no molde).

## Decisão

Padrão: Gama sozinho, com a régua como fallback se os pesos não estiverem montados.
A união fica disponível por `--extrator uniao`, para ligar só se a inspeção sem gabarito
(`avaliacao/sem_gabarito.py`) mostrar texto fora do estilo da organização: citação em
lista depois de "DJe", entre parênteses com "Rel. Min.", ementa em caixa-alta.

Reabrir se o conjunto cego mostrar divergência sistemática que um segundo modelo
resolveria.

Atualização (23/09): a [D-008](D-008_guarda-do-extrator.md) resolveu o texto fora do molde
de outro jeito. A régua entra só onde o modelo hesita (confiança < 0,95), o que no estilo da
organização não muda nenhum documento; a união continua como opção, mas deixou de ser a
proteção recomendada. A submissão roda com a configuração congelada: nenhuma escolha de
modo é feita depois de ver os documentos do conjunto cego (as regras do Kaggle vedam usar
predição humana sobre dados de teste).
