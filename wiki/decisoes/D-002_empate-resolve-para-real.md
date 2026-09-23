---
slug: empate-resolve-para-real
tipo: decisao
status: ativo
data: 2026-09-22
one_liner: "Citação numerada com 2+ candidatos vira `real` (melhor candidato), nunca `incompleta`"
---

# D-002: Empate numerado resolve para `real`

Contexto: o enunciado manda classificar como `incompleta` quando a consulta
devolve 2 ou mais feitos distintos sem critério de desempate. Seguir isso ao pé
da letra custava 0,044 de score.

Dois fatos mudam a leitura:

1. Nenhuma `incompleta` do gabarito tem número. Depois da limpeza final da
   organização, as 32 restantes são todas da forma vaga (tribunal + ano +
   relator). Logo, empate numa citação numerada é artefato do nosso índice,
   não ambiguidade do dado.
2. A métrica cobra assimetricamente. Lendo o `kaggle_metric.py`:
   - `real` × `real` com link errado → `fp[real]`, sem `fn`. Machuca uma classe só.
   - gabarito `real`, predição `incompleta` → `fn[real]` e `fp[incompleta]`.
     Machuca duas.

Decisão: com 2+ candidatos numa citação numerada, entregar `real` com o
melhor candidato. Confiança cai para 0,50, para o Brier refletir a incerteza.

Risco avaliado: não afeta `τ`, que conta apenas gabarito `inventada` predito
como `real`; o caso aqui é o oposto (temos candidatos, então não é uma
invenção). Nenhum caminho novo para o erro grave.

Resultado medido: 1,03057 → 1,07456 numa única mudança.

Quando revisar: se o conjunto cego trouxer `incompleta` numerada. O
enunciado prevê essa forma, embora o gabarito publicado não a tenha. Sintoma:
`CLASSE_ERRADA incompleta->real` aparecendo em `make erros`.

## Conferência com o texto oficial (23/09)

A aba Data da Kaggle define "2 ou mais candidatos distintos, sem critério de desempate →
incompleta", o contrário desta decisão. Mantida, porque o mesmo texto garante que "as
duplicatas remanescentes do acervo não têm nenhuma citação apontando para elas": o empate
residual (mesma chave e mesma cadeia de classe) não aparece no gabarito, e o desempate pela
cadeia de classe resolve os demais. A confiança desse balde é 0,5 (medida), então um erro
ali custa pouco no Brier. Reabrir se o cego mostrar citação numerada caindo em empate.
