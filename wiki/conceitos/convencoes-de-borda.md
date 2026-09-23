# Convenções de borda do gabarito

Derivadas das 192 citações do `goldenset_offsets.csv` em 22/09/2026, olhando o
span junto com o contexto imediato no `.txt`, porque o que ficou fora ensina tanto
quanto o que ficou dentro. Reproduzir com `docker compose run --rm gama python -m avaliacao.convencoes`.

Para que serve: o goldenset sintético treina o extrator neural. Borda
diferente da organização ensina o modelo a errar a borda, e erro de borda custa IoU
sem aparecer como erro de classe. O verificador `avaliacao.convencoes.violacoes_de_borda`
aplica estas regras ao gabarito (teste: zero violações) e a cada documento gerado.

## Regras universais (100% do gabarito)

| Regra | Evidência | Exemplo |
|---|---|---|
| Artigo ou preposição antes fica fora | 192/192 | `invocar o ⟦REsp nº 2020005/RJ⟧` · `reconheceu no ⟦julgado do STM…⟧` |
| Pontuação depois fica fora | 0/192 terminam em `.,;:` | `⟦Rcl 88.178/RS⟧, que trata` |
| Sem espaço nas bordas | 192/192 | |
| Quebra de linha dentro do span é normal | 47/192 | `⟦APL nº\n7000449-40.2023.7.00.0000/RS⟧` |

A quebra de linha interna é o ponto que mais pesa para o modelo: em referências
vagas ela aparece em 18 de 32. O tokenizador precisa preservar `\n` no
`offset_mapping`, e o teste de ida e volta da tokenização existe para pegar isso.

## Regras por forma

| Forma | n | Começa em | Termina em | Observações |
|---|---|---|---|---|
| Número clássico | 86 | classe processual (`AgInt`, `Rcl`, `Reclamação`, `REsp`, `EDcl`), nunca minúscula | UF, sempre (86/86): `/SP`, `- SP`, `(SP)` | `nº` presente em 68/86 |
| CNJ | 33 | classe (`APL`, `RSE`, `AgInt`, `ED`) ou `processo nº` | UF em só 11/33; senão o último grupo do número | o prefixo `processo nº` entra no span quando aparece |
| Súmula | 12 | `Súmula` / `Súm.` / `SÚMULA` / `5úmula` | tribunal incluído quando citado (9/12): `do STJ` | nunca usa `nº` |
| Tema | 1 | `Tema` | `da repercussão geral` incluído | amostra de 1: tratar como indício |
| Artigo de lei | 28 | sempre minúsculo: `art.` (17), `artigo` (6), `art` (5) | nome do diploma incluído: `da Constituição Federal`, `do CDC`, `da Lei Complementar nº 64/1990` | incisos entram (`, I,`, `, IX,`); nunca UF |
| Referência vaga | 32 | substantivo de abertura: `julgado` (9), `precedente` (6), `Reclamação` (5), `acórdão` (5), `Rcl` (4) | fim do nome do relator (`ZANIN`, `Magalhães`, `Rosa Weber`) | minúscula em 20/32; quebra de linha em 18/32 |

## Exemplos com contexto

```
...Invoca-se, ainda, o ⟦julgado do STF proferido em 2024 pela relatoria de Dias Toffoli⟧, no ponto
...Não se pode ignorar a ⟦Rcl 88.178/RS⟧, que trata
...Invoca-se, ainda,\no ⟦AgInt 7557430-50.2018.7.00.0000/DF⟧, no ponto
...por oportuno, a ⟦APL nº\n7000449-40.2023.7.00.0000/RS⟧, cuja
```

## O que isto implica para o gerador

1. O prompt pede a citação sem o artigo dentro dos marcadores: `o ⟦C1⟧…⟦/C1⟧`.
2. A pontuação que segue a citação fica fora do marcador.
3. Número clássico sempre com UF; CNJ com UF só às vezes (~1/3).
4. Súmula com tribunal na maioria; artigo com o nome do diploma por extenso ou sigla.
5. O injetor de ruído insere quebra de linha dentro do span com frequência
   próxima da observada (~25% das citações, ~55% das vagas).
6. Todo documento gerado passa por `violacoes_de_borda`; qualquer violação descarta
   o documento.

## Texto real: convenções votadas

Fora do estilo da organização aparecem padrões que o gabarito não cobre. Decididos por
padrão (não item a item), para medir robustez em ementas reais:

| Padrão | Exemplo | Convenção |
|---|---|---|
| Lista com classe compartilhada | "Rcls 36.958 e 40.652"; "CPC, artigos 219, 224 e 1.003" | uma citação por número (a primeira leva a classe) |
| Qualificador entre classe e número | "Recurso Especial repetitivo 1.495.146/MG" | o qualificador entra |
| Nome por extenso + sigla | "Arguição de Descumprimento de Preceito Fundamental - ADPF nº 130/DF" | começa no extenso |
| Diploma antes do artigo | "RISTF, art. 131, § 2º"; "CPC, art. 369" | o diploma entra (critério escrito) |
| Tema com qualificador | "Tema 660 da sistemática da RG" | o qualificador entra (critério escrito) |
