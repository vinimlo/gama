# Ementas reais

Todo o texto real do projeto vem de um lugar só: o
[`celsowm/jurisprudencias_br`](https://huggingface.co/datasets/celsowm/jurisprudencias_br),
publicado no Hugging Face pelo Celso F. São cerca de 781 mil decisões do STF, do STJ e do
TJRJ, coletadas pelo Juriscraper, ferramenta do
próprio autor, sob CC-BY-4.0. Usamos a revisão `9738075`.

## Por que precisávamos dele

O dev set tem 26 documentos e sai de um gerador por moldes. O nosso goldenset remonta esses
mesmos moldes. Um modelo que acerta tudo nos dois ainda não provou nada sobre texto escrito
por gente, e o conjunto cego pode trazer redação diferente.

Faltava texto jurídico de verdade, público e com licença que deixasse publicar o que
fizéssemos com ele. O LeNER-Br ficou de fora por isso: a licença dele consta como
desconhecida. O `jurisprudencias_br` tinha as três coisas, e ainda gravava de onde veio
cada registro.

## Como entrou no projeto

Tudo parte de uma amostra de 2.454 ementas, sorteadas por reservatório dentro de cada
tribunal (`reais/ingerir.py`). Cada registro guarda fonte, licença, revisão do dataset,
tribunal e classe.

| Conjunto | Ementas | Como foi montado | Onde entrou |
|---|---|---|---|
| Ouro do benchmark | 305, com 1.085 citações | 300 em que o Gama e a régua divergiam mais 60 em que concordavam, anotadas e adjudicadas; 55 descartadas | [texto real](../experimentos/2026-09-22_texto-real.md), [benchmark](../experimentos/2026-09-23_bench-extratores-crus.md), escolha da guarda ([D-008](../decisoes/D-008_guarda-do-extrator.md)) |
| Teste intocado | 172, com 682 citações | 200 sorteadas ao acaso fora das 360 anteriores e sem texto repetido (`reais/sortear_teste.py`); 172 passaram no alinhamento | confirmação da guarda e comparação v1.2 × v1.3 ([D-009](../decisoes/D-009_gama-v1-3-destilado.md)) |
| Destilação | 1.818, sem rótulo | o que sobrou da amostra, sem texto repetido de nenhum dos conjuntos acima | treino do v1.3, onde só vale a imitação do v1.2 ([destilação](../experimentos/2026-09-23_destilacao.md)) |

A anotação pede a duas LLMs de pesos abertos (DeepSeek-V4-Pro e Kimi-K3) que reescrevam a
ementa com marcadores em volta de cada citação. A resposta só vale se, tirando os
marcadores, o texto for idêntico ao original. Onde as duas concordam, o span entra. Onde
discordam, decide primeiro o critério escrito das [convenções de borda](convencoes-de-borda.md),
depois um terceiro voto (GLM-5.3, maioria de dois em três) e, por último, três convenções
votadas por padrão. Os casos ambíguos caíram de 81 para 1.

## O que o texto real mudou

Foi nele que o Gama mostrou o limite. Especializado no estilo da organização, o modelo
sozinho marca fragmentos soltos em ementa ("Rel", "2011", "DJe") e lê "Rel. Min. Fulano,
julgado em ..." como se fosse a referência vaga do molde. A pista para consertar também
veio daqui: no v1.2, esses fragmentos ficavam com confiança mediana entre 0,85 e 0,94,
enquanto nenhum dos 34.171 acertos no estilo da organização ficava abaixo de 0,98. A guarda nasceu dessa
diferença e leva o v1.3 de 0,620 a 0,818 nas 305 ementas.

Como a guarda foi escolhida olhando para as 305, elas deixaram de ser um teste limpo dela.
As 172 existem por isso. Nelas o ganho foi +0,205 (IC95 +0,154 a +0,246), o mesmo das 305.

A destilação usou as 1.818 restantes sem rótulo nenhum: o aluno aprende a reproduzir as
probabilidades do professor também em texto real, não só no sintético. No teste intocado
o v1.3 marca 0,839, contra 0,820 do v1.2.

## O que ele nunca foi

Base de resolução. O acervo do desafio é fechado, com 1.014 registros. Se um processo real
tirado daqui coincidisse com um número que a organização inventou, a citação sairia `real`,
e a métrica pune exatamente esse erro (τ). As ementas ensinam e medem onde a citação está.
Quem diz se ela existe é o acervo.

## Limites

- O gabarito é de máquina, adjudicado por critério escrito. Não é anotação humana
  independente. A leitura de 120 erros sorteados achou citações que o ouro esqueceu e lei
  sem artigo ora dentro, ora fora ([otimização medida](../experimentos/2026-09-23_otimizacao-medida.md)).
- A exigência de texto idêntico descartou 52 de 300 ementas divergentes e só 3 de 60
  concordantes. As difíceis ficaram sub-representadas, numa direção que só uma anotação
  sem descarte mediria.
- São ementas do STF, do STJ e do TJRJ. O que medimos vale para ementas dessas fontes, não
  para petição, parecer ou sentença.
- As 172 também já informaram uma escolha (v1.3 no lugar do v1.2). Para uma variante
  futura, deixam de ser um teste intocado.

## Onde estão

No dataset [`vinimlo/gama-goldenset`](https://huggingface.co/datasets/vinimlo/gama-goldenset):
`reais/amostra.jsonl` (as 2.454, com a proveniência), `reais/ouro_real.jsonl` (o gabarito
das 305, com a prata e os ambíguos ao lado), `reais/novo/` (o teste de 172),
`destilacao/reais_ids.json` (os ids da destilação) e `bench/` (entradas e saídas do
benchmark). Essas ementas seguem a CC-BY-4.0 da fonte; o resto do dataset é MIT.
