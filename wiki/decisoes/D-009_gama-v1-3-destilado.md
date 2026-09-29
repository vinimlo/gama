# D-009: Gama v1.3, o v1.2 destilado em 12 camadas

Data: 23/09/2026 · Status: aceita

## Critério

O extrator da submissão só troca se a solução inteira entregar o mesmo no estilo da
organização (o JSON final igual documento a documento no dev, no estresse difícil e no
`final_v1`) e passar nas mesmas checagens de entrega que o v1.2 passou. Sem isso, nenhum
ganho fora do molde justifica a troca.

## Decisão

A solução passa a usar o Gama v1.3 (`vinimlo/gama@5f924ca`, tag `v1.3`): as 12 primeiras
das 22 camadas do v1.2, com embeddings e cabeça herdados, treinadas para imitar as
probabilidades do v1.2 (`treino/destilar.py`). São os mesmos pesos medidos como aluno
base 12 camadas (revisão `f49e37b` do repositório de experimento, apagado do Hub depois da
escolha). A guarda, o resolver e a calibração não mudam.

A poda das camadas de cima segue [Sajjad et al., 2004.03844](https://arxiv.org/abs/2004.03844):
as camadas de baixo pesam mais, e tirar as de cima foi a melhor estratégia de poda que eles
mediram. O corte contíguo mantém a alternância de atenção global e local do ModernBERT sem
escolher camada por camada.

## Medida

Números completos na [destilação](../experimentos/2026-09-23_destilacao.md).

| | v1.2 | v1.3 |
|---|---|---|
| Camadas, parâmetros | 22, 307,5M | 12, 257,4M |
| Dev, estresse, final_v1, métrica oficial | 1,10000, 1,09999, 1,09999 | iguais |
| Documentos com JSON final diferente do v1.2 (4.626) | | 0 |
| Texto real com a guarda, 305 ementas | 0,8076 | 0,8180 |
| Texto real com a guarda, 172 ementas do teste novo | 0,8204 | 0,8392 (+0,019; IC95 +0,003 a +0,033) |
| L4, s por documento no estresse | 0,071 | 0,046 |
| CPU, s por documento no dev | 1,62 | 0,93 |
| Pesos | 1,23 GB | 1,03 GB |

Checagens de entrega: sondas de ruído 6 de 6, os mesmos casos; GPU igual a CPU em 100
documentos do estresse (spans idênticos, nenhuma confiança cruza 0,95 ou 0,98); imagem sem
rede no dev com 1,10000 e JSON idêntico ao do v1.2.

Alternativas medidas e descartadas: o aluno de 13 camadas espalhadas muda uma borda em 4.626
documentos; o de 9 camadas perde 0,00033 no estresse; os alunos mmBERT-small mudam bordas
com ruído pesado de OCR. O controle sem professor perde 0,23 no texto real: é a destilação
que segura a qualidade fora do molde.

## Riscos que ficam

- Dos 4.626 documentos do estilo, só os 26 do dev são da organização; o resto é sintético
  nosso. JSON idêntico ali não garante o mesmo no conjunto cego.
- O ganho no texto real é pequeno e só pesa se o cego trouxer texto fora do molde.
- A nota medida não muda: no estilo da organização a saída é a mesma do v1.2.

## Revisão em 29/09

Os [controles](../experimentos/2026-09-29_controles.md) retreinaram as duas receitas com as
sementes 7 e 21. Com a guarda, o v1.2 vai de 0,760 a 0,830 nas 305 ementas e o v1.3,
destilado do mesmo professor, fica entre 0,812 e 0,818. Os +0,019 do v1.3 nas 172 não passam
da variação entre sementes do v1.2, então não sustentam que o aluno acerta mais em texto
real. A decisão se mantém pelo critério dela, que nunca dependeu desse ganho: o mesmo JSON no
estilo da organização, as mesmas checagens de entrega e o tempo. No dev as seis sementes dão
o mesmo JSON.

Voltar ao v1.2 só por falha de reprodução nas checagens de entrega, nunca por comparar
saídas no conjunto cego (regra 4.b do Kaggle).
