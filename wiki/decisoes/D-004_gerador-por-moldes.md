# D-004: Goldenset por moldes da organização, não por redação livre de LLM

Data: 22/09/2026 · Status: aceita

## Contexto

O caminho óbvio era o LLM redigir o documento inteiro com a citação entre marcadores.
A medição do dev set mostrou que a organização gera por moldes (ver
[experimento](../experimentos/2026-09-22_gerador-sintetico.md)): cabeçalho, narrativa
por matéria, enchimento e carregadoras de bancos pequenos, quebra de linha determinística.

## Decisão

O gerador desmonta o dev set em bancos e remonta documentos citando fichas reais. O LLM
só expande os bancos (frases sem citação), nunca escreve a citação nem decide rótulo.

## Por quê

- Rótulo exato por construção: offsets calculados antes da quebra, que só troca espaço
  por `\n`; `real` aponta para a ficha sorteada.
- A distribuição-alvo é a da organização, e o cego sai do mesmo gerador.
- Offsets escritos por LLM exigiriam alinhamento e descarte; aqui não há o que alinhar.

## Consequências

- O dev set deixa de ser avaliação neutra para escolher o GERADOR (as frases dele estão
  nos bancos). Escolhas de gerador e de mistura se fazem na validação por dobras
  (`treino/dobras.py`): bancos sem a dobra avaliada.
- A fração da expansão por LLM (`montar.FRACAO_LLM`) é hiperparâmetro, decidido na
  [validação por dobras](../experimentos/2026-09-22_validacao-por-dobras.md).
