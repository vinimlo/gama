---
slug: indice-de-cabecalho
tipo: decisao
status: ativo
data: 2026-09-22
one_liner: "Índice canônico offline de cabeçalho + autorreferência, em vez de busca FTS por citação"
---

# D-001: Índice de cabeçalho, não FTS por citação

Contexto: resolver uma citação exige achar o registro que *é* aquele feito.
O FTS5 do acervo devolve qualquer documento que mencione o número, e
acórdãos citam uns aos outros o tempo todo: buscar `"1.276.977"` devolve seis
documentos do STF e nenhum é o RE 1.276.977. Todos apenas o citam. É a
armadilha de precisão que o próprio enunciado destaca.

Decisão: construir um índice uma vez, offline, mapeando
`chave numérica -> [doc_id]`, alimentado por duas fontes:

1. todos os números do cabeçalho curto (primeiros 300 chars), o que cobre STF, STJ, TSE, STM;
2. a autorreferência explícita: `nº TST-ED-E-ED-RR-<num>` até 2.000 chars, e a
   âncora semântica `"autos de ... nº <sigla>-<num>"` até 8.000 chars.

Por que a segunda fonte existe: os acórdãos do TST abrem com um preâmbulo
longo (`A C Ó R D Ã O SbDI-1 GMJRP/…`) e só citam o próprio número por volta do
caractere 1.100 (num caso, 4.309). Pior: citam a `Lei 13.015/2014` no preâmbulo,
o que já produzia uma chave e impedia o fallback de rodar. Sem isso, 10
processos trabalhistas ficavam fora do índice e toda citação de RR/ARR caía
como `inventada`.

Por que não alargar a janela: a partir da ementa o acórdão passa a citar
outros feitos. Janela grande indexaria o que ele apenas cita e arruinaria a
precisão, que é exatamente o problema que a decisão existe para resolver. A frase
`"autos de"` só aparece quando o feito fala de si mesmo, então ela permite ir
longe no texto sem pagar esse preço.

Consequências: lookup O(1) em vez de varredura FTS por citação; ~10 ms por
documento contra um teto de 60 s. Custo: 270 chaves ficam ambíguas (>1 acórdão) e são
tratadas pela [D-002](D-002_empate-resolve-para-real.md), que as contorna sem resolver.
