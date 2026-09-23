# 2026-09-22 · Gerador sintético no molde da organização

Pergunta: como produzir milhares de documentos rotulados que se pareçam com os da
organização (e, portanto, com o conjunto cego) sem depender de LLM para rotular?

Resposta curta: a organização gera os documentos por moldes. Desmontamos os 26
documentos do dev set em bancos de peças e remontamos documentos novos citando fichas
reais do acervo. O rótulo sai exato por construção.

## Evidência de que o gerador deles é por moldes

| Medida | Valor |
|---|---|
| Frases sem citação que aparecem em 2+ documentos | 62% |
| Frases carregadoras (com citação) distintas para 192 citações | 83 (59 depois de normalizar o artigo) |
| Datas distintas por documento | 1 em 26/26 (preâmbulo, narrativa e fecho usam a mesma) |
| Anos das datas | 2019–2025 |
| Parágrafos de prosa do nível 1 reproduzidos byte a byte pela quebra "termina a linha na palavra que chega a 86–91 caracteres" | 60/69 |
| Tribunal da citação real = tribunal da matéria do documento | 79/82 (STF quase nunca) |
| Referências vagas vindas do STF | ~60% |

Estimativa Chao1 da massa não vista no dev set: ~19% das carregadoras, ~33% do enchimento.

## Achados que viraram código

- Negativos difíceis: as ~33 "vagas sem fonte" que a organização tirou do gabarito são a
  MESMA carregadora preenchida com referência genérica ("veja-se o verbete sumular
  aplicável à espécie"). O gerador as reproduz sem rótulo, e o modelo precisa aprender a
  não marcar. Regex não aprende isso sozinha.
- Ruído em dois canais (medido no nível 2 por 1.000 palavras, fora das citações):
  dígito em palavra 0,16 (nosso injetor de canal único dava 15,7), m→rn ~1,4 (dava 21),
  "dc" 0,47 (dava 2,5). O ruído forte mora dentro das citações. `ruido.FORA` calibra.
- Inventadas: artigo de lei além do último artigo do diploma (art. 261 da CF, art.
  158 do CDC); Rcl com par de dígitos repetido na frente (33.235, 88.178); súmula e tema
  acima da numeração real. Número inventado é sempre conferido: zero candidatos no acervo.
- Fichas ambíguas ficam de fora: 73 das 77 chaves compartilhadas têm a mesma cadeia de
  classe. Não há desempate possível, e a organização não as cita.

## O gerador como teste do resolver

Cada citação renderizada de uma ficha tem rótulo conhecido; onde o resolver discorda, é bug
do resolver. Achados e corrigidos (todos com teste em `tests/test_resolver_formas.py`):

1. "Terceiro AgR na Rcl" lido como tema (`te[mr]` casava "Terceiro").
2. Súmula de outro tribunal resolvendo como real: "Súmula 211 do TSE" virava a 211 do
   STJ. Risco de τ latente no cego.
3. "LC nº 64/1990" sem resolver.
4. OCR no nome do diploma ("Códig0 de Pr0cesso Civil", "Constltuição"), resolvido por
   esqueleto: normalização muitos-para-um aplicada ao texto e ao apelido.
5. Número de lei com letra de OCR ("131O5/2015", "807B/1990"), lido do texto original
   porque a caixa carrega informação (B=8, b=6).
6. "artlgo", "ãrt." e 7. "VincuIante".

Discordância residual: 7/2.948 citações (0,24%), todas cadeias raras sob ruído.

## Validação adversarial

| Medida | AUC | Leitura |
|---|---|---|
| Bancos do dev inteiro × dev | 0,245 | vazamento, não qualidade: o doc deixado de fora tem frases que só ele tinha, rotuladas como "sintético" |
| Honesta (bancos de metade × outra metade), só bancos da org | 0,825 | pista: carregadoras e enchimento nunca vistos |
| Honesta, org + expansão por LLM | 0,894 | o LLM traz vocabulário próprio ("recursal", "deduzida", "pacificada") e não cobre as frases que faltam |

Conclusão provisória: a expansão por LLM afasta o estilo. Isso não prova que piora o
modelo, porque AUC mede distância de estilo, não utilidade. A decisão sai do experimento
da [validação por dobras](2026-09-22_validacao-por-dobras.md): treino em sintético de uma
metade, avaliação com a métrica oficial na outra metade, variantes "só org" × "org + LLM".

## Expansão por LLM (DeepSeek-V4-Pro, Kimi-K3, GLM-5.3 via Ollama)

400 carregadoras, 479 frases de enchimento, 152 referências genéricas, 103 blocos
narrativos, 32 gêneros novos, tudo validado (régua sem citação, sem dígito fora de slot,
sem duplicata do banco da org). Gotchas: o GLM-5.3 ignora `think=false` e escreve o
raciocínio dentro da resposta; em modo JSON a 0,9 entrou em laço (28 mil tokens). Kimi
embrulha o JSON em cerca ```.

## Ida e volta da tokenização

`rotular → decodificar` devolve os spans idênticos em 192/192 (dev) e 2.997/2.997
(sintético) para os tokenizadores do mmBERT e do BERTimbau. Aberto.

## Smoke test do caminho neural

mmBERT-small, 0,3 época, 57 documentos, CPU: 0,777 no dev pela métrica oficial,
passando pelo pipeline inteiro (extrator → resolver → classificação → JSON → harness).
Só prova que o caminho funciona; o número de verdade vem do treino em GPU.
