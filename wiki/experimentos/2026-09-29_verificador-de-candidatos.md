# 2026-09-29 · Verificador de candidatos: reservas, Laya e juízes treinados

Pergunta: a guarda de produção ([D-008](../decisoes/D-008_guarda-do-extrator.md)) tira o span
em que o Gama hesita e põe no lugar o da régua, sem conferir se a régua acertou. Um reserva
melhor que a régua renderia mais? E um juiz que olhe cada candidato, o do Gama e o da régua, e
decida quem fica?

A submissão está congelada. Nada daqui entra nela.

## Desenho

- Os candidatos são a união dos spans do Gama v1.3 cru e da régua nos três conjuntos de
  avaliação: 13.664 no total (`vinimlo/gama-goldenset@de07c93`). Cada um carrega um estado,
  os 300 caracteres de cada lado com o candidato entre `[[` e `]]`. Um verificador dá um
  score a cada candidato, e uma política decide quem entra.
- As políticas rodaram pelo CLI de `bench/controles/verificador/`, sem mudança:
  - A: os spans fortes do Gama (confiança ≥ 0,95) ficam. O verificador decide só onde a
    guarda age, entre o span fraco e o da régua que o cruza.
  - B: a A, mais os spans da régua que não cruzam nenhum span do Gama.
  - C: todo candidato é julgado, inclusive os fortes. Entre dois sobrepostos, vence o de maior
    score.
- O limiar τ do score sai das 305 ementas (grade de 0,05 a 0,95, maior F1 exato, empate pelo
  maior τ) e é só medido nas 172. F1 de extração com mesmo tipo, IoU ≥ 0,5, casamento 1 para 1
  e VAGA contada como JURIS, pelo harness `bench/controles/avaliar.py`. IC95 por bootstrap
  pareado por ementa contra a produção (o v1.3 com a guarda: 0,8180 nas 305, 0,8392 nas
  172), 2.000 reamostras, semente 0.
- No estilo da organização contamos os documentos cujo resultado muda em relação à produção,
  no estresse difícil (600 documentos, com a nota oficial) e no dev (26 documentos, só local,
  em CPU, no container; nada do dev saiu da máquina). Critério de uso: nenhum documento mudado.
- Um cuidado na leitura desse critério. A política A só age onde a guarda age, e no estilo da
  organização a guarda não age: nem o estresse nem o dev têm span do Gama abaixo de 0,95. O
  zero da A ali vem da construção. No dev a B também não tem candidato elegível. Quem testa a
  B é o estresse, que tem 61 elegíveis; o dev só testa a C.
- Treinos e inferência no HF Jobs, quase tudo em L4 (o treino do Bosun foi numa A100). Os
  dados de treino ficaram num dataset privado e temporário
  (`vinimlo/gama-exp-verificador-dados@24eede0`), e os modelos treinados em repositórios
  privados de experimento (`vinimlo/gama-exp-verif-<candidato>`). Os dois serão apagados depois
  da revisão. Os scores por candidato ficam no `vinimlo/gama-goldenset`, em
  `bench/saida/controles/verificador_treinado/`: encoder do Gama na revisão `8e3db8b`, Laya
  treinado em `c558179`, Eos em `89aa009`, Bosun em `15461c1` e GLiNER em `587dada`. Código em
  `bench/controles/verificador_treinado/`.

Cada verificador teve uma verificação independente, com código próprio só de biblioteca
padrão, num container sem rede e com o projeto montado só para leitura. Ela reimplementou a
guarda, as políticas, a escolha de τ, o F1, o bootstrap e o AUROC a partir dos arquivos
brutos, e conferiu o SHA-256 dos arquivos contra o Hub. Os números desta página são os que ela
confirmou. Onde ela corrigiu a leitura, a correção já está no texto; o que ficou sem
confirmação está listado no fim.

## 1. Os tetos

Antes de treinar qualquer coisa, pusemos o gabarito no lugar de cada componente. Isso mede
quanto cada desenho renderia com uma peça perfeita.

| Gabarito no lugar de | 305 | 172 |
|---|---|---|
| Nada (a produção, com a régua de reserva) | 0,8180 | 0,8392 |
| Reserva: onde o Gama hesita, entra o span do gabarito | 0,8515 | 0,8632 |
| Decisor da política A | 0,8432 | 0,8532 |
| Verificador na política B | 0,8678 | 0,8795 |
| Verificador na política C | 0,9542 | 0,9516 |

O reserva perfeito passa o decisor perfeito da A porque pode pôr um span que nem o Gama nem a
régua marcaram, e o decisor só escolhe entre os dois. A C muda de patamar. Julgando também os
spans confiantes, o teto fica 0,11 acima da produção nas 172. O maior pedaço do espaço que
sobra está nos spans fortes do Gama, justamente onde a guarda não mexe.

Reservas medidos com a guarda intacta:

| Reserva | 305 | 172 |
|---|---|---|
| Nenhum | 0,8058 | 0,8264 |
| Régua (a produção) | 0,8180 | 0,8392 |
| GLiNER multi v2.1 zero-shot | 0,8171 | 0,8316 |
| Qwen3-8B zero-shot | 0,8242 (não confirmado) | não medido |
| Régua e GLiNER juntos | 0,8016 (não confirmado) | não medido |
| Gabarito | 0,8515 | 0,8632 |

Nenhum chega perto do teto. Os GLiNER, nas duas versões, zero-shot e treinados, não se
distinguem da régua em nenhum dos dois conjuntos; a comparação completa está na
[página do GLiNER 2.5](2026-09-29_gliner-2-5.md). As linhas do Qwen3-8B e da régua com o
GLiNER não passaram pela verificação independente.

## 2. O Laya sem treino

O primeiro juiz testado foi o Laya (`convaiinnovations/laya@55cf4c4`, subpasta
`multilingual`, biblioteca `laya` 0.3.22), um modelo de decisão do Jev Decision Index (seção
4), sem treino nenhum. Três formulações da pergunta foram declaradas antes, e a escolha seria
pelo maior AUROC sobre os 2.979 candidatos das 305. Deram 0,5001, 0,4988 e 0,5040. Ficou a
terceira, de escolha A/B, que marca 0,4955 nas 172 e 0,4383 no estresse.

| Política | τ | 305 | 172 | 172 − produção (IC95) | Estresse, documentos mudados |
|---|---|---|---|---|---|
| A | 0,95 | 0,8022 | 0,8267 | −0,0125 (−0,0264 a −0,0006) | 0 |
| B | 0,95 | 0,8030 | 0,8293 | −0,0098 (−0,0239 a +0,0026) | 1 |
| C | 0,05 | 0,5740 | 0,5952 | −0,2440 (−0,2866 a −0,1997) | 563 |

É ruído. Na A e na B o τ foi para o topo da grade, e mesmo aceitando o mínimo a A fica abaixo
da produção com o IC95 inteiro abaixo de zero. Na C o Laya desmonta o estresse.

## 3. Dados para treinar um verificador

Um verificador treinado precisa de candidatos rotulados parecidos com os da avaliação e de
nenhum documento da avaliação. O conjunto foi montado do zero, com dois pedaços.

Ementas. A fonte são as 1.818 ementas que a destilação do v1.3 viu sem rótulo. Nenhuma tem id
ou texto normalizado igual a um documento das 305, das 172, do estresse ou do dev. Uma regra
de quase-duplicata declarada antes de olhar os casos (Jaccard de shingles de 8 palavras
≥ 0,5) tirou mais 17, 10 parecidas com ementas das 305 e 7 com as 172, com Jaccard de até
0,98. Ficaram 1.801.

A prata saiu do mesmo processo que anotou o gabarito das ementas, com o código de anotação e
de adjudicação reaproveitado sem mudança. DeepSeek-V4-Pro e Kimi-K3 anotam com temperatura 0
e precisam devolver o texto idêntico. O que os dois marcam igual (IoU ≥ 0,8) vira prata; cada
divergência passa pelos critérios, depois pelas convenções votadas, depois pelo terceiro voto
do GLM-5.3, e o que continua ambíguo sai. Rodou nas 1.801, sem amostra, em cerca de 2h16.
Foram 1.504 ementas anotadas e 297 descartadas no alinhamento (16,5%), com 5.110 spans de
prata, 4.487 deles por concordância. Sobraram 37 spans ambíguos em 11 ementas, e os 45
candidatos que os cruzam saíram.

Estilo da organização. São 500 documentos do split de treino do `final_v3` (244 N1, 256 N2),
sorteados entre 4.874 elegíveis. Saíram antes os 536 cujo identificador coincide com um do
estresse; é só o esquema de nomes (Jaccard máximo contra o estresse de 0,28), mas ficaram de
fora por conservadorismo. O rótulo vem do ouro sintético. Cada documento ganhou até 2
negativos de borda, um span do ouro com a borda deslocada até IoU < 0,5, sem invadir outra
citação: 1.000 ao todo (268 com o início truncado, 263 com o fim truncado, 204 estendidos,
265 deslocados).

Os candidatos saem do Gama v1.3 cru e da régua, rodados no container, aparados como no harness
e marcados com os grupos da guarda. Rótulo 1 se o candidato casa com o ouro ou a prata (mesmo
tipo, IoU ≥ 0,5). Nas ementas VAGA conta como JURIS; no `final_v3` ela fica separada, como o
oráculo faz no estresse. O construtor, rodado sobre os spans do harness, reproduz campo a campo
os 13.664 candidatos da avaliação, estado incluído.

O split é por documento e estratificado por conjunto. Uma validação interna com 10% dos
documentos (151 ementas e 50 do `final_v3`) escolheu hiperparâmetros e épocas, e nenhum
documento aparece nos dois lados.

| | Exemplos | Positivos |
|---|---|---|
| Ementas, candidatos do Gama | 8.731 | 48,7% |
| Ementas, candidatos da régua | 5.163 | 66,5% |
| `final_v3`, candidatos do Gama | 3.704 | 100% |
| `final_v3`, candidatos da régua | 3.572 | 88,9% |
| `final_v3`, negativos de borda | 1.000 | 0% |
| Total (treino / validação) | 22.170 (20.062 / 2.108) | 14.563 (13.173 / 1.390) |

Nas ementas, 55,3% dos exemplos são positivos; entre os candidatos de avaliação são 55,2% nas
305 e 57,4% nas 172. Entre os elegíveis da A nas ementas, 13,5% são positivos.

Ficou um buraco, e ele volta mais adiante. O Gama v1.3 foi treinado no `final_v3`, então lá
os candidatos dele são todos fortes e todos positivos, e nenhum é elegível para a A. Os
negativos no estilo da organização vêm só da régua (cerca de 11%) e das bordas. E ali o
candidato da régua fora de qualquer span do Gama, que é o que a B decide, tem 9 exemplos no
treino, todos negativos.

## 4. Quem treinar: a triagem do Jev Decision Index

O [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) é um
placar comunitário de reproduções abertas do Jev, o modelo de decisão da TypeSafe que só
existe por API e por isso não é elegível ([modelos elegíveis](../conceitos/modelos-elegiveis.md)).
Todo modelo roda o mesmo painel congelado, e o índice resume o resultado no balanced_skill:
média de cinco áreas com peso igual, cada benchmark corrigido pelo acaso (0 é chute, 100 é
perfeito). Usamos a edição 0.2.1.

O filtro teve quatro condições. Licença permissiva. Até cerca de 1,5B de parâmetros, para
caber em US$ 2,50 de treino por candidato e na checagem do dev em CPU, num container de 4 GB.
Carga pela classe de classificação de sequência do transformers, sem código remoto. E um
tokenizer que não despedace português jurídico. Entre os que passaram, pesaram as áreas
Retrieval & Classification (R&C) e Language Understanding (LU), e dentro delas os benchmarks
mais parecidos com a tarefa, que é julgar um trecho dentro do contexto: ContractNLI, NLI4CT,
HoVer e ANLI.

| Escolhido | Índice: balanced / R&C / LU | Base | Por quê |
|---|---|---|---|
| `llm-semantic-router/Decision-1.0-Eos-0.8B@363c4a5` | 18,41 / 30,6 / 14,6 | Qwen3.5-0.8B híbrido (18 camadas GatedDeltaNet, 6 de atenção plena), 752M | maior índice com licença permissiva até 1,5B; o melhor do recorte nos NLI (ContractNLI 18,4, NLI4CT 23,3) |
| `Hanno-Labs/bosun-v3.1-0.6b@1d8b6f9` | 14,32 / 34,2 / 3,8 | Qwen3-0.6B com atenção padrão, LoRA de decisão por cima | o melhor R&C do recorte até 1B; outra arquitetura, caso o DeltaNet desse problema; o mais barato de treinar. Fraco em LU, com 0 no ContractNLI e no ANLI |
| `convaiinnovations/laya@55cf4c4` | 6,04 / 7,1 / 9,0 | encoder ModernBERT, na subpasta multilíngue | já testado sem treino (seção 2) |

Os três são Apache-2.0. Entraram mais dois verificadores de fora do índice. O encoder do
próprio Gama v1.3 com uma cabeça nova é o controle sem pré-treino de decisão. O GLiNER multi
v2.1 treinado dos [controles](2026-09-29_controles.md) entra como verificador sem treino novo.

| Ficou de fora | Motivo |
|---|---|
| jpt-0.8b | licença não comercial (CC-BY-NC-4.0). Seria o mais forte do recorte (19,22; LU 23,0) |
| kev-0.8b | mesma base e mesmo tokenizer do Eos; R&C 19,1 contra 34,2 do Bosun; o adapter mira o backbone de texto e a base fixada é um checkpoint multimodal |
| bosun-v3.1-1.7b, decider-2b, this-that-model-1.2 | acima do teto de tamanho, entre 1,7B e 1,9B. O decider-2b tem o maior índice abaixo de 4B (28,97) |
| Intern-Decision-0.8B | fraco nas áreas pedidas (11,94; R&C 6,4) e checkpoint multimodal |
| Tev1-0.8B-experimental | sem licença declarada |
| lavoir | licença não comercial, só inglês |
| família GLiNER (GLiNER2.5-Decide e afins) | o fine-tune do GLiNER já está coberto pelos controles e pela página do 2.5, e GLiNER não roda no container, o que impede a checagem do dev |
| mojev, Lumma-fev, Julia-1, system-one-mini | código remoto ou pacote próprio, só inglês, ou da mesma família do encoder do Gama |
| Decision-1.0-Kai-0.6B, Decision-1.0-Lex-0.6B | formato nativo sem classe do transformers e termos do Gemma no repositório |
| LFM2.5-350M-RLCD | licença própria, sem português, sem classe de classificação no transformers |
| rlcd-modernbert-151m, Qwen-2.5-1B-RLCD, system-one-gemma | cobertura baixa no índice, técnica de inferência sem pesos, ou pesos fora do Hub |
| todo o resto, de 2B para cima | acima do teto |

## 5. Verificadores treinados

| Verificador | Ponto de partida | O que treina | Entrada | Escolhas, só na validação interna | Treino |
|---|---|---|---|---|---|
| Encoder do Gama | `vinimlo/gama@5f924ca`, cabeça de 2 rótulos com o classificador reiniciado | tudo (257M) | o estado, até 512 tokens | lr {2e-5, 5e-5} × pooling {mean, cls}, até 4 épocas: 5e-5, cls, época 3 | 28 min, L4 |
| Laya treinado | `laya@55cf4c4`, subpasta multilíngue | encoder e cabeça de decisão, pelo laço do notebook oficial de fine-tune | o estado e a pergunta da seção 2 | lr do encoder {2,5e-5; 1e-5}, até 3 épocas, parada por Brier: 2,5e-5, época 3 | 53 min, L4 |
| Eos 0.8B | backbone do `Decision-1.0-Eos-0.8B@363c4a5`, cabeça linear nova de 1 logit | tudo menos os embeddings (498M de 752M) | o estado e um sufixo com trecho, tipo, forma e pergunta, até 384 tokens | lr {1e-5, 2e-5}, até 2 épocas: 1e-5 | cerca de 51 min por ponto da grade, L4 |
| Bosun 0.6B | `Qwen/Qwen3-0.6B@c1899de` com o LoRA do `bosun-v3.1-0.6b@1d8b6f9`, cabeça nova de 1 logit | o LoRA (r=16) e a cabeça | o estado e um sufixo com a pergunta, até 384 tokens | lr {1e-4, 3e-4}, até 3 épocas: 1e-4, passo 1.872 | 35 min, A100, a grade inteira |
| GLiNER v2.1 treinado | `vinimlo/gama-exp-gliner-ft@12f4e18`, dos controles | nada novo | os spans que ele já tinha extraído | score = maior score de span dele do mesmo tipo com IoU ≥ 0,5, ou 0 | nenhum job |

Todos com uma semente só. O Laya não tem função de treino na biblioteca, então o laço do
notebook oficial (RLCD somado à entropia cruzada, depois o ajuste de temperatura) foi
reproduzido com as funções dela numa L4 em bf16, no lugar de duas T4. A parada do Laya mudou
no meio. Um primeiro job parava pela perda log e foi cancelado na primeira época, quando ela
subiu de 0,25 para 0,54 enquanto Brier e AUROC melhoravam, e o critério virou Brier; a troca
usou só a validação interna e ficou registrada. No Laya o melhor ponto foi a última
avaliação, e no Eos o melhor ficou nos últimos passos; mais épocas não foram testadas.

| AUROC | 305 | 172 | Estresse |
|---|---|---|---|
| Laya sem treino | 0,504 | 0,496 | 0,438 |
| GLiNER v2.1 treinado | 0,849 | 0,834 | 0,995 |
| Encoder do Gama | 0,988 | 0,989 | 0,998 |
| Laya treinado | 0,979 | 0,983 | 0,994 |
| Eos 0.8B | 0,990 | 0,993 | 0,997 |
| Bosun 0.6B | 0,987 | 0,991 | 0,997 |

Na validação interna, que escolheu grade e época e por isso é otimista, os quatro treinados
ficam entre 0,9875 e 0,9957. O AUROC do estresse engorda com os 4.447 spans do Gama, todos
positivos e fáceis; só na régua, o do Eos é 0,9941. No dev não há AUROC, porque os 384
candidatos são todos positivos.

F1 de extração por política. O τ é o escolhido nas 305; a diferença é contra a produção.

Política A (teto: 0,8432 nas 305, 0,8532 nas 172, +0,0140 sobre a produção nas 172):

| Verificador | τ | 305 | 172 | 172 − produção (IC95) | Estresse, mudados | Dev, mudados |
|---|---|---|---|---|---|---|
| Laya sem treino | 0,95 | 0,8022 | 0,8267 | −0,0125 (−0,0264 a −0,0006) | 0 | 0 |
| GLiNER v2.1 treinado | 0,90 | 0,8191 | 0,8369 | −0,0023 (−0,0146 a +0,0087) | 0 | 0 |
| Encoder do Gama | 0,80 | 0,8364 | 0,8465 | +0,0073 (+0,0013 a +0,0139) | 0 | 0 |
| Laya treinado | 0,60 | 0,8373 | 0,8479 | +0,0087 (+0,0011 a +0,0170) | 0 | 0 |
| Eos 0.8B | 0,85 | 0,8387 | 0,8502 | +0,0110 (+0,0041 a +0,0186) | 0 | 0 |
| Bosun 0.6B | 0,45 | 0,8365 | 0,8488 | +0,0096 (+0,0025 a +0,0173) | 0 | 0 |

Política B (teto: 0,8678 e 0,8795, +0,0403 nas 172):

| Verificador | τ | 305 | 172 | 172 − produção (IC95) | Estresse, mudados | Dev, mudados |
|---|---|---|---|---|---|---|
| Laya sem treino | 0,95 | 0,8030 | 0,8293 | −0,0098 (−0,0239 a +0,0026) | 1 | 0 |
| GLiNER v2.1 treinado | 0,70 | 0,8411 | 0,8602 | +0,0210 (+0,0065 a +0,0346) | 0 | 0 |
| Encoder do Gama | 0,80 | 0,8597 | 0,8704 | +0,0312 (+0,0213 a +0,0417) | 48 | 0 |
| Laya treinado | 0,60 | 0,8610 | 0,8731 | +0,0339 (+0,0235 a +0,0449) | 29 | 0 |
| Eos 0.8B | 0,85 | 0,8627 | 0,8747 | +0,0356 (+0,0262 a +0,0458) | 52 | 0 |
| Bosun 0.6B | 0,45 | 0,8606 | 0,8733 | +0,0341 (+0,0239 a +0,0450) | 40 | 0 |
| Sem verificador: produção e todo span da régua fora do Gama | | 0,8281 | 0,8487 | +0,0096 (−0,0023 a +0,0211) | 60 | 0 |

Política C (teto: 0,9542 e 0,9516, +0,1124 nas 172):

| Verificador | τ | 305 | 172 | 172 − produção (IC95) | Estresse, mudados | Dev, mudados |
|---|---|---|---|---|---|---|
| Laya sem treino | 0,05 | 0,5740 | 0,5952 | −0,2440 (−0,2866 a −0,1997) | 563 | não medido |
| GLiNER v2.1 treinado | 0,70 | 0,8008 | 0,7857 | −0,0535 (−0,1134 a −0,0012) | 3 | não medido |
| Encoder do Gama | 0,55 | 0,9140 | 0,9119 | +0,0727 (+0,0496 a +0,0962) | 383 | 14 |
| Laya treinado | 0,90 | 0,9113 | 0,9154 | +0,0763 (+0,0550 a +0,0976) | 463 | 17 |
| Eos 0.8B | 0,05 | 0,9205 | 0,9105 | +0,0713 (+0,0488 a +0,0940) | 376 | 18 (não confirmado) |
| Bosun 0.6B | 0,10 | 0,9177 | 0,9117 | +0,0725 (+0,0522 a +0,0937) | 254 | 15 |

Na A, os quatro treinados passam a produção nas 172 com o IC95 inteiro acima de zero, de
+0,0073 a +0,0110, e pegam de 52% a 79% do que o decisor perfeito ganharia ali. Nas 305 a
diferença vai de +0,018 a +0,021, mas é o conjunto em que o τ foi escolhido. É ganho real e
pequeno. Pouco mais de um ponto de F1 no melhor caso, com o limite inferior entre +0,0011 e
+0,0041. Não houve comparação pareada entre os verificadores, e a distância entre eles nas 172
(0,8465 a 0,8502) cabe com folga dentro de qualquer um desses intervalos. O F1 nas 305 fica
quase plano ao longo da grade de τ para todos, então o τ escolhido sai mais do desempate que
de um pico. O GLiNER não rende nada aqui: entre os elegíveis da A, ele dá score 0 a metade dos
positivos.

Na B o espaço é maior, e os treinados pegam de 77% a 88% dele nas 172. Nenhum passa no
estresse. Os 61 candidatos elegíveis da B no estresse são todos negativos: o número CNJ do
próprio processo, no cabeçalho, como "Habeas Corpus nº ..." em documentos do Superior
Tribunal Militar e "Autos do recurso nº ..." nos dos TREs. Os treinados aceitam esse número
com score alto. No encoder do Gama os 48 aceitos vão de 0,81 a 0,999 (mediana 0,959); no Eos,
de 0,964 a 0,9997. O Laya treinado aceita 29 dos 61 no τ 0,60. Nenhum τ da grade resolve: o
Eos muda 52 documentos em toda a grade, o encoder do Gama ainda muda 28 em 0,95, o Bosun 38,
e o Laya vai de 51 (τ 0,05) a 8 (τ 0,95). A nota oficial do estresse cai de 1,09999 para
1,09200 com o Eos e para 1,09379 com o Bosun. A causa mais provável está na seção 3: o treino
tinha 9 exemplos desse tipo.

A linha sem verificador foi medida pela verificação, depois de ver os resultados, e serve
só de referência. Aceitar todo span da régua fora do Gama muda 60 documentos do estresse.
Os treinados aceitam de 29 a 52 dos mesmos 61 negativos, ou seja, no estilo da organização
ficam mais perto de aceitar tudo que de rejeitar tudo. Nas ementas, a estimativa deles nas
172 fica cerca de três vezes acima da regra trivial, uma diferença que ninguém testou.

O GLiNER treinado é o único que passa na B sem mudar documento. Ele dá score 0 aos 61
negativos e sobe o F1 nas 172 com o IC95 acima de zero. Só que perto da metade desse ganho
vem sem verificador nenhum. Contra a regra trivial, a B com o GLiNER ganha +0,013 (IC95
+0,001 a +0,025) nas 305 e +0,0115 (−0,0035 a +0,0266) nas 172, que não se distingue de zero.
O que só ele entrega é a rejeição do cabeçalho.

A C dá os números mais altos, +0,071 a +0,076 nas 172, e muda de 254 a 463 documentos do
estresse e de 14 a 18 do dev. O mecanismo é saturação. Dois candidatos sobrepostos e ambos
positivos, o do Gama e o da régua com outra borda, recebem score perto de 1, e o "maior score
vence" decide pelo ruído numérico. No Laya treinado o score máximo é 0,97716 (a temperatura
ficou limitada em 5,0) e 60% dos candidatos passam de 0,976; 98,5% das trocas do estresse têm
margem menor que 1e-3. No encoder do Gama, inverter o desempate nos quase empates (score
arredondado em 5 casas, régua primeiro) leva o estresse de 383 para 589 documentos mudados,
sem mexer no F1 da A e da B. No dev o F1 continua 1,0, mas a nota oficial cai (1,09971 com o
Bosun, contra 1,10000 da produção), porque borda e forma mudam. No Laya treinado, 14 das 29
trocas do dev põem dentro do span a preposição anterior ("no", "da"), que a própria pergunta
manda deixar fora. A verificação aponta uma causa provável. O rótulo de treino é IoU ≥ 0,5,
então borda um pouco deslocada conta como positiva, e os negativos de borda do treino têm
todos IoU < 0,5. O verificador nunca aprendeu a preferir a borda exata. Uma C com outro
desempate, em que o span forte do Gama vence dentro de uma margem, mudaria a política e não
foi testada.

O GLiNER piora a C. Nas ementas, entre 18% e 23% dos spans fortes corretos recebem score 0
dele. No estresse isso quase não acontece (3 de 4.447 spans fortes), e são exatamente os 3
documentos mudados.

## 6. GLiNER2.5-multi-Decide

A triagem deixou de fora a família GLiNER, e o `fastino/GLiNER2.5-multi-Decide` (revisão
`a35a0cd`, 287M, Apache-2.0) foi testado depois, no mesmo protocolo. É o classificador
multilíngue da família 2.5, feito para escolher entre rótulos dados na hora. A pergunta foi
binária (sim ou não, sobre o trecho entre `[[ ]]`), com três formulações fixadas antes e a
escolha nas 305.

Sem treino ele também não tem sinal. As três formulações dão AUROC de 0,449, 0,478 e 0,446 nas
305, abaixo do acaso; a escolhida fica em 0,515 nas 172. Na política A ele perde da produção
(0,8054 e 0,8264; nas 172, −0,0127, IC95 −0,0260 a −0,0014).

Fine-tunado nos mesmos dados (taxa do encoder e época escolhidas na validação interna), cai na
faixa dos outros quatro:

| Política | τ | F1 nas 305 | F1 nas 172 | 172 contra a produção (IC95) | Estresse mudados |
|---|---|---|---|---|---|
| A | 0,90 | 0,8367 | 0,8504 | +0,0113 (+0,0041 a +0,0192) | 0 |
| B | 0,90 | 0,8599 | 0,8755 | +0,0364 (+0,0268 a +0,0468) | 31 |
| C | 0,70 | 0,9189 | 0,9167 | +0,0775 (+0,0547 a +0,1010) | 36 |

A AUROC fica em 0,988 nas 305 e 0,989 nas 172. Os 36 documentos da C no estresse enganam:
quase todos os scores ali saturam em 1,0, e é o desempate entre Gama e régua que segura o
número; com a régua vencendo o empate, sobem para 591. A B aceita o mesmo número de processo
no cabeçalho que os outros aceitavam. O dev não foi medido, porque a biblioteca `gliner2` não
está na imagem. Custo: cerca de US$ 0,90 em quatro jobs na L4.

## Custo e tempo

| | Jobs | Hardware | Custo | Por candidato, GPU | Por candidato, CPU do container |
|---|---|---|---|---|---|
| Dados | 0 | container local e as LLMs da anotação | US$ 0 de crédito | | |
| Encoder do Gama | 4 | L4 | US$ 0,44 | 3,9 ms (fp32) | cerca de 165 ms |
| Laya treinado | 5 | L4 | US$ 0,97 | 13,8 a 19,6 ms | cerca de 512 ms por estado |
| Eos 0.8B | 6 | L4 | US$ 1,63 | 15,9 ms (bf16) | 796 ms |
| Bosun 0.6B | 5 | L4 e A100 | US$ 1,67 | 10,3 ms em lote de 64; 24,5 ms por estado em lote de 1 | 6,5 s, numa medida contaminada |
| GLiNER v2.1 treinado | 0 | | US$ 0 | 11,4 ms | não roda no container |
| Total | 20 | | US$ 4,71 | | |

Tempo de execução de cada job pelos preços do `hf jobs hardware` (L4 US$ 0,80/h, A100
US$ 2,50/h); fila não conta. A verificação refez a conta a partir dos tempos faturados e
chegou a US$ 4,70. O job cancelado do Laya entra pelo limite superior (11,6 min), porque o HF
não informa a duração de job cancelado. O orçamento era de US$ 11.

Com GPU, todos cabem no envelope da competição com folga: um documento tem de 10 a 15
candidatos, o que dá décimos de segundo. Em CPU, só o encoder do Gama fica num custo parecido
com o do próprio Gama, cerca de 1,6 s a mais por ementa. O Eos em CPU ficaria entre 8 e 12 s
por documento. O tempo do GLiNER é o dele por documento dividido pelo número de candidatos: 0,06 s
por documento nas 305, 0,08 s nas 172 e 0,21 s no estresse. Em todos os casos a produção
ganharia um segundo modelo: o encoder do Gama tem 257M de parâmetros, e o Eos, 752M.

## Leitura

1. Treinar dá o sinal que o zero-shot não tinha. O Laya vai de AUROC 0,50 a 0,98, e os quatro
   treinados ficam entre 0,979 e 0,993 nas ementas. O índice pouco disse. O Laya, com o menor
   índice dos três escolhidos (6,04, contra 18,41 do Eos), fica a 0,01 de AUROC do Eos e na
   mesma faixa de F1, e o encoder do próprio Gama, que não está em índice nenhum, chega ao
   mesmo lugar sendo o mais barato. Na nossa leitura, com 20 mil candidatos rotulados, quem faz
   o verificador são os dados.
2. Na A o ganho existe e é pequeno: +0,007 a +0,011 nas 172, sobre um gabarito de LLM. A
   segurança dela no estilo da organização é a da guarda, por construção, porque ela só decide
   onde a guarda já decidiria. Além disso não foi testada, porque não há onde testar.
3. B e C são onde está o espaço, e nenhum verificador treinado as deixa usáveis. A B aceita o
   número do próprio processo no cabeçalho, que o treino quase não tinha. A C troca borda
   certa por outra quase certa num desempate que o ruído decide.
4. O GLiNER treinado é o único que mexe na B sem tocar o estilo da organização, e nas 172 o
   ganho dele sobre a regra sem verificador não se distingue de zero.
5. Na competição nada muda. A submissão está congelada, e a única política que passaria no
   critério, a A, não mexe em documento nenhum no estilo da organização.

## O que mudaria a decisão sobre a guarda

A guarda fica como está. Nós só reabriríamos a questão com um destes resultados, medidos num
conjunto que não escolheu nada:

- A com verificador mantendo o ganho num gabarito humano, grande o bastante para pagar um
  segundo modelo. Com +0,01 sobre gabarito de LLM, não paga.
- B retreinada com negativos do tipo que falhou (número do próprio processo no cabeçalho,
  no estilo da organização), sem mudar nenhum documento no estresse e no dev, e com ganho
  sobre a regra sem verificador com o IC95 acima de zero.
- C com um desempate que preserva o span forte do Gama, também sem mudar documento no estilo
  da organização. É o maior prêmio: o teto fica 0,11 acima da produção nas 172.

## Limites gerais

- O gabarito das ementas é de LLM adjudicado, e a prata do treino saiu do mesmo processo. O
  verificador aprende a convenção dos anotadores, e parte do AUROC perto de 0,99 mede
  concordância com ela.
- As 305 são validação: todo τ saiu delas, e os números ali são otimistas. Só as 172
  confirmam, e elas já tinham ajudado a escolher o v1.3. São seis verificadores e três
  políticas sobre as mesmas 172; escolher o melhor olhando as 172 tiraria delas o papel de
  confirmação.
- Uma semente por verificador e grades pequenas, de dois a quatro pontos. Não houve comparação
  pareada entre verificadores nem ablação do Bosun contra o Qwen3-0.6B sem o adapter, então
  nenhum ganho se atribui ao pré-treino de decisão.
- Contaminação residual. Nenhum id nem texto das avaliações está no treino, mas de 7 a 9
  candidatos das 172 têm a janela idêntica à de um exemplo de treino (parágrafos-padrão do
  STF), e até três pares de ementas passam de 0,5 de Jaccard com outra tokenização. O efeito medido
  é desprezível. Sem os documentos marcados, o Bosun mantém +0,10 e +0,07 na C e +0,02 e +0,01
  na A; o Eos, sem 13 documentos, mantém todos os IC95 do mesmo lado do zero.
- O estresse e o `final_v3` do treino saem do mesmo gerador. O dev é pior: 323 dos 500
  documentos do `final_v3` de treino foram gerados com um documento do dev como molde, e 55,5%
  dos shingles de 10 palavras do dev aparecem literalmente no treino. O dev não tem candidato
  negativo. Como teste de segurança ele vale pouco.
- Execução não determinística. O kernel do Eos (flash-linear-attention, em Triton) repontuou a
  validação com diferença de até 0,0148 e o mesmo AUROC; dois jobs do Laya com a mesma semente
  divergem na meia época. Em bf16 o Laya difere do fp32 em até 0,128, o que embaralharia os
  desempates da C.
- Achado ao lado, fora do escopo. As 17 quase-duplicatas de ementas de avaliação estavam entre
  as 1.818 que a destilação do v1.3 viu sem rótulo humano, só com a saída do professor. O
  sorteio das ementas de teste deduplica apenas por texto normalizado idêntico. O efeito disso
  na leitura do v1.3 não foi medido.

## O que ficou sem confirmação

- As notas oficiais do estresse e do dev do encoder do Gama (1,09259 na B, 1,08994 na C, 1,09975
  no dev), do Laya treinado (1,09547, 1,09403, 1,09964) e do GLiNER na C (1,0997) saíram do
  harness, e a verificação não as refez. As do Eos (1,09200 e 1,09179) e do Bosun (1,09379,
  1,09235 e 1,09971 no dev) foram recalculadas a partir dos JSON gravados.
- Os 18 documentos mudados no dev pelo Eos na C (nota 1,09966): os scores do dev por candidato
  não foram gravados.
- Os reservas Qwen3-8B (0,8242) e régua com GLiNER (0,8016) nas 305 saíram do script original
  da análise da guarda, rodado de novo sem mudança. A verificação independente cobriu só
  nenhum, régua, GLiNER v2.1 zero-shot e gabarito.
- Os tempos por candidato foram medidos nos jobs e no container. A verificação refez o do Eos
  (15,9 ms) e o do GLiNER (11,4 ms). O do Laya (13,8 ms) é o mais favorável, com estados
  deduplicados e sem a tokenização. O de CPU do Bosun foi tomado com outro container usando a
  máquina.
- As medidas de tokenizer da triagem: em 800 janelas, o Qwen3.5 dá 186,5 tokens por janela,
  contra 185,8 do tokenizer do Gama, e o Qwen3 sai 12% mais longo.
- O `--timeout` dos jobs só pôde ser conferido nos do Eos; o `hf jobs inspect` não mostra o
  campo.
