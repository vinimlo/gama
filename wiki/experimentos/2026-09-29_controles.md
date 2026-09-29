# 2026-09-29 · Controles: encoder congelado, concorrentes treinados, sementes e confiança

Pergunta: o [benchmark](2026-09-23_bench-extratores-crus.md) pôs o Gama contra extratores
sem treino e deixou perguntas que ele não respondia. Quanto do resultado vem de adaptar o
encoder, e não só de treinar uma camada de saída em cima dele? Um GLiNER ou um BERTimbau
treinados nos mesmos dados chegam ao Gama? A vantagem do v1.3 sobre o v1.2 passa da variação
entre sementes? Os trechos do Qwen3-8B que não se alinharam ao texto eram citações que ele
não viu ou citações que ele reescreveu? E a confiança do extrator ainda ordena erros fora do
molde?

## Desenho

Tudo passou pelo mesmo harness de avaliação (`bench/controles/avaliar.py`), sem mudança
entre experimentos:

- Estresse difícil (600 documentos): métrica oficial pelo caminho de produção (aparar,
  resolver, confiança calibrada, `vendor/kaggle_metric.py`) e F1 de extração.
- Texto real: F1 de extração com mesmo tipo, IoU ≥ 0,5, casamento 1 para 1 e VAGA contada
  como JURIS, nas 305 ementas do benchmark e nas 172 sorteadas depois. Entradas
  `bench/entrada_remota.jsonl` e `bench/entrada_novas.jsonl` do dataset
  `vinimlo/gama-goldenset`, revisão `2fff5f6`.
- As 305 são validação: todo limiar escolhido em texto real saiu delas. As 172 são
  confirmação e não informaram nenhuma escolha. Hiperparâmetro de treino, quando houve
  escolha, saiu da reserva interna de 590 documentos do `final_v3` (campo `split` do
  `meta.jsonl`), que é do estilo da organização.
- O gabarito das 305 e das 172 é de LLM adjudicado (DeepSeek-V4-Pro e Kimi-K3, desempate pelo
  GLM-5.3), não anotação humana. Toda conclusão sobre texto real vale para esse gabarito.
- Cada candidato aparece cru e com a guarda de produção
  ([D-008](../decisoes/D-008_guarda-do-extrator.md)), no limiar 0,95 e no limiar escolhido
  nas 305 numa grade de 0,50 a 0,99 (medido, e só medido, nas 172).
- IC95 por bootstrap pareado por ementa, 2.000 reamostras, semente 0. A referência é o v1.3
  com a guarda (`vinimlo/gama@5f924ca`): 0,8180 nas 305 e 0,8392 nas 172.
- Treinos no HF Jobs em A100 (o GLiNER em A10G), extrações na L4, como no benchmark. Nenhum
  documento do dev saiu da máquina. Os modelos treinados ficaram em repositórios privados de
  experimento, que serão apagados depois da revisão; os spans extraídos ficam no dataset, em
  `bench/saida/controles/`. Código em `bench/controles/`.

Uma verificação independente recalculou os números principais de cada experimento com código
próprio, sem importar o harness, e reproduziu a métrica oficial a partir dos JSON. Os números
desta página são os que ela confirmou. Onde ela corrigiu a leitura, a correção já está no
texto.

O que treina em cada variante:

| Variante | Ponto de partida | O que treina | Dados | Como as escolhas foram feitas |
|---|---|---|---|---|
| Encoder congelado | `jhu-clsp/mmBERT-base@c595503` | só a saída: a camada densa 768×768 com LayerNorm, herdada do pré-treino, e o classificador de 7 rótulos (596 mil de 307,5 milhões de parâmetros) | `final_v3@31474b1`, os 5.410 documentos do v1.2, mesma receita | taxa de aprendizado numa grade de 5e-5 a 1e-1, pela F1 na reserva; ficou 3e-2 |
| Cabeça sem treino | a mesma base, com o classificador inicial exato do encoder congelado | nada | | diagnóstico, não concorrente |
| GLiNER treinado | `urchade/gliner_multi-v2.1@443d26d`, o do benchmark | tudo (289M) | os mesmos 5.410, em janelas de 200 palavras como na inferência | hiperparâmetros da documentação do GLiNER, sem busca; época e limiar na reserva (época 2, limiar 0,5); largura máxima de span de 33 palavras, a maior citação do treino, fixada antes |
| BERTimbau | `neuralmind/bert-base-portuguese-cased@94d69c9` | tudo (108,3M) | os mesmos, `treino/treinar.py` sem mudança | nenhuma: receita do v1.2, só com `max_len` 512, o teto do BERT |
| v1.2, sementes 7 e 21 | mmBERT-base | tudo | os mesmos | receita e argumentos do treino oficial, trocando só a semente |
| v1.3, sementes 7 e 21 | 12 primeiras camadas do v1.2 oficial | tudo, destilado do v1.2 oficial | `final_v3` e as 1.818 ementas sem rótulo | receita e argumentos da destilação oficial, trocando só a semente |
| Qwen3-8B | `Qwen/Qwen3-8B@b968826` | nada; mesma configuração do benchmark, rodada de novo para gravar as respostas brutas | | categorias da auditoria e sorteio da amostra fixados antes de olhar |

A semente 13 é sempre o peso publicado, não um retreino.

## 1. Encoder congelado: quanto vale adaptar o encoder

O mmBERT original não tem cabeça de rótulos, então "cru" só faz sentido com uma cabeça nova.
Rodamos as duas coisas: a cabeça nova sem treino, como diagnóstico, e a camada de saída
treinada sobre o encoder congelado, que é o controle útil.

| | Estresse, oficial | Estresse, F1 | 305 | 172 |
|---|---|---|---|---|
| Encoder congelado, cru | 0,90182 | 0,8407 | 0,2428 | 0,2467 |
| Encoder congelado, guarda 0,95 | 0,94551 | 0,9084 | 0,3498 | 0,3526 |
| Encoder congelado, guarda 0,99 (escolhido) | 0,90675 | 0,8902 | 0,3879 | 0,3844 |
| v1.2 cru, mesma receita com o encoder treinável | 1,09999 | 1,0000 | 0,6054 | 0,6151 |
| v1.2, guarda 0,95 | 1,09999 | | 0,8076 | 0,8204 |
| Cabeça sem treino, cru | 0 | 0 | 0,0005 | 0,0005 |
| Cabeça sem treino, guarda 0,99 (escolhido) | 0,79708 | | 0,6001 | 0,6174 |

Diferença pareada, encoder congelado menos v1.2:

| | 305 | 172 |
|---|---|---|
| Cru contra cru | −0,3625 (IC95 −0,4011 a −0,3240) | −0,3685 (−0,4122 a −0,3311) |
| Guarda 0,95 contra guarda 0,95 | −0,4578 (−0,4887 a −0,4255) | −0,4678 (−0,5005 a −0,4337) |

Contra o v1.3 com a guarda, todas as variantes ficam entre −0,43 e −0,59, e nenhum IC95 chega
perto de zero.

No estilo da organização a representação pré-treinada já leva longe. Com só a saída treinada,
o mmBERT marca 0,902 no estresse, acima da régua (0,849) e do Qwen3-8B (0,811). No texto real
ele desaba. São 733 acertos para 4.219 falsos positivos nas 305, precisão de 0,15. Com a
mesma receita e o encoder livre, o v1.2 cru marca 0,605. É a adaptação do encoder que separa uma
coisa da outra.

A guarda ajuda menos aqui que no v1.2 e não chega perto de fechar a distância: +0,107 no
limiar 0,95 e +0,145 no 0,99, nas 305, contra +0,202 do v1.2. O motivo é que 63% dos spans do
encoder congelado nas 305 (3.144 de 4.952) têm confiança ≥ 0,95. Esses falsos positivos
confiantes ficam e bloqueiam o span da régua.

A cabeça sem treino confirma o que se esperava dela. F1 de 0,0005: ruído. Com a guarda em
0,99 ela sobe para 0,600 nas 305, quase a régua sozinha (0,663). Só 615 de 73.527 spans
aleatórios passam de 0,99; os outros saem e a régua entra no lugar. Com a guarda, a nota de
um extrator fraco mede a régua, não o extrator. Por isso a comparação que isola o encoder é a
crua.

Limites deste braço. Uma semente só. O orçamento foi o do v1.2 (3 épocas, `max_len` 1024,
lote 8), e a perda ainda caía; mais épocas não foram testadas. O que treina não é um
classificador linear puro, e sim a camada densa herdada do pré-treino mais o classificador; a
versão só com o classificador não rodou. A taxa escolhida (3e-2) e o limiar escolhido (0,99)
ficaram perto da borda das grades. No estresse, a confiança entra pela calibração medida para
o Gama.

## 2. GLiNER treinado nos mesmos dados

| | Estresse, oficial | Estresse, F1 | 305 | 172 |
|---|---|---|---|---|
| GLiNER zero-shot, do benchmark | 0,32732 | 0,453 | 0,4093 | não medido |
| GLiNER treinado, cru | 1,09631 | 0,9993 | 0,7922 | 0,7519 |
| GLiNER treinado, guarda 0,95 | 1,09631 | 0,9993 | 0,7958 | 0,7606 |
| GLiNER treinado, guarda 0,99 (escolhido) | 1,09631 | 0,9993 | 0,7968 | 0,7623 |
| Gama v1.3 cru | 1,09999 | 1,0000 | 0,6201 | 0,6351 |
| Gama v1.3, guarda 0,95 | 1,09999 | 1,0000 | 0,8180 | 0,8392 |

Diferenças pareadas:

| | 305 | 172 |
|---|---|---|
| GLiNER treinado cru − v1.3 cru | +0,1722 (IC95 +0,1468 a +0,1956) | +0,1168 (+0,0788 a +0,1552) |
| GLiNER treinado cru − v1.2 cru | +0,1869 (+0,1623 a +0,2105) | +0,1367 (+0,1033 a +0,1708) |
| GLiNER treinado, guarda 0,99 − v1.3 com a guarda | −0,0212 (−0,0597 a +0,0160) | −0,0768 (−0,1436 a −0,0148) |
| GLiNER treinado, guarda 0,99 − v1.2 com a guarda | −0,0108 (−0,0497 a +0,0258) | −0,0580 (−0,1175 a −0,0019) |
| GLiNER treinado cru − GLiNER zero-shot cru | +0,3830 (+0,3360 a +0,4280) | |

A linha contra o v1.2 importa porque o v1.3 viu as 1.818 ementas sem rótulo na destilação, e o
GLiNER não. O v1.2 teve o mesmo acesso a dados que ele. As contas contra o v1.2 são da
verificação.

No estilo da organização o GLiNER treinado encosta no teto: 6 erros em 4.447 spans, todos em
JURIS. A nota oficial de 1,09631 usa o formato curto, sem confiança de modelo, como o
zero-shot publicado; com o score dele no lugar da confiança, e a calibração do Gama, daria
1,09778.

Sozinho, em texto real, ele passa o Gama sozinho com folga. Com a guarda, não se distingue do
v1.3 nas 305 e perde nas 172; contra o v1.2 o quadro é o mesmo, com a perda nas 172 mais
apertada. A guarda quase não mexe nele (+0,004 a +0,010), e a razão está no score: nas 305,
179 dos 188 falsos positivos do GLiNER treinado têm score ≥ 0,99. No v1.3, 704 dos 958 falsos
positivos crus ficam abaixo de 0,95. A guarda depende de o extrator saber quando hesita. O
GLiNER treinado não sabe.

Limites deste braço. Uma semente e uma configuração, tiradas da documentação, sem busca. A
reserva de 590 documentos fica no teto (F1 de 0,9987 a 0,9993 em todas as épocas e limiares),
então a escolha de época e limiar não discrimina nada. A largura máxima de span subiu de 12
para 33 palavras, e o ganho sobre o zero-shot mistura treino e largura. A divisão de JURIS
em "precedente judicial com número" e "súmula ou tema de tribunal" é heurística; na avaliação
os dois voltam a JURIS. Rodou com transformers 4.57.6, versão fixada no cabeçalho do script
(o gliner 0.2.29 aceitaria qualquer versão abaixo da 5.17). O tempo por documento foi
relatado, mas a verificação não o refez.

## 3. BERTimbau no protocolo final

As dobras já mostravam o BERTimbau empatado com o mmBERT no estilo da organização. Faltava
o protocolo de hoje, com texto real.

| | Estresse, oficial | 305 | 172 |
|---|---|---|---|
| BERTimbau cru | 1,09999 | 0,6968 | 0,7045 |
| BERTimbau, guarda 0,95 | 1,09999 | 0,8133 | 0,8257 |
| BERTimbau, guarda 0,96 (escolhido) | 1,09999 | 0,8146 | 0,8237 |
| v1.2 cru / guarda 0,95 | 1,09999 | 0,6054 / 0,8076 | 0,6151 / 0,8204 |
| v1.3 cru / guarda 0,95 | 1,09999 | 0,6201 / 0,8180 | 0,6351 / 0,8392 |

Diferenças pareadas:

| | 305 | 172 |
|---|---|---|
| BERTimbau − v1.2, cru | +0,0915 (IC95 +0,0594 a +0,1225) | +0,0894 (+0,0579 a +0,1196) |
| BERTimbau − v1.2, guarda 0,95 | +0,0057 (−0,0165 a +0,0287) | +0,0054 (−0,0254 a +0,0342) |
| BERTimbau − v1.3, cru | +0,0767 (+0,0458 a +0,1042) | +0,0695 (+0,0338 a +0,1012) |
| BERTimbau − v1.3, guarda 0,95 | −0,0048 (−0,0249 a +0,0170) | −0,0135 (−0,0444 a +0,0177) |
| BERTimbau − v1.3, cada um no limiar escolhido (0,96 e 0,98) | −0,0034 (−0,0249 a +0,0192) | −0,0105 (−0,0414 a +0,0215) |

Tempo na L4, os três no mesmo job (s/doc):

| | BERTimbau | v1.2 | v1.3 |
|---|---|---|---|
| Estresse | 0,0662 | 0,0753 | 0,0462 |
| 305 | 0,0201 | 0,0273 | 0,0175 |

No estresse os três ficam no teto. O JSON final do BERTimbau é igual ao do v1.3 nos 600
documentos, então o número só confirma que o estresse satura.

Sem guarda, o BERTimbau é melhor em texto real, e a vantagem vem da precisão: 0,620 contra
0,472 do v1.2, com 529 falsos positivos contra 1.023. O recall é um pouco menor (0,795 contra
0,843). Essa vantagem crua passa da variação entre sementes do mmBERT (seção 4): o BERTimbau
fica acima das seis sementes cruas nos dois conjuntos. Vale para a semente 13 dele; a
variação do próprio BERTimbau não foi medida.

Com a guarda, não dá para distinguir o BERTimbau de nenhum dos dois mmBERT. Não é
equivalência demonstrada. Contra o v1.3 a estimativa sai negativa nos dois conjuntos, e a
variação entre sementes da receita do v1.2 com a guarda (0,7605 a 0,8297 nas 305) é maior que
esses intervalos.

Na L4 ele é mais rápido que o v1.2 e mais lento que o v1.3, porque o documento longo vira
várias janelas de 512 tokens. Uma segunda rodada, com a ordem dos modelos invertida, manteve
a mesma ordem; o v1.2 variou até 6,4% entre as rodadas.

Uma curiosidade, exploratória. O vocabulário do BERTimbau não tem "º" depois de letra ou
dígito, e "nº" ou "1º" viram token desconhecido em 30,8% das citações de treino. Sozinho, ele
não mostra penalidade de recall nessas citações; contra o v1.2, o déficit de recall nas 305
é maior nelas (0,797 contra 0,881) que nas outras (0,795 contra 0,828), e nas 172 isso se
inverte.

## 4. Sementes do v1.2 e do v1.3

F1 em texto real com a guarda de produção (0,95):

| | Semente 13, publicada | 7 | 21 | Média | Amplitude |
|---|---|---|---|---|---|
| v1.2, 305 | 0,8076 | 0,7605 | 0,8297 | 0,7993 | 0,0692 |
| v1.3, 305 | 0,8180 | 0,8116 | 0,8174 | 0,8157 | 0,0064 |
| v1.2, 172 | 0,8204 | 0,7847 | 0,8425 | 0,8159 | 0,0578 |
| v1.3, 172 | 0,8392 | 0,8358 | 0,8420 | 0,8390 | 0,0062 |

Cru:

| | Semente 13, publicada | 7 | 21 | Média | Amplitude |
|---|---|---|---|---|---|
| v1.2, 305 | 0,6054 | 0,6106 | 0,6286 | 0,6149 | 0,0232 |
| v1.3, 305 | 0,6201 | 0,6236 | 0,6192 | 0,6210 | 0,0044 |
| v1.2, 172 | 0,6151 | 0,5830 | 0,6556 | 0,6179 | 0,0726 |
| v1.3, 172 | 0,6351 | 0,6296 | 0,6308 | 0,6318 | 0,0055 |

A regra de leitura estava escrita no código de análise: a diferença v1.3 − v1.2 passa da
variação entre sementes se todo v1.3 superar todo v1.2 nas 9 combinações e se a diferença
das médias passar a maior amplitude dentro de um braço. A verificação não conseguiu provar que
a regra é anterior à primeira pontuação, só que o cálculo já estava no código da primeira
rodada.

| | Diferença das médias | As 9 combinações | Positivas | Maior amplitude | Passa? |
|---|---|---|---|---|---|
| Guarda, 172 | +0,0231 | −0,0067 a +0,0573 | 6 | 0,0578 | não |
| Guarda, 305 | +0,0164 | −0,0181 a +0,0575 | 6 | 0,0692 | não |
| Cru, 172 | +0,0139 | −0,0260 a +0,0521 | 6 | 0,0726 | não |
| Cru, 305 | +0,0061 | −0,0094 a +0,0182 | 6 | 0,0232 | não |

O v1.2 com a semente 21 (0,8425 nas 172) passa os três v1.3. O bootstrap das três sementes
somadas dá +0,0226 (IC95 +0,0044 a +0,0396) nas 172, mas ele mede a amostra de ementas, não a
semente.

Depois de ver esses números, comparamos cada aluno com o próprio professor (o v1.2 semente
13). Nas 172 os três ficam acima: +0,0188 (IC95 +0,0030 a +0,0331), +0,0154 (−0,0026 a +0,0322)
e +0,0216 (+0,0053 a +0,0387). Nas 305 ficam entre +0,004 e +0,010, e nenhum IC exclui zero.
É uma análise feita depois, e fica marcada assim.

No estilo da organização as sementes quase não importam. No dev (26 documentos, CPU local) o
JSON final é idêntico ao publicado nas seis, com 1,10000. No estresse, com a guarda, as seis
marcam 1,09999. Sem guarda, o v1.3 com a semente 7 marca uma VAGA espúria ("êsteira",
confiança 0,873) e cai para 1,09966; a guarda remove o span. A reextração da semente 13
reproduz os spans publicados nos três conjuntos, com diferença de confiança zero.

A leitura pede cuidado. O v1.2 varia muito em texto real. A semente 7 perde recall (0,745
contra 0,84 e 0,85 das outras), e a guarda não devolve citação que o modelo não marcou. O
braço do v1.3 varia dez vezes menos, mas mede só a semente da destilação, com o professor
fixo; não diz nada sobre a variação que viria do professor. Comparar as duas receitas de
ponta a ponta pediria destilar também dos v1.2 com sementes 7 e 21, e isso não foi feito.

## 5. Qwen3-8B: o que eram os 583 trechos que não se alinharam

O `bench/extrair_qwen.py` grava só os spans alinhados. As respostas do benchmark se
perderam. Rodamos o Qwen3-8B de novo com a mesma configuração (A100, lote 32, mesma revisão,
mesma build do torch) e o mesmo script, agora gravando a resposta bruta. A rerodada reproduz
o benchmark: spans e contagem de trechos não alinhados iguais por documento nos 905. O arquivo
não é idêntico byte a byte, porque o tempo por documento muda, e a rodada original não gravou
os trechos, só a contagem; que sejam exatamente os mesmos 583 não dá para provar.

Por código, dos 583: 526 não aparecem literalmente no texto, 56 aparecem mas caíram na
política de ocorrência (repetem uma citação já alinhada ou ficam dentro de um span já aceito)
e 1 tem formato inválido. A página do benchmark diz que os 583 não existem no texto como
foram escritos; os 56 existem. E 149 dos 583 (25,6%) são cópias literais dos exemplos do
prompt.

Numa amostra de 120 (semente 20260929; 66 do estresse, 54 das 305), com categorias fixadas
antes de olhar:

| Categoria | Itens | Fração | IC95 (Wilson) |
|---|---|---|---|
| Citação reconhecida e reescrita | 72 | 60% | 51% a 68% |
| Trecho inexistente no documento | 35 | 29% | 22% a 38% |
| Ocorrência ou alinhamento | 10 | 8% | 5% a 15% |
| Não é citação | 3 | 2,5% | 1% a 7% |
| Truncado | 0 | 0 | 0 a 3,1% |
| Formato inválido | 0 | 0 | 0 a 3,1% |

Dos 35 inexistentes, 34 são exemplos do prompt copiados num documento que não os cita. Nas
reescritas, os casos mais comuns são troca de caixa, correção de OCR ou de acento e quebra de
linha desfeita dentro do número; só caixa, sem outra mudança, são 21 das 72. Os subtipos
saíram das notas depois de rotular e são exploratórios.

Quanto disso é citação que o adaptador perdeu? Cerca de 70 de 120 (58%). Os 10 itens de
ocorrência caem sobre um trecho que o próprio Qwen já tinha alinhado, 7 deles como repetição
pura, sem perda nenhuma; duas reescritas também. Falha de reconhecimento mesmo são 38 de
120 (inexistentes mais não citações). A rotulagem teve um revisor só, e 4 das 5 trocas manuais
em relação à sugestão automática foram para "reescrita"; mantidas as sugestões, seriam 69
reescritas. A conclusão não muda. A maior parte do que o Qwen perdeu no alinhamento era
citação que ele reconheceu e escreveu diferente do documento.

A nota do Qwen no benchmark continua valendo: falhar no alinhamento é falha do sistema que
foi avaliado. A auditoria explica o mecanismo. Não calculamos quantos trechos virariam acerto
com outro adaptador.

## 6. A confiança fora do molde

No estresse não há o que ordenar. O v1.3 com a guarda faz 4.447 previsões e não erra
nenhuma; a confiança calibrada fica entre 0,9923 e 0,995 (mais uma de 0,5, que acertou), com
Brier de 0,000083 nos pares.

No texto real, com a confiança do extrator e erro definido como falso positivo no casamento
1 para 1:

| | 305 | 172 |
|---|---|---|
| AUROC, cru | 0,930 (IC95 0,917 a 0,943) | 0,944 (0,927 a 0,958) |
| Risco total (1 − precisão), cru | 0,511 | 0,491 |
| Risco total, com a guarda | 0,200 | 0,163 |
| AURC com a guarda | 0,075 (0,059 a 0,095) | 0,059 (0,041 a 0,079) |
| AURC do oráculo / da ordem aleatória | 0,022 / 0,200 | 0,014 / 0,163 |
| Erro na faixa de 0,95 a 0,98, com a guarda | 41% (172 spans) | 38% (101 spans) |
| Erro na faixa de 0,999 a 1, com a guarda | 6,2% (598 spans) | 6,25% (368 spans) |
| Erro dos spans que a régua pôs no lugar | 43% (36 de 83) | 35% (13 de 37) |

Antes da guarda a confiança separa bem: abaixo de 0,8, 434 de 438 spans são erro nas 305
(267 de 269 nas 172). Acima de 0,95 ainda há gradiente, e a faixa de 0,95 a 0,98 erra quatro
em cada dez. Subir o limiar não rende F1. O limiar escolhido para o v1.3 nas 305 é
0,98, com 0,8181 contra 0,8180 em 0,95, e nas 172 ele dá 0,8342, abaixo dos 0,8392. Os spans
que a régua põe no lugar também erram bastante.

O AUROC com a guarda (0,784 nas 305, 0,777 nas 172) mede outra população, já filtrada, e não
se compara com o cru como melhora ou piora. Ao span da régua inserido pela guarda foi dada a
maior confiança dos spans fracos que ele cruza, uma convenção. O bootstrap reamostra só os
documentos com alguma previsão. O ouro das ementas tem omissões (por exemplo, "ENUNCIADO 284
DA SÚMULA/STF" em `real_01426`), então o risco medido é um teto. E a confiança calibrada, a
que entra no Brier, não é avaliada fora do estilo da organização, porque o ouro do texto real
não tem `id_canonico`.

## Custo

| Experimento | Jobs | Hardware | Custo |
|---|---|---|---|
| Encoder congelado | 3 | A100, L4 | US$ 1,35 |
| GLiNER treinado | 3 | A10G, L4 | US$ 2,16 |
| BERTimbau | 4 | CPU, A100, L4 | US$ 0,51 |
| Sementes | 6 | A100, L4 | US$ 2,73 |
| Qwen3-8B de novo | 1 | A100 | US$ 1,26 |
| Total | 17 | | US$ 8,01 |

Tempo de execução de cada job pelos preços do `hf jobs hardware` (A100 US$ 2,50/h, A10G
US$ 1,50/h, L4 US$ 0,80/h); fila não conta.

## Leitura

1. O que o fine-tune acrescenta ao mmBERT é a adaptação do encoder. Com só a saída treinada,
   ele vai longe no estilo da organização (0,902 no estresse) e desaba no texto real (0,243
   contra 0,605 do v1.2 cru, mesma receita).
2. Com a guarda, nenhum concorrente treinado nos mesmos dados passou o v1.3. Sozinhos,
   BERTimbau e GLiNER treinados passam o Gama com folga. A vantagem do sistema entregue não
   é o mmBERT ser o melhor extrator cru, porque ele não é. É a confiança dele separar erro de
   acerto, o que deixa a régua entrar onde precisa. O GLiNER treinado não tem esse sinal.
3. A receita do v1.2 varia muito entre sementes fora do molde, de 0,760 a 0,830 com a guarda
   nas 305. O +0,019 do v1.3 sobre o v1.2 nas 172 é do par publicado e não passa dessa
   variação. Na nossa leitura, o que dá para afirmar é mais estreito: o aluno fica acima do
   próprio professor nas 172 nas três sementes da destilação, numa análise feita depois.
4. Número com guarda de um extrator fraco mede a régua. A cabeça sem treino, com a guarda a
   0,99, chega a 0,600 nas 305. Toda comparação de extratores precisa da coluna crua ao lado.
5. O limite do Qwen3-8B no benchmark é mais de interface que de leitura: cerca de 58% da
   amostra é citação que ele reconheceu e o alinhamento perdeu. Um quarto dos 583 trechos,
   porém, são exemplos do prompt copiados, e isso é alucinação mesmo.
6. No estilo da organização nada disso muda a nota. Seis sementes, BERTimbau e GLiNER
   treinado ficam no teto do estresse ou encostados nele, e a submissão continua a mesma.

## Limites gerais

- O gabarito do texto real é de LLM adjudicado. Um teste com gabarito humano, num conjunto
  que não participou de nenhuma escolha, segue em aberto. As 172 não informaram nenhuma
  escolha nestes controles, mas antes tinham ajudado a escolher o v1.3.
- Encoder congelado, GLiNER treinado e BERTimbau têm uma semente cada. Os IC95 medem a
  amostra de ementas, não a variação de treino.
- O estresse com a guarda satura: todos os treinados completos ficam em 1,09999 ou perto, e
  ele não discrimina entre eles.
- O estresse e o `final_v3` usam os mesmos 600 identificadores (`syn_n1_*`, `syn_n2_*`), mas
  são documentos diferentes: nenhum texto se repete, e a sobreposição média das citações sob
  o mesmo identificador é de 0,1%. Das 3.951 citações distintas do estresse, 1.175 aparecem
  em algum documento do `final_v3`, porque os dois saem dos mesmos bancos do acervo. Isso
  vale igual para todo candidato treinado no `final_v3`.
- Na métrica oficial, os modelos que não são o Gama entram com a calibração medida para o
  Gama.
