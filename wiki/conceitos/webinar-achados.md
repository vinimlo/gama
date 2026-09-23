---
slug: webinar-achados
tipo: conceito
status: ativo
data: 2026-09-22
fonte: gravação do webinar de 28/08/2026, compartilhada pela organização
one_liner: "O que o webinar da organização (28/08) acrescenta ao que está escrito no site e no Kaggle"
---

# Achados do webinar (28/08/2026)

Destilado da gravação de 58 min. A transcrição automática erra nomes próprios, então
cada item foi conferido no trecho original (timestamp indicado).

## 1. Gerar dados sintéticos é explicitamente permitido (o achado que mais vale)

> *"Nada impede aqui que vocês criem um golden set de vocês, que vocês criem um
> dataset de vocês ao longo do desenvolvimento. (…) A gente tem a nossa base
> canônica congelada, que são peças públicas, decisões públicas. Então, se vocês
> quiserem baixar outras decisões, criar um outro dataset, não tem problema. E os
> documentos de entrada (…) foram gerados com IA. Então, se vocês quiserem fazer
> um trabalho parecido para aumentar o dataset de vocês durante o
> desenvolvimento, não tem problema, não é proibido."* (19:03 a 19:40)

E descrevem a receita: *"Foi um caminho bem simples mesmo. É ter esse dataset de
decisões públicas e a partir dele gerar os documentos sintéticos, como se fosse
uma IA escrevendo uma peça."* (19:42)

Isso importa porque destrava as duas limitações do
[baseline](../experimentos/2026-09-22_baseline-deterministico.md):

- validação honesta: com documentos gerados por nós dá para medir generalização
  de verdade, em vez de reportar um número ajustado às 192 citações do dev set;
- dado de treino para um extrator neural: 192 citações é pouco para fine-tune; o
  [gerador por moldes](../decisoes/D-004_gerador-por-moldes.md) produz milhares.

## 2. Qualquer motor de busca/banco é permitido

> Pergunta: *"Seja o FTS5 com SQLite, mas a gente consegue usar porventura outro SGBD ou
> outro mecanismo de busca, de indexação?"*
>
> Resposta: *"Consegue, consegue. É só garantir que a gente vai conseguir rodar a mesma
> coisa aqui do lado de cá. Se você conseguir containerizar o negócio e deixar
> fechadinho com versões, não dificulta a reprodutibilidade."* (35:15 a 35:53)

Nosso índice em memória construído do SQLite está coberto.

## 3. Reprodutibilidade não é burocracia: eles re-executam

> *"Essa questão da reprodutibilidade nem é tanto burocrática, porque de fato a
> gente vai rankear vocês em cima dessa execução e desse conjunto que vocês não
> conhecem."* (31:43)

> *"Essas soluções têm que generalizar o suficiente para também obter bom
> score aqui no conjunto que a gente tem fechado do nosso lado."* (31:10)

## 4. Falso positivo custa, e acerto exige classe e link

> *"Se você extrair a mais vai ser penalizado na precisão. Se não teve match
> nenhum, vai ser um falso positivo. É bom ter cuidado para não sair extraindo
> tudo e acabar extraindo muita coisa que seja lixo."* (36:18)

> *"Só vai ser verdadeiro positivo de fato se houve a linkagem correta e se a
> classe estava correta. Se você extraiu a citação corretinha e no final não
> conseguiu fazer o link com o documento, vai ser descartado como erro."* (36:48)

Bate exatamente com o `vendor/kaggle_metric.py`.

## 5. A limpeza do dataset nasceu daqui

Um participante levanta, aos 37:17, que documentos citam trechos vagos que às
vezes estão no gabarito e às vezes não: `"cumpre observar que a orientação dos
tribunais superiores é firme no ponto"` (fora) contra `"normas de regência da
matéria"` (dentro). A organização responde:

> *"A gente encarou isso como sendo algo muito vago, mas o que é vago não é bem
> definido. E aí pode ficar confuso em penalizar ali no final, no score. É um
> caso que a gente pode avaliar com mais calma e retornar com vocês."* (40:40)

E foi o que fizeram: a atualização final removeu 33 citações `incompleta`
"que não apontavam para uma fonte específica". É por isso que a classe ficou
trivialmente separável: as que sobraram têm todas tribunal + ano + relator.
O achado que tínhamos medido nos dados tem agora a explicação de origem.
