# Modelos elegíveis

## A regra

Só pesos abertos, baixáveis de repositório público, com revisão fixa, executáveis
offline pela organização dentro do envelope (1 GPU 24 GB, 8 vCPUs, 32 GB RAM, média
≤ 60 s/documento). API paga ou proprietária desclassifica.

## Desclassificados

- GPT, Claude, Gemini: API.
- Jev (TypeSafe AI): API proprietária em early access, sem pesos públicos.
- Sabiá-2 e Sabiá-3 (Maritaca): API paga. Só o `sabia-7b` tem pesos, e é LLaMA-1 de
  2023 com licença de pesquisa.

Destilar o Sabiá-3 não resolveria o problema central: o que separa `real` de `inventada`
está no SQLite do acervo, não em peso de modelo. Destilar ensinaria o aluno a chutar com
fluência, que é a alucinação que o desafio existe para pegar.

## Avaliados

| Modelo | Licença | Papel | Resultado |
|---|---|---|---|
| [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base) | MIT | extrator (escolhido), fine-tune BIO | contexto de 8 mil tokens: documento numa passada, 0,077 s/doc na L4 |
| [`neuralmind/bert-base-portuguese-cased`](https://huggingface.co/neuralmind/bert-base-portuguese-cased) (BERTimbau) | MIT | extrator alternativo, mesmo protocolo | empata com o mmBERT nas dobras e no estresse; 512 tokens exigem janelas |
| DeepSeek-V4-Pro, Kimi-K3, GLM-5.3 (abertos) | licenças verificadas | só desenvolvimento: expandir bancos de frases, anotar texto real | grandes demais para o envelope; nunca em tempo de execução |
| [`fastino/gliner2.5-multi-v1`](https://huggingface.co/fastino/gliner2.5-multi-v1) (GLiNER 2.5) | Apache-2.0 | comparação zero-shot no benchmark e fine-tune nos mesmos dados ([GLiNER 2.5](../experimentos/2026-09-29_gliner-2-5.md)) | zero-shot: 0,444 no estresse, 0,608 em texto real; treinado: 1,09615 no estresse, 0,808 em texto real sem guarda |
| [`urchade/gliner_multi-v2.1`](https://huggingface.co/urchade/gliner_multi-v2.1) (GLiNER v2.1, substituído pela 2.5) | Apache-2.0 | comparação zero-shot no [benchmark](../experimentos/2026-09-23_bench-extratores-crus.md) | 0,327 no estresse, 0,409 em texto real; não reconhece referência vaga (0 de 714) |
| Qwen3-8B | Apache-2.0 | comparação zero-shot no benchmark | 0,811 no estresse, 0,707 em texto real; 11,7 s por documento na L4 |

O empate entre mmBERT e BERTimbau está em
[validação por dobras](../experimentos/2026-09-22_validacao-por-dobras.md) e
[estresse difícil](../experimentos/2026-09-22_estresse-dificil.md). O critério de
desempate foi prático: o mmBERT lê o documento inteiro de uma vez.

## Controles com treino

A primeira rodada comparou o Gama só com extratores sem treino. Os
[controles de 29/09](../experimentos/2026-09-29_controles.md) treinaram concorrentes nos
mesmos 5.410 documentos sintéticos do v1.2 e mediram todos pelo mesmo harness, sozinhos e com
a guarda. F1 de extração em texto real, 305 ementas de validação e 172 de confirmação:

| Modelo | Sozinho, 305 / 172 | Com a guarda (0,95), 305 / 172 |
|---|---|---|
| Gama v1.3 | 0,620 / 0,635 | 0,818 / 0,839 |
| BERTimbau, receita do v1.2 com `max_len` 512 | 0,697 / 0,705 | 0,813 / 0,826 |
| GLiNER 2.5 multi treinado | 0,808 / 0,824 | 0,816 / 0,837 |
| GLiNER multi v2.1 treinado | 0,792 / 0,752 | 0,796 / 0,761 |
| mmBERT com o encoder congelado, só a saída treinada | 0,243 / 0,247 | 0,350 / 0,353 |

O mmBERT original sem cabeça de rótulos também foi pontuado, só como diagnóstico, com uma
cabeça nova sem treino. Deu F1 de 0,0005. Mede o sorteio dos pesos. O controle útil é
o encoder congelado com a saída treinada, e ele responde quanto o fine-tune acrescenta à base.
No estresse ele chega a 0,902, acima da régua e do Qwen3-8B; em texto real cai para 0,243,
contra 0,605 do v1.2 cru treinado pela mesma receita. Rende a adaptação do encoder.

Treinado, o GLiNER encosta no teto do estilo da organização (v2.1: 1,09631, 6 erros em 4.447
spans; 2.5: 1,09615, 10 erros) e passa o Gama sozinho em texto real. A v2.1 com a guarda não
se distingue do v1.3 nas 305 e perde nas 172. A 2.5 não se distingue do v1.3 com a guarda
mesmo sem guarda nenhuma, nos dois conjuntos. Nas duas versões a guarda quase não ajuda: o
score fica acima de 0,99 na maior parte dos erros, e sem hesitação a régua não tem por onde
entrar. Detalhes em [GLiNER 2.5](../experimentos/2026-09-29_gliner-2-5.md).

O BERTimbau empata no teto de novo e, sozinho, passa o Gama em texto real pela precisão. Com a
guarda, não dá para distingui-lo do v1.2 nem do v1.3. Na L4 ele roda em 0,066 s por
documento no estresse, mais rápido que o v1.2 (0,075 s) e mais lento que o v1.3 (0,046 s). O
desempate prático de antes, ler o documento inteiro de uma vez, só vira velocidade depois da
destilação.

Os três concorrentes têm uma semente cada, e o gabarito do texto real é de LLM adjudicado.
Nenhuma comparação foi feita com gabarito humano. Na nossa leitura, a escolha do mmBERT se
sustenta pelo que a confiança dele permite à guarda. Sozinho, ele não é o melhor extrator.

## Jev Decision Index

`huggingface.co/spaces/multimodalart/jev-decision-index` é um rastreador comunitário de
reproduções abertas do Jev, com 19 benchmarks gerais em inglês. Nenhum mede localizar span
em texto jurídico em português, então não serve como fonte de modelo. Serve de referência
para uma pergunta previsível: por que não um modelo de decisão? Porque saber se a citação
existe é consulta ao acervo, não inferência.
