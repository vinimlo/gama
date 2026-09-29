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
| [`urchade/gliner_multi-v2.1`](https://huggingface.co/urchade/gliner_multi-v2.1) (GLiNER) | Apache-2.0 | comparação zero-shot no [benchmark](../experimentos/2026-09-23_bench-extratores-crus.md) | 0,327 no estresse, 0,409 em texto real; não reconhece referência vaga (0 de 714) |
| Qwen3-8B | Apache-2.0 | comparação zero-shot no benchmark | 0,811 no estresse, 0,707 em texto real; 11,7 s por documento na L4 |

O empate entre mmBERT e BERTimbau está em
[validação por dobras](../experimentos/2026-09-22_validacao-por-dobras.md) e
[estresse difícil](../experimentos/2026-09-22_estresse-dificil.md). O critério de
desempate foi prático: o mmBERT lê o documento inteiro de uma vez.

## Comparações que não fizemos

O mmBERT original sem fine-tune não entra em comparação nenhuma. Ele foi pré-treinado para
prever palavra mascarada e não tem cabeça de rótulos BIO; pontuá-lo exigiria uma cabeça
nova com pesos aleatórios, o que mede ruído. O Gama sem a guarda (`--extrator neural-cru`)
é o mmBERT fine-tunado e aparece no benchmark como "Gama sozinho". O controle que isolaria
o ganho do fine-tune sobre a base, encoder congelado com só a cabeça treinada, não foi
rodado.

O GLiNER não foi fine-tunado. No estilo da organização dois encoders treinados já
empatavam no teto, e um terceiro não tinha o que ganhar ali; ele ainda traria a biblioteca
`gliner` para a organização reproduzir offline. Em texto real a pergunta continua aberta:
ali o Gama sozinho marca 0,620 e com a guarda 0,818, e um GLiNER treinado nos mesmos dados
nunca foi medido.

## Jev Decision Index

`huggingface.co/spaces/multimodalart/jev-decision-index` é um rastreador comunitário de
reproduções abertas do Jev, com 19 benchmarks gerais em inglês. Nenhum mede localizar span
em texto jurídico em português, então não serve como fonte de modelo. Serve de referência
para uma pergunta previsível: por que não um modelo de decisão? Porque saber se a citação
existe é consulta ao acervo, não inferência.
