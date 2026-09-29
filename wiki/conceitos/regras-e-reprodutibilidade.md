# Regras e reprodutibilidade

O que as regras exigem, relidas em 23/09/2026 no site do desafio, no FAQ, nas páginas da
competição no Kaggle (incluindo as regras-base do Kaggle, que prevalecem em conflito) e no
[webinar](webinar-achados.md), e como a solução cumpre cada ponto.

## O que é reproduzido

A organização re-executa a solução sobre o conjunto cego e ranqueia por essa execução. O
objeto da reprodução é a inferência: "o comando exato que reproduz as saídas submetidas" e
"o critério é a reprodutibilidade da inferência, não onde você rodou" (FAQ). O envelope de
execução vale para "a solução completa", o pipeline que gera as saídas: 1 GPU de 24 GB,
cerca de 8 vCPUs e 32 GB de RAM. O que não cabe nele é tratado como não reproduzível e
desclassificado.

Treino e preparo de dados não são re-executados. Para fine-tune, a exigência é publicar os
pesos resultantes com link e revisão fixa, "de modo que a organização consiga carregá-los e
executá-los".

## Exigências e como a solução cumpre

| Exigência | Onde está |
|---|---|
| Só pesos e código abertos, sem chave de API nem serviço pago na execução | mmBERT-base (MIT), pesos do Gama no Hub com revisão fixa (`MODELO.md`); dependências com licença aberta; o container roda sem rede |
| Pesos de fine-tune publicados, link e revisão | `vinimlo/gama`, revisão no `MODELO.md` (repositório público na entrega) |
| Envelope de 1 GPU de 24 GB, ~8 vCPUs, 32 GB | 0,07 s por documento numa L4 de 24 GB; 1,6 a 3 s por documento só em CPU; pesos de 1,2 GB |
| Bundle: repositório, README, modelos, ambiente, comando exato | este repositório, `Dockerfile`, `requirements.txt`, `make predizer` e o comando no README |
| Decodificação determinística | `eval()`, sem amostragem, algoritmos determinísticos do torch; duas execuções na L4 deram os mesmos spans byte a byte |
| Qualquer dataset público no treino | documentos sintéticos gerados aqui e, na destilação do v1.3, 1.818 ementas sem rótulo de [`celsowm/jurisprudencias_br`](https://huggingface.co/datasets/celsowm/jurisprudencias_br) (público, CC-BY-4.0); as ementas anotadas ficam só na avaliação. Ver [ementas reais](ementas-reais.md) |
| Nada de extrair ou inferir o conjunto de teste privado | nenhum dado do cego entra no desenvolvimento |
| Nada de rotulagem manual ou predição humana sobre dados de teste (regras-base do Kaggle, 4.b) | a submissão roda com a configuração congelada; nenhuma escolha de modo é feita depois de ver o cego |

## Desenvolvimento

- LLMs de pesos abertos (DeepSeek-V4-Pro, Kimi-K3, GLM-5.3) foram usados só no
  desenvolvimento, servidos por um provedor hospedado, para ampliar bancos de frases e
  anotar ementas reais. O FAQ permite servir modelo de pesos abertos por API durante o
  desenvolvimento. Nenhum deles participa da inferência, e o que produziram está congelado
  no dataset do treino, com revisão fixa: refazer os pesos não exige chamar LLM nenhum.
- O treino roda em GPU alugada (HF Jobs). A mesma receita (mmBERT-base, 3 épocas,
  `max_len` 1024) treinou os modelos da validação por dobras numa A10G de 24 GB, a mesma
  classe de máquina do envelope; a A100 do v1.2 só encurtou o tempo.
- Modelos avaliados e descartados estão em [modelos elegíveis](modelos-elegiveis.md).
