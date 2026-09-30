# Regras e reprodutibilidade

O que as regras exigem e como a solução cumpre cada ponto. Duas fontes: o regulamento,
relido em 23/09/2026 no site do desafio, no FAQ, nas páginas da competição no Kaggle
(incluindo as regras-base do Kaggle, que prevalecem em conflito) e no
[webinar](webinar-achados.md); e as regras de envio da solução final, que a organização
mandou às equipes em 30/09/2026.

## O que é reproduzido

A organização executa o código de cada equipe sobre um `.db` novo e um conjunto novo de
documentos, no mesmo formato da amostra de desenvolvimento, e a nota oficial vem dessa
execução. Não há comparação entre CSVs, e o leaderboard do Kaggle não entra no ranking
final. O objeto da reprodução é a inferência: "o critério é a reprodutibilidade da
inferência, não onde você rodou" (FAQ). O envelope vale para a execução da solução: no
regulamento, 1 GPU de 24 GB, cerca de 8 vCPUs e 32 GB de RAM; nas regras de envio, uma GPU
de até 24 GB de VRAM, sem internet e sem API externa. O que não cabe nele é tratado como
não reproduzível.

Treino e preparo de dados não são re-executados. Para fine-tune, a exigência é publicar os
pesos resultantes com link e revisão fixa, disponíveis para download antes da execução.
Modelos usados só para preparar dados de treino, como os que ampliaram frases e anotaram
ementas aqui, podem ficar fora do limite de hardware, desde que não participem da execução.

## Envio da solução final

Prazo: 01/10/2026, 23h59, horário de Brasília. Vai por e-mail à organização o nome da
equipe e dos integrantes, o link do repositório e o hash do commit da versão final. O
repositório precisa trazer:

| Exigência | Onde está |
|---|---|
| Código completo da solução | `src/gama`, `vendor/` e `run.sh` |
| README com a abordagem e o passo a passo de execução | [README](../../README.md), seções "Para quem vai avaliar" e "Como funciona" |
| Ambiente declarado em Docker | `Dockerfile` (Python 3.12, torch 2.14.0 com CUDA 12.6) e `requirements.txt` |
| Pesos incluídos ou referenciados em revisão fixa, baixáveis antes da execução | `vinimlo/gama`, revisão no `MODELO.md`; `bash run.sh --preparar` baixa |
| Ponto de entrada único, que recebe o `.db` e a pasta de `.txt` e gera a saída no formato das submissões | `bash run.sh <caminho_db> <pasta_txt> <arquivo_saida.csv>` |

E a execução precisa respeitar:

| Regra | Como a solução cumpre |
|---|---|
| GPU de até 24 GB de VRAM | pesos de 1,03 GB em FP32; com a imagem numa L4 de 24 GB, 0,046 s por documento e pico de 1.108 MiB de VRAM ([imagem em GPU](../experimentos/2026-09-30_imagem-em-gpu.md)); 0,93 s por documento só em CPU |
| Sem internet nem API externa | o container roda com `--network none`; a imagem tem `HF_HUB_OFFLINE=1` |
| Do zero, em máquina limpa, sem caminho absoluto nem passo manual | o `run.sh` resolve os caminhos a partir de onde está e prepara o que faltar; testado num clone limpo, com a base em outro caminho |
| Pré-processamento do `.db` a partir do formato original | não há artefato pré-calculado: o índice canônico é montado do `.db` recebido a cada execução (`src/gama/indice.py`) |
| Seeds fixas e sem amostragem, para resultado estável | `eval()`, algoritmos determinísticos do torch, `PYTHONHASHSEED=0`; GPU e CPU dão a mesma saída |
| Disco com bom senso | imagem de cerca de 4 GB e 1,03 GB de pesos |

## O regulamento, ponto a ponto

| Exigência | Onde está |
|---|---|
| Só pesos e código abertos, sem chave de API nem serviço pago na execução | mmBERT-base (MIT), pesos do Gama no Hub com revisão fixa (`MODELO.md`); dependências com licença aberta; o container roda sem rede |
| Pesos de fine-tune publicados, link e revisão | `vinimlo/gama`, revisão no `MODELO.md` |
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
- O caminho inteiro, com o comando de cada passo, está em
  [reprodução passo a passo](../guias/reproducao.md).
- Modelos avaliados e descartados estão em [modelos elegíveis](modelos-elegiveis.md).
