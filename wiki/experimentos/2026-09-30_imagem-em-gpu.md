# Imagem de avaliação em GPU

Até 30/09/2026 a imagem Docker da solução só tinha rodado em CPU. Os ensaios na L4 usavam
os scripts do HF Jobs, numa imagem de outra origem. Este teste roda a imagem do nosso
`Dockerfile` numa GPU de 24 GB, a do envelope do desafio, e achou um defeito que só aparece
ali.

## O teste

A imagem foi construída a partir do `Dockerfile` por um Space Docker privado e temporário, e
um job do HF Jobs numa NVIDIA L4 rodou dentro dela o extrator de produção (o modelo com a
guarda) sobre os 905 documentos do benchmark: os 600 do estresse difícil e as 305 ementas
reais. Duas comparações:

- contra a mesma execução em CPU, feita na máquina local: um hash dos trechos (início, fim,
  tipo, forma e dígitos de cada um) e outro das confianças com 4 casas;
- GPU contra CPU dentro da própria imagem, em 101 documentos (1 a cada 9).

O acervo é dado da organização e não sai da máquina, então o job mede a extração, que é a
parte que a GPU muda. Resolver e classificação são código de CPU e rodam iguais nos dois
casos.

## Primeira rodada: a imagem quebra em GPU

Com a GPU visível, a execução parou no primeiro documento:

```
RuntimeError: Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl.
```

Em GPU, o torch 2.14 com CUDA passa uma operação do mmBERT (o produto das posições na
codificação rotativa) por um kernel do Triton, e o Triton compila um módulo C na primeira
chamada. A `python:3.12-slim-bookworm` não traz compilador. Em CPU esse caminho não é usado,
e por isso todo teste local passava. Os ensaios anteriores na L4 também passavam, porque a
imagem dos scripts do HF Jobs tem gcc.

Num avaliador com GPU e runtime NVIDIA, o `run.sh` teria falhado.

## Correção e segunda rodada

`gcc` e `libc6-dev` entram na imagem, antes do torch. Com isso:

| | GPU, NVIDIA L4 | CPU, máquina local |
|---|---|---|
| torch | 2.14.0+cu126 | 2.14.0+cpu |
| Documentos / trechos | 905 / 5.582 | 905 / 5.582 |
| Hash dos trechos | `d3377daa…02da34` | `d3377daa…02da34` |
| Hash das confianças, 4 casas | `f968c494…401530` | `f968c494…401530` |
| Tempo por documento, estresse / ementas | 0,046 s / 0,016 s | 1,49 s / 0,47 s |
| Pico de VRAM | 1.108 MiB | |

Dentro da imagem, GPU e CPU dão os mesmos trechos nos 101 documentos da amostra, e a maior
diferença de confiança é 3,1e-07.

## O que o teste não cobre

- O `docker run --gpus all` e o `run.sh` em si. O job entra na imagem pelo HF Jobs, não pelo
  Docker de uma máquina com o runtime NVIDIA. O `run.sh` inteiro foi testado em CPU, num
  clone limpo.
- O resolver com GPU ligada. Ele não usa a GPU, mas não rodou neste job, porque o acervo não
  sai da máquina.
