# 2026-09-23 · Benchmark: Gama contra extratores sem treino

Pergunta: quanto o fine-tune rende contra o que dava para conseguir sem treinar nada,
dentro das regras do desafio (pesos abertos, revisão fixa, 1 GPU de 24 GB)?

## Desenho

| Extrator | O que é |
|---|---|
| Régua | regex escrita à mão, sem modelo (o baseline de 22/09) |
| GLiNER | [`urchade/gliner_multi-v2.1`](https://huggingface.co/urchade/gliner_multi-v2.1)@`443d26d` (289M, Apache-2.0), NER zero-shot; só os nomes dos rótulos em português, limiar 0,5, janelas de 200 palavras |
| Qwen3-8B | [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B)@`b968826` (Apache-2.0), LLM zero-shot; prompt com os três tipos, as convenções de borda e exemplos de formato genéricos; decodificação gulosa, sem modo de raciocínio; cada trecho devolvido volta ao texto por casamento exato, com espaço em branco flexível |
| Gama v1.2 | `vinimlo/gama@ad06ffd`, mmBERT-base fine-tunado só com documentos sintéticos |

- Uma tentativa por extrator. Prompt e rótulos foram escritos uma vez, sem ajuste olhando
  os conjuntos avaliados.
- Todos passam pelo mesmo caminho de produção: `aparar`, resolver no acervo, classe e
  confiança. GLiNER e Qwen entram sem confiança de modelo, com os mesmos priores da régua.
- Métrica oficial onde há `id_canonico` (dev e estresse); F1 de extração (mesmo tipo,
  IoU ≥ 0,5, casamento 1 para 1) nos três conjuntos. No texto real o ouro não separa
  referência vaga, então VAGA conta como JURIS para todos.
- O dev é da organização e não sai da máquina: nele só rodam régua e Gama, localmente.
- Estresse e texto real foram extraídos no HF Jobs, numa NVIDIA L4. O Qwen rodou na A100
  com lote 32 para caber no tempo; o tempo dele por documento foi medido na L4, com lote 4.
- Código em `bench/`; números em `bench/resultados.json`; spans no dataset privado
  `vinimlo/gama-goldenset`, pasta `bench/saida/`.

## Resultado

Números com o resolver de 23/09 de manhã. Depois do desempate por cadeia compatível
([D-008](../decisoes/D-008_guarda-do-extrator.md)), o estresse passou a 1,09999 (Gama),
0,84880 (régua) e 0,81065 (Qwen3-8B); o GLiNER não mudou. `bench/resultados.json` e o
gráfico do README trazem os números atuais.

| Conjunto | Gama v1.2 | Régua | Qwen3-8B | GLiNER |
|---|---|---|---|---|
| Dev, métrica oficial (26 docs) | 1,10000 | 1,09848 | não medido | não medido |
| Estresse difícil, métrica oficial (600 docs) | 1,09974 | 0,84865 | 0,81051 | 0,32732 |
| Estresse difícil, F1 de extração | 1,000 | 0,858 | 0,825 | 0,453 |
| Texto real, F1 de extração (305 ementas) | 0,605 | 0,663 | 0,707 | 0,409 |
| Tempo por documento, L4 | 0,057 s | 0,001 s | 11,7 s | 0,113 s |

F1 de extração por tipo:

| | Gama | Régua | Qwen3-8B | GLiNER |
|---|---|---|---|---|
| Estresse, JURIS | 1,000 | 0,893 | 0,875 | 0,477 |
| Estresse, LEI | 1,000 | 0,784 | 0,818 | 0,569 |
| Estresse, VAGA | 1,000 | 0,763 | 0,547 | 0,000 |
| Texto real, JURIS | 0,475 | 0,707 | 0,753 | 0,578 |
| Texto real, LEI | 0,747 | 0,620 | 0,663 | 0,299 |

## Leitura

1. No estilo da organização, o fine-tune é a diferença. No estresse o Gama marca 1,0997;
   o melhor extrator sem treino, o Qwen3-8B, fica em 0,811, abaixo até da régua (0,849).
   O Qwen perde mais na referência vaga (F1 0,547; 425 das 714 não achadas) e no ruído: dos
   5.515 trechos que ele devolveu, 583 não se alinham ao texto, 283 deles em N2 contra 57
   em N1: 526 não aparecem literalmente, 56 existem mas caem na regra de ocorrência e 1 vem
   em formato inválido. A auditoria dos [controles](2026-09-29_controles.md) achou que 149
   dos 583 são cópias dos exemplos do prompt e que, numa amostra de 120, 60% são citações
   reescritas: ele reconhece a citação e devolve outra grafia, e aí o trecho não se alinha. Tudo isso a 11,7 s por documento na L4, cerca de
   200 vezes o tempo do Gama.
2. Em texto real a ordem se inverte: o Qwen3-8B é o melhor (0,707), à frente da régua
   (0,663) e do Gama (0,605). Um LLM genérico lê formatos que nunca viu; o Gama foi
   treinado para um formato só.
3. O GLiNER não reconhece referência vaga: 0 das 714 do estresse. E parte o dispositivo
   de lei em dois ("art. 5º" de um lado, "da Constituição Federal" do outro), o que gera
   652 falsos positivos de LEI. Os nomes dos rótulos não bastam para ensinar a convenção
   de borda.
4. A régua se sai bem onde foi escrita (dev) e cai no estresse (0,849), principalmente em
   N2: ruído de OCR na palavra-chave quebra o regex.
5. Em texto real a régua também ganha do Gama (0,663 contra 0,605). O v1.2 tem recall de
   precedente de 0,76 contra o ouro adjudicado (a [medida de 22/09](2026-09-22_texto-real.md),
   com o v1 e contra a prata, dava 0,49), mas fora do molde marca fragmentos soltos ("Rel",
   "2011", "DJe 19/08/2019"): 710 falsos positivos de JURIS. É o preço da especialização no estilo da organização, que é o
   formato anunciado para o cego; a [D-007](../decisoes/D-007_sem-ensemble.md) mantém a
   união com a régua como opção para texto fora do molde. A
   [otimização medida](2026-09-23_otimizacao-medida.md) mostra que a confiança do próprio
   modelo separa esses fragmentos: filtrado, o Gama vai a 0,808 no texto real, sem mudar
   nada no estilo da organização.

## Gama v1.3

O benchmark foi refeito com o v1.3 (`vinimlo/gama@5f924ca`, [D-009](../decisoes/D-009_gama-v1-3-destilado.md)),
no mesmo harness: job L4 `6ab48b19` com `bench/extrair_gama.py`, spans em
`bench/saida/gama.jsonl` do dataset (revisão `684563c`), dev em CPU local. A régua foi
extraída de novo no mesmo job e saiu idêntica (0 spans diferentes em 905 documentos); Qwen3-8B
e GLiNER não mudam.

| | Gama v1.2 | Gama v1.3 |
|---|---|---|
| Estresse difícil, métrica oficial, sozinho e com a guarda | 1,09999 | 1,09999 |
| Estresse difícil, spans crus diferentes do v1.2 (600 docs) | | 0 |
| Texto real, F1 de extração, sozinho (305 ementas) | 0,605 | 0,620 |
| Texto real, F1 de extração, com a guarda | 0,808 | 0,818 |
| Texto real, falsos positivos de JURIS, sozinho | 710 | 662 |
| Tempo por documento, L4 (905 docs), sozinho / com a guarda | 0,057 s / 0,058 s | 0,035 s / 0,037 s |
| Tempo por documento, CPU no dev | 1,58 s | 0,95 s |

Os spans crus do v1.3 diferem dos do v1.2 em 154 das 305 ementas reais, fora do molde, e
em nenhum documento do estresse. O quadro geral não muda: sozinho, o Gama continua atrás da
régua e do Qwen3-8B em texto real (0,620 contra 0,663 e 0,707); com a guarda, na frente.
O Qwen3-8B custa agora cerca de 330 vezes o tempo do Gama por documento.
