# Gama

Verificador de citações jurídicas do Desafio Caça-Alucinações (BRACIS 2026 × Jusbrasil).

O Gama lê um documento judicial em `.txt`, encontra cada citação de jurisprudência e de
lei e diz se ela é `real`, `inventada` ou `incompleta`. As reais saem ligadas ao `doc_id`
do registro correspondente num acervo congelado de 1.014 decisões. É o problema de
checar se um texto gerado por IA inventou precedente, e o motivo de o desafio existir.

O nome homenageia Luiz Gama (1830–1882), advogado abolicionista que libertou centenas de
pessoas nos tribunais citando a lei com precisão.

[Competição no Kaggle](https://www.kaggle.com/competitions/desafio-jusbrasil-bracis-2026) ·
[pesos do extrator](https://huggingface.co/vinimlo/gama) ·
[manifesto do modelo](MODELO.md) · [base de conhecimento](wiki/INDEX.md)

## Resultados

Tudo medido com a métrica oficial (`vendor/kaggle_metric.py`, cópia exata da que roda no
servidor): `macroF1 · (1 − 0,5·τ) · (1 + 0,10·(1 − Brier))`, com o nível 2 pesando o dobro.
O teto prático é 1,10000.

| Conjunto | Gama v1.2 | Régua (regex, sem modelo) |
|---|---|---|
| Dev da organização, 26 documentos, 192 citações | 1,10000 | 1,09848 |
| Kaggle, fase de treino (o mesmo dev, pontuado no servidor) | 1,09999 | 1,09830 |

O dev set sozinho não diz muita coisa: a própria régua, ajustada a ele, chega a 1,098, e
qualquer candidato bom tira F1 = 1,0 ali. Na [validação por dobras](wiki/experimentos/2026-09-22_validacao-por-dobras.md),
um modelo treinado só com dados sintéticos acerta as 192 citações de documentos da
organização que nunca viu. O que separa os candidatos é o que vem a seguir.

### Contra extratores sem treino

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="bench/grafico-escuro.svg">
  <img alt="Estresse difícil, métrica oficial: Gama 1,0997, régua 0,8487, Qwen3-8B 0,8105, GLiNER 0,3273. Texto real, F1 de extração: Gama 0,605, régua 0,663, Qwen3-8B 0,707, GLiNER 0,409." src="bench/grafico-claro.svg">
</picture>

Para medir o que o fine-tune rende, comparamos o Gama com o que dava para conseguir sem
treinar nada dentro das regras do desafio: a régua, o GLiNER multi v2.1 e o Qwen3-8B, os
dois últimos com pesos abertos e em zero-shot. Os quatro passam pelo mesmo resolver e pela
mesma métrica.

| | Gama v1.2 | Régua | Qwen3-8B | GLiNER |
|---|---|---|---|---|
| Estresse difícil, métrica oficial (600 documentos) | 1,0997 | 0,8487 | 0,8105 | 0,3273 |
| Texto real, F1 de extração (305 ementas) | 0,605 | 0,663 | 0,707 | 0,409 |
| Tempo por documento numa NVIDIA L4 | 0,057 s | 0,001 s | 11,7 s | 0,113 s |

O [estresse difícil](wiki/experimentos/2026-09-22_estresse-dificil.md) segue o estilo da
organização, que é o formato anunciado para o conjunto cego, com frases escritas por um LLM
que nenhum modelo viu no treino e ruído de OCR forte. Ali o fine-tune é a diferença: o
melhor extrator sem treino fica abaixo até da régua. Em texto real (ementas do STF, STJ e
TJRJ) a ordem se inverte e o Gama mostra o próprio limite: especializou no estilo da
organização e, fora dele, marca fragmentos soltos. Por isso a união com a régua fica
disponível como opção ([D-007](wiki/decisoes/D-007_sem-ensemble.md)). Desenho, números
por tipo e leitura completa no [benchmark](wiki/experimentos/2026-09-23_bench-extratores-crus.md).

## Como funciona

```
documento .txt
  │
  ├─ extrator Gama      mmBERT-base fine-tunado, classificação de tokens BIO (JURIS, LEI, VAGA);
  │                     lê o documento inteiro numa passada. Sem pesos montados, cai para a régua.
  │
  ├─ normalização       desfaz o OCR letra→dígito (l→1, S→5, O→0, G→6, g→9), só no núcleo numérico
  │
  ├─ índice canônico    construído uma vez a partir do SQLite: cabeçalho + autorreferência do feito
  │
  ├─ resolver           lookup O(1) → candidatos; desempate pela cadeia de classe processual
  │
  └─ classificação      1 candidato → real · 0 → inventada · referência vaga → incompleta
                        confiança = probabilidade de acerto medida para aquele tipo de caso
```

A ideia central é dividir o trabalho. O modelo só aprende onde a citação começa e termina
e de que tipo ela é. Quem decide se ela existe é o acervo, por consulta exata. Um modelo que
decidisse `real` ou `inventada` por conta própria estaria chutando com fluência, que é
justamente a alucinação que o desafio quer pegar.

### Treino só com dados sintéticos

A organização gera os documentos do desafio por moldes: 62% das frases se repetem entre
documentos, cada documento usa uma única data e a quebra de linha segue uma regra que
conseguimos reproduzir byte a byte. Então desmontamos os 26 documentos do dev set em
bancos de peças (preâmbulo, narrativa, enchimento, frases que carregam citação) e
remontamos milhares de documentos novos citando fichas reais do acervo. O rótulo sai
exato por construção, porque sabemos onde pusemos cada citação.

LLMs de pesos abertos (DeepSeek-V4-Pro, Kimi-K3, GLM-5.3) só expandem os bancos de frases.
Nunca escrevem a citação nem decidem rótulo, e nenhum deles roda na solução avaliada. Um
injetor de ruído calibrado no nível 2 da organização completa o conjunto. O Gama v1.2 foi
treinado em 6.000 documentos, 3 épocas, semente 13.

O gerador também serviu de teste para o resolver: cada citação renderizada tem rótulo
conhecido, então toda discordância é bug. Foi assim que achamos, por exemplo, uma súmula
do TSE resolvendo como a de mesmo número do STJ, um falso `real` que custaria caro no
conjunto cego.

### Decisões de projeto

Cada escolha que fechou uma porta tem registro próprio, com contexto, conta e medida:

- [D-001](wiki/decisoes/D-001_indice-de-cabecalho.md): índice de cabeçalho em vez de busca FTS por citação, porque acórdãos citam uns aos outros e a busca devolve quem menciona o número, não quem é o processo.
- [D-002](wiki/decisoes/D-002_empate-resolve-para-real.md): empate numa citação numerada resolve para `real`, pela assimetria de custo da métrica.
- [D-004](wiki/decisoes/D-004_gerador-por-moldes.md): o goldenset remonta os moldes da organização em vez de pedir a um LLM que redija documentos.
- [D-006](wiki/decisoes/D-006_calibracao-decide-o-topo.md): a confiança é a taxa de acerto medida por balde, porque no topo do ranking quem desempata é o bônus de calibração.
- [D-007](wiki/decisoes/D-007_sem-ensemble.md): sem ensemble; a união com a régua fica como opção para texto fora do estilo da organização.

## Como rodar

Tudo roda em container. Os dados da organização não são redistribuídos: baixe a aba Data
da competição e coloque `desafio1_bracis.db` e a pasta `txt/` em `dados/`.

```bash
scripts/baixar_pesos.sh   # pesos na revisão fixa do MODELO.md, em ./modelos
make build                # imagem de desenvolvimento
make dados                # confere se os dados estão no lugar
make predizer             # dados/txt → saidas/json
make avaliar              # métrica oficial, por nível
make erros                # erros restantes por categoria
make submissao            # saidas/submission.csv pelo conversor oficial
make test                 # testes
```

### Contrato de execução da organização

A imagem avaliada é o alvo padrão do `Dockerfile` (torch com CUDA 12.6). Pesos e dados
chegam por volume e nada sai para a rede:

```bash
docker build -t gama .
docker run --rm --gpus all --network none \
  -v "$PWD/modelos:/models:ro" \
  -v "$PWD/dados:/app/dados:ro" \
  -v /caminho/entrada:/data/in:ro \
  -v /caminho/saida:/data/out \
  gama --input /data/in --output /data/out
```

O stderr informa `extrator: neural`. Se o volume dos pesos faltar, aparece um aviso e o
pipeline segue com a régua, porque uma saída válida vale mais que uma submissão vazia. Sem
GPU visível o extrator roda em CPU, em cerca de 3 s por documento, dentro do teto de 60 s.

## Reprodutibilidade

| Item | Onde está |
|---|---|
| Código | este repositório; cada submissão cita o commit que produziu as saídas |
| Modelo | [`vinimlo/gama`](https://huggingface.co/vinimlo/gama), revisão fixa no [MODELO.md](MODELO.md); base [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base) (MIT), também com revisão fixa |
| Ambiente | `Dockerfile` (Python 3.12, torch 2.14.0) e `requirements.txt` com versões fixadas |
| Comando | o bloco acima |
| Determinismo | inferência em `eval()` sem amostragem, algoritmos determinísticos do torch, `PYTHONHASHSEED=0`; a saída é a mesma em GPU e em CPU |
| Rede | nenhuma chamada em tempo de execução (`HF_HUB_OFFLINE=1`) |
| Dados | fora da imagem e fora do repositório; chegam por volume |

O treino é um script UV (`treino/treinar.py`, dependências fixadas no cabeçalho) que roda
em HF Jobs, lê o goldenset numa revisão fixa do Hub e publica os pesos:

```bash
hf jobs uv run --flavor a10g-large --secrets HF_TOKEN treino/treinar.py \
  --dados vinimlo/gama-goldenset --revisao <sha> \
  --modelo-base jhu-clsp/mmBERT-base --saida vinimlo/gama
```

## Estrutura

```
src/gama/     pipeline: extração, normalização, índice, resolver, classificação
avaliacao/    harness da métrica oficial, catálogo de erros, calibração, inspeção sem gabarito
geracao/      gerador sintético pelos moldes da organização e expansão dos bancos por LLM
treino/       fine-tune, validação por dobras, ensaio no hardware-alvo
reais/        robustez em texto real: ingestão, anotação prata, adjudicação
vendor/       código da organização, intocado (métrica e conversor oficiais)
tests/        regressões do resolver, sondas de ruído, determinismo
bench/        benchmark contra extratores sem treino (GLiNER, Qwen3-8B, régua)
wiki/         decisões, experimentos e conceitos
```

Quatro invariantes quebram em silêncio se forem violadas:

1. A tabela de OCR letra→dígito só se aplica ao núcleo numérico (`span.digitos`), nunca ao
   trecho inteiro. Aplicada a um prefixo como `AgInt` ou `TST-ED-E-ED-RR`, ela gera uma
   chave fantasma e o sintoma é só "zero candidatos". Esse bug apareceu três vezes.
2. `vendor/` não se edita. É a cópia exata da métrica do servidor; mexer nela faz o número
   local divergir do leaderboard.
3. Em `extrair()`, as referências vagas registram antes dos números. Senão o ano
   (`em 2024`) é capturado como número de processo.
4. A confiança vem de `src/gama/calibracao.json`, medida por `avaliacao/calibrar.py` e
   nunca escrita à mão. Nunca 1,0: confiança alta em cima de erro é o pior caso do Brier.

## Licença

MIT, ver [LICENSE](LICENSE). A exceção é `vendor/`, cópia do código da organização do
desafio (métrica e conversor oficiais), que segue os termos dela.

## Equipe

Vinícius Melo e Gabriel Siron
