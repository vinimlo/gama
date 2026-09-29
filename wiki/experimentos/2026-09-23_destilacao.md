# 2026-09-23 · Destilação: o mesmo Gama, menor e mais rápido

Pergunta: um aluno menor, treinado para imitar o Gama, entrega o mesmo resultado com menos
tamanho e mais velocidade? Critério do dono do projeto: nenhuma perda de qualidade.

## Desenho

- Professor: Gama v1.2 (`vinimlo/gama@ad06ffd`, mmBERT-base, 307,5M parâmetros).
- Aluno A: o próprio Gama com 13 das 22 camadas, `[0,1,2,6,7,8,12,13,14,18,19,20,21]`, que
  preservam o padrão de atenção global/local do ModernBERT; embeddings, cabeça e camadas
  herdados (262,4M).
- Aluno B: `jhu-clsp/mmBERT-small@abc3262` (140,6M), mesmo tokenizador do professor.
- Controle: o aluno B treinado só com o ouro, sem o professor.
- Perda: `0,5 · CE(ouro) + T² · KL(professor/T ‖ aluno/T) + 0,5 · KL(professor ‖ aluno)`, com
  T = 2. O professor roda no próprio treino, em FP32 e janelas de 2.048 tokens, como na
  produção. A guarda (D-008) e a calibração dependem da confiança, então o aluno precisa
  copiar as probabilidades, não só o rótulo.
- Dados: o `final_v3` (o mesmo treino do professor) e 1.818 ementas reais sem rótulo, fora
  de qualquer teste (`reais/sortear_teste.py`), onde só vale o termo do professor.
- Treino: `treino/destilar.py`, HF Jobs, A100; A com 3 épocas (16 min), B com 5 (26 min),
  controle com 5 (6 min).
- Avaliação: `bench/extrair_alunos.py` (L4, FP32, os quatro modelos no mesmo job) e
  `bench/alunos.py`, pelo caminho de produção inteiro: extrator, guarda, resolver,
  confiança calibrada, JSON final. O dev roda em CPU, localmente.

## Resultado

Estilo da organização, métrica oficial:

| | Dev (26) | Estresse (600) | final_v1 (4.000) | Documentos com JSON diferente do professor |
|---|---|---|---|---|
| Professor | 1,10000 | 1,09999 | 1,09999 | |
| A | 1,10000 | 1,09999 | 1,09999 | 1 de 4.626 |
| B | 1,10000 | 1,09966 | 1,09994 | 6 |
| Controle | 1,10000 | 1,09999 | 1,09999 | 11 |

Texto real, F1 de extração com a guarda, diferença para o professor com IC95 por bootstrap
pareado:

| | 305 ementas do ouro antigo | 172 ementas do teste novo |
|---|---|---|
| Professor | 0,8076 | 0,8204 |
| A | 0,8191 (+0,012; −0,004 a +0,028) | 0,8426 (+0,022; +0,007 a +0,038) |
| B | 0,8154 (+0,008; −0,007 a +0,023) | 0,8396 (+0,019; +0,002 a +0,036) |
| Controle | 0,5743 (−0,233) | 0,5881 (−0,232) |

Custo de execução:

| | Parâmetros | Pesos FP32 | L4, s/doc no estresse | VRAM | CPU, s/doc no dev |
|---|---|---|---|---|---|
| Professor | 307,5M | 1,23 GB | 0,071 | 1.286 MiB | 1,62 |
| A | 262,4M | 1,05 GB | 0,047 | 1.106 MiB | 0,96 |
| B | 140,6M | 0,56 GB | 0,040 | 613 MiB | 0,81 |

## Leitura

1. A destilação é o que segura a qualidade fora do molde. O controle, com o mesmo modelo do
   aluno B e só o ouro, fica idêntico ao professor no estilo da organização e perde 0,23 no
   texto real: aprende o rótulo mas não a incerteza, e a guarda deixa de funcionar.
2. O aluno A empata com o professor na nota em todos os conjuntos do estilo da organização e
   melhora no texto real, com ganho significativo no teste novo. A única diferença em
   4.626 documentos é uma borda em N2 ("No TST-AIRR-..." em vez de "processo No
   TST-AIRR-..."), que ainda casa com o ouro (IoU 0,80) e não muda classe, link nem
   confiança. É 1,5 vez mais rápido na L4 e 1,7 vez em CPU, mas só 15% menor.
3. O aluno B entrega o que mais se pediu em tamanho e velocidade (2,2 vezes menor, 1,8 vez
   mais rápido na L4, 2 vezes em CPU) e também melhora no texto real, mas perde no ruído
   pesado de OCR: uma referência vaga muito ruidosa não é achada, uma citação é partida em
   duas e duas confianças caem de faixa. Pela regra de zero perda, não passa.
4. Nenhum aluno é byte a byte idêntico ao professor no estilo da organização. O aluno A
   empata na nota, e a única diferença não muda classe, link nem confiança publicada.

## Segunda rodada: o que a literatura sugeria

- Poda das camadas de cima ([Sajjad et al., 2004.03844](https://arxiv.org/abs/2004.03844)):
  as camadas de baixo pesam mais, e tirar as de cima foi a melhor estratégia deles. Alunos
  com as 12 e as 9 primeiras camadas do Gama, mesma receita do A (o padrão global/local se
  mantém por construção).
- Mais dados onde o aluno pequeno falhava ([Stanton et al., 2106.05945](https://arxiv.org/abs/2106.05945)):
  B2, o mmBERT-small com 8 épocas, documentos N2 com peso dobrado e 3.000 documentos das
  pastas de dobras (sementes 100 e 101, fora dos testes).

| | Dev / estresse / v1 | JSON diferente do professor (4.626 docs) | Real 305 | Real 172 novas | L4 s/doc (estresse) | CPU s/doc (dev) | Pesos |
|---|---|---|---|---|---|---|---|
| Professor | 1,10000 / 1,09999 / 1,09999 | | 0,8076 | 0,8204 | 0,071 | 1,62 | 1,23 GB |
| Base 12 camadas | igual | 0 | 0,8180 | 0,8392 (+0,019; +0,003 a +0,033) | 0,046 | 0,93 | 1,03 GB |
| Base 9 camadas | 1,10000 / 1,09966 / 1,09999 | 4 | 0,8174 | 0,8386 | 0,037 | 0,64 | 0,97 GB |
| B2 | igual | 1 | 0,8139 | 0,8225 (+0,002; −0,012 a +0,016) | 0,041 | 0,84 | 0,56 GB |

O aluno com as 12 primeiras camadas sai byte a byte idêntico ao professor em todos os 4.626
documentos do estilo da organização e melhora no texto real, com ganho significativo no
teste intocado. Com 9 camadas começa a perder no ruído pesado. O B2 fecha a perda do B na
nota; a única diferença que sobra é o mesmo ponto fraco, um prefixo ruidoso ("Ernbargos de
Declaração no") deixado de fora, que ainda casa com o ouro e resolve para o mesmo registro.

## Checagens de entrega do aluno A

As mesmas que o professor passou antes de ir para a submissão:

| Checagem | Professor | Aluno A |
|---|---|---|
| Sondas de ruído do extrator (`tests/test_sondas_ruido.py`) | 6 de 6 | 6 de 6, os mesmos casos |
| Tabela de calibração | medida (D-006) | igual por construção: o JSON final, com a confiança publicada, é o mesmo em 4.625 de 4.626 documentos, e no restante a citação cai no mesmo balde |
| GPU igual a CPU (100 documentos do estresse, metade N2) | idêntico | idêntico, confiança com diferença zero |
| Imagem de entrega sem rede, no dev | 1,10000 | 1,10000, `extrator: neural`, 1,09 s por documento em CPU |

## Checagens de entrega do aluno base 12 camadas

As mesmas, com os pesos na revisão fixa `f49e37b` do repositório de experimento do aluno. Os
alunos foram apagados do Hub depois da escolha; os pesos do base 12 camadas são os do
`vinimlo/gama@5f924ca` (v1.3), e os números de todos os alunos ficam em `bench/alunos.json`:

| Checagem | Professor | Base 12 camadas |
|---|---|---|
| Sondas de ruído do extrator (`tests/test_sondas_ruido.py`) | 6 de 6 | 6 de 6, os mesmos casos |
| Tabela de calibração | medida (D-006) | igual por construção: o JSON final, com a confiança publicada, é o mesmo nos 4.626 documentos |
| GPU igual a CPU (100 documentos do estresse, metade N2) | idêntico | spans idênticos; confiança com diferença máxima de 6,6e-9, nenhuma cruza os cortes de 0,95 (guarda) e 0,98 (faixa da calibração) |
| Imagem de entrega sem rede, no dev (4 vCPUs, 4 GB) | 1,10000, 1,62 s por documento | 1,10000, `extrator: neural`, 0,99 s por documento; JSON idêntico ao do professor nos 26 documentos |
