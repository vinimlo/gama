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
4. Nenhum aluno é byte a byte idêntico ao professor no estilo da organização; a submissão
   segue com o professor.
