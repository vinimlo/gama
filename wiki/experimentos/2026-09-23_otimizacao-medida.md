# 2026-09-23 · Otimização medida: filtro de fragmentos, resolução, vieses e custo

Pergunta: o [benchmark](2026-09-23_bench-extratores-crus.md) deixou hipóteses abertas. O
Gama perde para a régua e para o Qwen3-8B em ementas reais por causa de fragmentos? Um
filtro barato resolve sem custar nada no estilo da organização? O ouro do texto real
favorece o Qwen? E onde mais dá para ganhar em qualidade, tempo, armazenamento e custo?

## Desenho

- Tudo o que dava para medir offline foi medido com os spans já gravados do benchmark, sem
  GPU: `bench/hipoteses.py` (ações `filtros`, `selecao`, `vieses`, `auditoria`, `precisao`),
  resultados em `bench/hipoteses.json`.
- Critério de aceitação de qualquer mudança: nenhum documento muda no dev (26) e no
  estresse difícil (600). Com esses spans iguais, a nota oficial é igual por construção.
- Uma revisão independente recebeu o contexto e os números e propôs um plano priorizado. Ela chegou aos mesmos números de reponderação e ao
  mesmo filtro de confiança por conta própria; o que ela trouxe de novo está marcado abaixo.

## 1. Fragmentos no texto real: pós-filtro

A confiança do Gama por span (média dos tokens rotulados como entidade) separa bem acerto
de fragmento:

| | Acertos | Menor confiança | Acertos < 0,95 |
|---|---|---|---|
| Dev | 192 | 0,9998 | 0 |
| Estresse difícil | 4.447 | 0,9920 | 0 |
| Texto real | 915 | 0,5785 | 35 |

No texto real, os falsos positivos têm confiança mediana entre 0,85 e 0,94. E 406 deles são
VAGA: o modelo lê "Rel. Min. Fulano, julgado em ..." colado a um precedente numerado como se
fosse a referência vaga do molde da organização. No dev e no estresse, a VAGA mais próxima de
outra citação está a 4 caracteres (". O "), sempre depois de fim de frase; no texto real, 371
das 408 VAGA previstas estão a até 5 caracteres de outro span.

F1 de extração no texto real (IoU ≥ 0,5), com os documentos mudados no dev e no estresse:

| Filtro sobre o Gama | Texto real | Dev mudados | Estresse mudados |
|---|---|---|---|
| nenhum | 0,605 | 0 | 0 |
| exigir algum dígito | 0,660 | 0 | 0 |
| exigir chave para o acervo | 0,621 | 0 | 8 (nota 1,09877) |
| comprimento ≥ 10 | 0,679 | 0 | 0 |
| comprimento ≥ 12 | 0,678 | 0 | 10 (nota 1,09909) |
| confiança ≥ 0,90 | 0,742 | 0 | 0 |
| confiança ≥ 0,95 | 0,766 | 0 | 0 |
| confiança ≥ 0,98 | 0,766 | 0 | 0 |
| VAGA a até 2 caracteres de outro span sai | 0,684 | 0 | 0 |
| confiança < 0,95 trocada pelo span da régua que a cruza | 0,777 | 0 | 0 |
| VAGA colada sai, depois troca pela régua abaixo de 0,95 | 0,808 | 0 | 0 |

A troca pela régua só age onde o modelo hesita. No estilo da organização nenhum acerto fica
abaixo de 0,99, então ali nada muda; fora dele, a régua cobre o ponto em que o modelo não tem
certeza. É diferente da união da [D-007](../decisoes/D-007_sem-ensemble.md), que acrescentava
a régua em todo lugar e custava 61 falsos positivos em N2.

O ganho sobrevive à escolha? Escolhendo o filtro numa metade das ementas e medindo na outra
(5 sorteios, 2 sentidos): as 10 escolhas caem no mesmo filtro, com ganho médio de +0,202
fora da metade usada (mínimo +0,172). Bootstrap pareado por ementa (2.000 reamostras):

| | F1 | IC95 |
|---|---|---|
| Gama filtrado | 0,808 | 0,783 a 0,830 |
| Qwen3-8B | 0,707 | 0,671 a 0,742 |
| Régua | 0,663 | 0,629 a 0,696 |
| Gama sem filtro | 0,605 | 0,569 a 0,643 |

Diferença Qwen3-8B menos Gama filtrado: −0,100 (IC95 −0,138 a −0,062); o Gama filtrado é
melhor em 100% das reamostras.

Filtros por dígito e por comprimento ficam de fora. Além de renderem menos, o estresse não
os protege de verdade: o injetor de ruído preserva o primeiro dígito do número, então um
"Súmula l" com o número inteiro convertido em letra nunca foi testado (achado da revisão
independente).

## 2. O texto real mede o que parece medir?

- Amostragem. De 2.454 ementas, 1.919 tinham spans diferentes entre Gama v1 e régua. A
  amostra anotada ficou com 248 divergentes e 57 concordantes, quase a proporção da
  população (81% contra 78%). Reponderar pelos estratos quase não muda nada: Gama 0,6065,
  régua 0,6641, Qwen3-8B 0,7049. Mas a anotação descartou 17% das divergentes (52 de 300)
  contra 5% das concordantes (3 de 60), por exigir reprodução exata do texto: ementas mais
  difíceis ficaram sub-representadas, em direção que não dá para medir sem anotá-las
  (achado da revisão independente).
- Convenção de borda. O ouro foi anotado por LLMs, e a borda pesa na ordem:

  | Casamento | Gama filtrado | Qwen3-8B | Régua | Gama |
  |---|---|---|---|---|
  | Exato | 0,668 | 0,659 | 0,382 | 0,494 |
  | IoU ≥ 0,5 | 0,808 | 0,707 | 0,663 | 0,605 |
  | Qualquer sobreposição | 0,863 | 0,738 | 0,854 | 0,658 |

  No casamento exato o Qwen quase empata com o Gama filtrado e deixa a régua longe; com
  qualquer sobreposição a régua passa o Qwen. A vantagem do Qwen sobre a régua é, em boa
  parte, borda no estilo do anotador. O Gama filtrado fica em primeiro nos três critérios.
- Qualidade do ouro. Lendo 120 erros sorteados (20 FP e 20 FN de cada extrator): 2 dos 20
  falsos positivos do Gama filtrado são citações que o ouro esqueceu ("VERBETE Nº 54 DA
  SÚMULA/STJ", "Código Civil, arts. 768 e 798"); lei citada sem artigo ("Lei nº 6.830/80")
  às vezes entra no ouro e às vezes não, o que pesa mais contra o Qwen (8 dos 20 falsos
  positivos dele). Os 20 falsos negativos do Gama filtrado são citações de verdade, quase
  todas em ementa em caixa-alta ("SÚMULA 7/STJ", "ART. 387, IV, DO CÓDIGO DE PROCESSO
  PENAL") ou em lista depois de "Precedentes:". É o ponto fraco que sobra.

## 3. Resolução no estilo da organização

O estresse tinha extração perfeita e nota 1,09974, não 1,1. Os dois únicos erros são links
errados em N2, os dois em cadeia de agravo interno em embargos de divergência ("Ag. Int. nos
Emb. Div. nos EDv no REsp", "Agravo Interno nos Ernbarg0s dc Divergêneia"), resolvidos por
chute com confiança 0,5. O leitor de cadeia não reconhece "Ag. Int." com ponto e espaço nem
o ruído de OCR em "Embargos de Divergência", e o desempate só aceitava cadeia igual.

Duas regras gerais, sem exceção por número:

- Quarto nível de desempate: fica o único candidato cuja cadeia contém a cadeia lida como
  subsequência, cada elemento lido sendo prefixo do elemento da ficha ("E" lido casa com
  "EDv"); havendo mais de um, o de mesmo comprimento ("AgRG" lido como "Ag").
- Número CNJ tem 20 dígitos: com mais que isso, é prefixo de classe grudado pelo OCR
  ("AgR-A1 0603026-69..."), e fica com os 20 últimos.

Medido com os spans do Gama (dev, estresse) e com o ouro como extração nos conjuntos
sintéticos que não foram usados para achar os erros:

| Conjunto | Antes | Depois | Resoluções mudadas |
|---|---|---|---|
| Dev | 1,10000 | 1,10000 | 0 |
| Estresse difícil | 1,09974 | 1,09999 | 2 |
| v0 (400 docs) | 1,09949 | 1,09989 | 2 |
| v1 (4.000) | 1,09992 | 1,09999 | 5 |
| v2 (4.000) | 1,09988 | 1,09999 | 6 |
| v3 (6.000) | 1,09990 | 1,09999 | 7 |

Todas as mudanças vão para o id certo. O que sobra é "ARE nº 1356440/SP", sem cadeia no
texto, com duas fichas no acervo que só diferem no ordinal dos embargos: ambíguo de verdade,
e ali confiança 0,5 é a resposta honesta. O desempate não mexe em τ: só escolhe entre
candidatos que já existem, e a classe já era `real`. A regra do CNJ pode, em tese, achar
candidato onde antes não havia; para isso uma citação inventada precisaria ter mais de 20
dígitos e os 20 últimos serem um processo do acervo. Em 15.026 documentos ela agiu uma vez,
no caso certo.

## 4. Tempo, armazenamento e custo

CPU. O benchmark tinha gravado 18,6 s por documento no dev em CPU (pico de 77 s), o que
punha em dúvida o teto de 60 s caso a GPU do avaliador não fosse usada. Medido de novo, no
mesmo container (cota de 4 CPUs), com os mesmos spans byte a byte: 1,58 s por documento. A
medida antiga foi tirada com o disco quase cheio e o Docker degradado; `bench/resultados.json`
foi regenerado e só esse número mudou. Numa amostra de 68 documentos (8 do dev, 30 do
estresse N2, 30 ementas):

| Configuração em CPU | Dev | Estresse N2 | Texto real | Saída |
|---|---|---|---|---|
| Padrão (o torch abre 10 threads) | 2,09 s | 3,26 s | 0,73 s | referência |
| 4 threads, igual à cota | 1,73 s | 2,81 s | 0,52 s | idêntica |
| 4 threads, janelas sem sobreposição | 1,66 s | 2,81 s | 0,51 s | idêntica nesta amostra |
| 4 threads, int8 dinâmico | 1,36 s | 1,71 s | 0,34 s | 16 de 68 documentos mudam |

O torch dimensiona as threads pelos núcleos da máquina, não pela cota do container; alinhar
as duas dá 14% a 29% de ganho sem mudar nada na saída. O int8 muda spans e confianças (até
0,18) e fica de fora.

GPU (L4, `bench/extrair_dtype.py`, cerca de US$ 0,15 por rodada), sobre o estresse, o
texto real e o `final_v1` (4.000 documentos sintéticos de outra semente, que o v1.2 não
treinou). Cada precisão parte dos pesos FP32 do disco, num modelo novo. A execução em FP32
reproduz os spans do benchmark nas 305 ementas, byte a byte.

| Precisão | s/doc estresse | s/doc v1 | VRAM pico | Estresse + v1: docs com spans diferentes | Texto real: docs com spans diferentes | F1 real com o filtro |
|---|---|---|---|---|---|---|
| FP32 | 0,073 | 0,061 | 1.285 MiB | referência | referência | 0,8076 |
| BF16 | 0,029 | 0,027 | 671 MiB | 0 (confiança anda até 0,0012) | 27 | 0,8071 |
| FP16 | 0,030 | 0,027 | 671 MiB | 0 (até 0,00007) | 5 | 0,8073 |

Uma primeira rodada deste job tinha dois defeitos, apontados pela revisão independente: a
rodada FP16 herdava um arredondamento BF16 e uma cópia FP32 ficava na GPU, inflando os picos
(2.476 e 1.863 MiB). A tabela acima é a da rodada corrigida. Guardar os pesos em FP16 e
voltar a FP32 na carga também não é exato: num teste em CPU com 126 documentos, o dev e 40
documentos N2 do estresse ficaram iguais, mas 1 de 60 ementas reais mudou um span.

Meia precisão é 2,5 vezes mais rápida, usa metade da memória e guardaria os pesos em 0,62 GB
em vez de 1,23 GB. No estilo da organização a saída é a mesma; fora dele, perto da fronteira
de decisão, muda um pouco, e o F1 filtrado cai 0,0003 (FP16) a 0,0005 (BF16). Pela regra de
zero perda, fica de fora; com 0,07 s por documento contra um teto de 60 s, o tempo também não
pede a troca.

O mesmo job mediu a cauda de confiança no `final_v1`: 29.532 acertos, nenhum falso
positivo, menor confiança 0,9867. Somando dev, estresse e v1, nenhum de 34.171 acertos no
estilo da organização fica abaixo de 0,98; o corte do filtro em 0,95 tem folga.

Armazenamento. 64% dos 307,5 milhões de parâmetros são a tabela de embeddings (vocabulário
de 256 mil tokens). Quantizar só as camadas lineares deixaria os pesos em cerca de 0,9 GB;
podar o vocabulário é arriscado, porque nome próprio e ruído de OCR podem cair em token
podado. No disco local, uma cópia temporária de um modelo das dobras e o cache do Hub somam
2,4 GB recuperáveis. No Hub, os 8 modelos das dobras ocupam 9,8 GB privados.

Custo. O treino do v1.2 levou 12,3 min numa A100 (cerca de US$ 0,51). Com o Gama a 1,6 s por
documento em CPU, avaliar no estilo da organização cabe no Mac, sem job; GPU só para o que
precisa de LLM.

## Leitura

1. A queda do Gama em texto real era fragmento, e o próprio modelo sabe quando está
   inseguro. Com o filtro (VAGA colada sai; abaixo de 0,95, vale a régua), o Gama passa de
   0,605 para 0,808 em ementas reais, à frente do Qwen3-8B zero-shot (0,707) e da régua
   (0,663), sem mudar um único documento no dev, no estresse ou no v1.
2. O texto real mede com viés de borda: o ouro de LLM premia o Qwen no casamento exato e a
   régua no frouxo. A conclusão principal resiste aos três critérios. A amostragem pesa
   pouco; os descartes da anotação são o viés que continua sem medida.
3. No estilo da organização, o que restava era resolução, não extração: duas regras gerais
   de desempate e de CNJ levam o estresse de 1,09974 a 1,09999, confirmado em 14.400
   documentos que não serviram para achar o erro.
4. Tempo e custo não são gargalo. A medida de CPU que preocupava estava errada (1,6 s, não
   18,6 s). Meia precisão e threads alinhadas à cota rendem velocidade sem mudar a saída no
   estilo da organização, mas não mudam a nota.
5. O ponto fraco que sobra fora do molde é a ementa em caixa-alta ("SÚMULA 7/STJ"). Todo
   documento do estilo da organização tem cabeçalho em caixa-alta, então qualquer tratamento
   precisa agir só onde o modelo não marcou nada, e ser medido de novo.
