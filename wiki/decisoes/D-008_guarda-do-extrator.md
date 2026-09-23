# D-008: Guarda do extrator e desempate por cadeia compatível

Data: 23/09/2026 · Status: aceita

## Critério

Uma mudança no pipeline entra se não mudar nenhum documento no estilo da organização (dev,
estresse difícil e o `final_v1`, que o v1.2 não treinou) e melhorar algo medido fora dele,
com a escolha validada fora dos dados em que foi feita. Números completos na
[otimização medida](../experimentos/2026-09-23_otimizacao-medida.md).

## Decisão

1. O extrator de produção (`--extrator neural`, o da imagem) é o Gama com a guarda
   (`src/gama/extratores/guarda.py`):
   - referência vaga a até 2 caracteres de outro span sai;
   - span com confiança abaixo de 0,95 sai, e no lugar entra o span da régua que o cruza, se
     houver e se não cruzar um span confiante.

   O modelo sozinho continua disponível como `--extrator neural-cru`.
2. O resolver ganha um quarto nível de desempate (a cadeia lida cabe na cadeia da ficha,
   elo a elo por prefixo; entre as que cabem, a de mesmo comprimento) e trata número CNJ com
   mais de 20 dígitos como classe grudada pelo OCR, ficando com os 20 últimos.

## Medida

| | Antes | Depois |
|---|---|---|
| Dev, métrica oficial | 1,10000 | 1,10000 |
| Estresse difícil, métrica oficial | 1,09974 | 1,09999 |
| Documentos mudados pela guarda (dev + estresse + final_v1, 4.626) | | 0 |
| Texto real, F1 de extração (305 ementas) | 0,605 | 0,808 |
| Conjuntos sintéticos v0 a v3 com o ouro como extração | 1,09949 a 1,09992 | 1,09989 a 1,09999 |

- Nenhum de 34.171 acertos no estilo da organização tem confiança abaixo de 0,98; o corte em
  0,95 tem folga.
- Escolhendo o filtro numa metade das ementas e medindo na outra, 10 vezes, a escolha é
  sempre a mesma e o ganho médio é +0,20.
- O desempate não mexe em τ: só escolhe entre candidatos que já existem.

## Por que não é a união da D-007

A união acrescentava a régua em todo lugar e custava 61 falsos positivos em N2. A guarda só
chama a régua onde o modelo hesita, e no estilo da organização ele não hesita: por isso o
custo ali é zero.

## Riscos que ficam

- O ruído do estresse preserva o primeiro dígito do número. Um conjunto cego com ruído mais
  pesado poderia baixar a confiança de uma citação legítima; aí a régua assume aquele trecho,
  e o que se perde é só o que ela também não achar.
- Um documento do cego fora do estilo, com cabeçalho em caixa-alta contendo citação, fica com
  o que a régua achar ali.

Reabrir se o cego mostrar citação legítima com confiança abaixo de 0,95.
