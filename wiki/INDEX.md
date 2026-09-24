# Gama: base de conhecimento

> Verificador de citações do Desafio Caça-Alucinações (BRACIS 2026 × Jusbrasil). O nome
> homenageia Luiz Gama, advogado abolicionista.

Comece por aqui. Uma página por conceito, experimento ou decisão; o destilado vale mais
que o bruto.

## Referência rápida

- Tarefa: achar citações jurídicas em `.txt` e classificar `real` / `inventada` /
  `incompleta`; as reais exigem o `id_canonico` correto no acervo congelado (1.014 registros).
- Métrica: `macroF1 · (1 − 0,5·τ) · (1 + 0,10·(1 − brier))`, níveis combinados
  `(1·N1 + 2·N2)/3`. Teto prático 1,10000. τ = fração das `inventada` do gabarito
  preditas como `real`, o erro grave.
- Regras que restringem a solução: só pesos abertos com revisão fixa; execução
  offline; 1 GPU 24 GB, 8 vCPUs, 32 GB RAM; média ≤ 60 s/documento.
- Solução: extrator neural Gama (mmBERT-base fine-tunado, BIO) com a guarda (régua onde o
  modelo hesita, [D-008](decisoes/D-008_guarda-do-extrator.md)) → normalização do OCR no
  núcleo numérico → índice canônico construído do SQLite → classe + confiança medida por
  balde. A régua (regex) é o fallback sem pesos.
- LLMs abertos (DeepSeek-V4-Pro, Kimi-K3, GLM-5.3): só no desenvolvimento, para
  expandir bancos de frases e anotar texto real. Nenhum em tempo de execução.

## O que já sabemos (achados que mudam decisão)

- O dev set sai de um gerador por moldes. 62% das frases se repetem; 1 data por
  documento; quebra de linha reproduzível byte a byte. O cego vem do mesmo gerador. Ver
  [gerador sintético](experimentos/2026-09-22_gerador-sintetico.md).
- Treinar só em sintético transfere. O Gama acerta 192/192 citações em documentos da
  organização que nunca viu; a expansão por LLM é o que leva a esse teto. Ver
  [validação por dobras](experimentos/2026-09-22_validacao-por-dobras.md).
- O dev set saturou como seletor. Todo candidato bom tira F1 = 1,0 nele; o desempate
  é o [estresse difícil](experimentos/2026-09-22_estresse-dificil.md) (redação nunca
  treinada), onde a régua cai para 0,848 e o Gama não erra borda em 4.447 spans.
- No topo, a calibração decide. Com F1 = 1,0, a ordem sai do bônus de Brier. Ver
  [D-006](decisoes/D-006_calibracao-decide-o-topo.md).
- Sem treino não chega lá. No estresse, o melhor extrator zero-shot (Qwen3-8B) marca
  0,811 e o Gama 1,0997. Ver [benchmark](experimentos/2026-09-23_bench-extratores-crus.md).
- O Gama especializou no estilo da organização. Em ementas reais fica atrás da régua e
  do Qwen3-8B zero-shot, marcando fragmentos soltos. Ver
  [benchmark](experimentos/2026-09-23_bench-extratores-crus.md) e
  [D-007](decisoes/D-007_sem-ensemble.md).
- O Gama sabe quando está inseguro. Nenhum de 34.171 acertos no estilo da organização (dev,
  estresse, v1) tem confiança abaixo de 0,98; os fragmentos em ementa real ficam abaixo.
  Tirar VAGA colada a outro span e trocar span abaixo de 0,95 pelo da régua leva o texto
  real de 0,605 a 0,808 sem mudar um documento no estilo da organização. Ver
  [otimização medida](experimentos/2026-09-23_otimizacao-medida.md).
- No estilo da organização, o que resta é resolução. Os únicos erros do estresse são
  desempate de cadeia ("Ag. Int.", embargos com ruído de OCR); duas regras gerais levam o
  estresse a 1,09999. Ver [otimização medida](experimentos/2026-09-23_otimizacao-medida.md).
- O leaderboard da fase de treino é ruído. Ele pontua contra a amostra de
  desenvolvimento, cujo gabarito foi distribuído. Quando o conjunto cego for ativado, o
  leaderboard reinicia.
- A documentação da Kaggle contradiz os dados da Kaggle. A aba Data diz 1.016
  registros e 225 citações; os arquivos têm 1.014 e 192. Fonte da verdade é o arquivo.
- `incompleta` virou trivialmente separável. A limpeza final removeu as 33 vagas sem
  fonte; as 32 restantes são todas `jurisprudencia` no molde tribunal + ano + relator, sem
  número. Ver [achados do webinar](conceitos/webinar-achados.md).
- As "vagas sem fonte" removidas do gabarito são negativos difíceis: mesma frase
  carregadora, preenchida com referência genérica. O modelo tem que aprender a não marcar.
- O ruído de OCR troca letra por dígito, nunca dígito por dígito (`l`→1, `S`→5,
  `O`→0, `G`→6, `g`→9). A normalização é reversível por construção, desde que aplicada
  só ao núcleo numérico.
- Errar o link custa menos que errar a classe. `real` com id errado gera só
  `fp[real]`; `inventada` onde era `real` gera os dois, `fn[real]` e `fp[inventada]`. Ver
  [D-002](decisoes/D-002_empate-resolve-para-real.md).
- A regra EXTRA perdoa sub-spans. Predição sem par ≥ 90% contida numa citação já
  casada é ignorada, não vira falso positivo.
- Jev (TypeSafe) e Sabiá-2/3 (Maritaca) desclassificam. Ver
  [modelos elegíveis](conceitos/modelos-elegiveis.md).

## Decisões

- [D-001](decisoes/D-001_indice-de-cabecalho.md): índice offline de cabeçalho em vez de FTS por citação
- [D-002](decisoes/D-002_empate-resolve-para-real.md): empate numerado resolve para `real`, não `incompleta`
- [D-004](decisoes/D-004_gerador-por-moldes.md): goldenset remontado dos moldes da organização; LLM só expande bancos
- [D-006](decisoes/D-006_calibracao-decide-o-topo.md): confiança = probabilidade de acerto medida por balde
- [D-007](decisoes/D-007_sem-ensemble.md): sem ensemble; a união com a régua fica como opção
- [D-008](decisoes/D-008_guarda-do-extrator.md): guarda do extrator (régua só onde o modelo hesita) e desempate por cadeia compatível
- [D-009](decisoes/D-009_gama-v1-3-destilado.md): Gama v1.3, o v1.2 destilado em 12 camadas; mesmo JSON no estilo da organização

## Experimentos

- [Baseline determinístico](experimentos/2026-09-22_baseline-deterministico.md): a régua, com 1,09830 no dev, confirmado no Kaggle
- [Gerador sintético](experimentos/2026-09-22_gerador-sintetico.md): moldes da organização, rótulo exato por construção
- [Validação por dobras](experimentos/2026-09-22_validacao-por-dobras.md): 192/192 em documentos nunca vistos
- [Estresse difícil](experimentos/2026-09-22_estresse-dificil.md): redação nunca treinada e ruído forte
- [Texto real](experimentos/2026-09-22_texto-real.md): ementas reais, fora do estilo da organização
- [Benchmark contra extratores sem treino](experimentos/2026-09-23_bench-extratores-crus.md): Gama × régua × GLiNER × Qwen3-8B zero-shot
- [Otimização medida](experimentos/2026-09-23_otimizacao-medida.md): filtro de fragmentos, desempate de cadeia, vieses do texto real, CPU, meia precisão
- [Destilação](experimentos/2026-09-23_destilacao.md): alunos de 13 camadas e mmBERT-small; a destilação segura o texto real, o aluno de 13 camadas empata no estilo da organização; o de 12 primeiras camadas sai idêntico e vira o v1.3

## Conceitos

- [Convenções de borda](conceitos/convencoes-de-borda.md): onde cada citação começa e termina no gabarito
- [Modelos elegíveis](conceitos/modelos-elegiveis.md): o que as regras permitem e o que foi avaliado
- [Achados do webinar](conceitos/webinar-achados.md): o que a organização disse além do site e do Kaggle
- [Regras e reprodutibilidade](conceitos/regras-e-reprodutibilidade.md): o que é re-executado, o envelope vale para a inferência, e como a solução cumpre cada exigência
- [Literatura](conceitos/literatura.md): papers do arXiv que sustentam cada parte da solução, e a lacuna que ela ocupa

## Como registrar

- Rodada medida nova → `experimentos/AAAA-MM-DD_<slug>.md` + linha acima.
- Decisão que fecha uma porta → `decisoes/D-NNN_<slug>.md` + linha acima.
- Fato que muda decisão → item em "O que já sabemos", com link para a página.
