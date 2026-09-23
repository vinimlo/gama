# Literatura: onde a solução se apoia

Papers do arXiv ligados a cada parte da solução, com o que cada um afirma e o que medimos
aqui. Levantamento de 23/09/2026: busca por tema no índice de papers do Hugging Face (que
espelha o arXiv) e conferência de id, título e data direto na API do arXiv. Os resumos
citados foram lidos na fonte.

## 1. O problema: citação jurídica inventada

- [Large Legal Fictions](https://arxiv.org/abs/2401.01301) (Dahl et al., 2024): LLMs erram
  fatos jurídicos com frequência, não corrigem premissas falsas do usuário e muitas vezes
  não reconhecem o próprio erro.
- [Hallucination-Free?](https://arxiv.org/abs/2405.20362) (Magesh et al., 2024): primeira
  avaliação pré-registrada de ferramentas jurídicas comerciais com RAG; alucinam entre 17% e
  33% das vezes, apesar de anunciarem o contrário.
- [LegalCiteBench](https://arxiv.org/abs/2605.10186) (Chen et al., 2026): sem consulta
  externa, 21 LLMs ficam abaixo de 7/100 em recuperar citações; em 20 deles a taxa de
  resposta enganosa passa de 94%, e pedir que o modelo se abstenha não melhora a correção.
- [Evaluating LLM-based Approaches to Legal Citation Prediction](https://arxiv.org/abs/2412.06272)
  (Han et al., 2024): LLMs gerais ou jurídicos sozinhos ficam perto de zero; fine-tuning
  numa tarefa específica está entre as melhores soluções.
- [Is this Citation on Point?](https://arxiv.org/abs/2608.12571) (2026): citações
  alucinadas são "em grande parte pegas por consulta a base de dados"; o problema difícil é
  outro, a citação real que não sustenta o que se afirma, fora do escopo do desafio.

O que isso sustenta: decidir se uma citação existe é consulta a um acervo, não inferência de
modelo. É a arquitetura do Gama: o modelo só acha o trecho, e um índice exato no SQLite
decide real, inventada ou incompleta. Nenhum LLM roda na inferência.

- [Source or It Didn't Happen](https://arxiv.org/abs/2605.08583) (Li et al., 2026, CiteTracer):
  o trabalho mais próximo em desenho, para referências científicas. Classifica em real,
  potencial e alucinada, faz casamento determinístico de campos e só manda o ambíguo para
  juízes especializados, numa cascata. O benchmark deles é sintético, feito de sementes reais
  com mutações controladas, como o nosso gerador (fichas reais do acervo e citações
  inventadas de propósito).

## 2. Extração como classificação de tokens num encoder

- [ModernBERT](https://arxiv.org/abs/2412.13663) (Warner et al., 2024) e
  [mmBERT](https://arxiv.org/abs/2509.06888) (Marone et al., 2025): o encoder que usamos,
  pré-treinado em 3 trilhões de tokens de mais de 1.800 línguas, com contexto longo e
  atenção alternando global e local.
- [Portuguese Named Entity Recognition using BERT-CRF](https://arxiv.org/abs/1909.10649)
  (Souza et al., 2019): NER em português com BERT fine-tunado, dos autores do BERTimbau.
- [GLiNER](https://arxiv.org/abs/2311.08526) (Zaratiana et al., 2023): NER zero-shot com
  encoder compacto, uma das bases do nosso benchmark.
- NER jurídico em outras línguas mostra o mesmo padrão, modelo de domínio fine-tunado:
  [alemão](https://arxiv.org/abs/2003.13016), [Índia](https://arxiv.org/abs/2211.03442),
  [Sérvia](https://arxiv.org/abs/2502.10582), e [LEGAL-BERT](https://arxiv.org/abs/2010.02559)
  para adaptação ao domínio.
- Brasil: [LegalNLP](https://arxiv.org/abs/2110.15709) (Polo et al., 2021) publica modelos
  para a linguagem jurídica brasileira; [LegalBench-BR](https://arxiv.org/abs/2604.18878)
  (2026) mostra um BERTimbau com LoRA 22 a 28 pontos acima de LLMs comerciais em classificar
  decisões do TJSC.
- [Citation graph from 100 million Ukrainian court decisions](https://arxiv.org/abs/2605.15362)
  (2026): extração de citações por regex em escala, com precisão 1,00 numa amostra de 200
  decisões. Sustenta a régua como reserva: no formato canônico, regex funciona.

No nosso benchmark, o Gama fine-tunado marca 1,0997 no estresse difícil, contra 0,811 do
Qwen3-8B zero-shot e 0,327 do GLiNER.

## 3. Dados sintéticos e ruído de OCR

- [Data Centric Domain Adaptation for Historical Text with OCR Errors](https://arxiv.org/abs/2107.00927)
  (März et al., 2021): injetar erros de OCR sintéticos nos dados de treino melhora NER em
  texto ruidoso. É o papel do nosso injetor de ruído, calibrado no nível 2 do dev.
- Aumento de dados para NER: [survey](https://arxiv.org/abs/2105.03075) (Feng et al., 2021) e
  [LLM-DA](https://arxiv.org/abs/2402.14568) (2024), que usa LLM para ampliar dados de NER.
  No nosso caso o LLM só amplia bancos de frases; a citação e o rótulo vêm do acervo, exatos
  por construção.

## 4. Anotação por LLM e qualidade do gabarito

- [ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks](https://arxiv.org/abs/2303.15056)
  (Gilardi et al., 2023) e [Open-Source LLMs for Text Annotation](https://arxiv.org/abs/2307.02179)
  (Alizadeh et al., 2023): LLMs, inclusive abertos, anotam com qualidade comparável ou
  superior à de anotadores de multidão.
- [NoiseBench](https://arxiv.org/abs/2405.07609) (Merdjanovska et al., 2024): ruído real de
  rótulo em NER, inclusive o de LLMs, é bem mais difícil que ruído simulado.

O que isso sustenta: o ouro de texto real, anotado por dois LLMs abertos e adjudicado, é
aceitável como medida, mas tem viés. Por isso medimos com bootstrap, auditamos 120 erros à
mão, e sorteamos um teste novo que não serviu para escolher nada
([otimização medida](../experimentos/2026-09-23_otimizacao-medida.md)).

## 5. Confiança, calibração e a guarda

- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) (Guo et al.,
  2017) e [Calibration of Pre-trained Transformers](https://arxiv.org/abs/2003.07892) (Desai
  e Durrett, 2020): redes modernas são mal calibradas por padrão; BERT e RoBERTa saem
  calibrados no domínio, o caso difícil é fora dele, e escala de temperatura e suavização de
  rótulo ajudam. É o motivo de a confiança publicada ser medida por balde
  ([D-006](../decisoes/D-006_calibracao-decide-o-topo.md)) e de a guarda usar a confiança do
  modelo só como separador de acerto e fragmento, não como probabilidade.
- [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500)
  (Geifman e El-Yaniv, 2017) e [SelectiveNet](https://arxiv.org/abs/1901.09192): rejeitar a
  predição incerta reduz o erro a uma taxa controlada. A guarda
  ([D-008](../decisoes/D-008_guarda-do-extrator.md)) é essa ideia com reserva: o trecho
  incerto é rejeitado e, onde a régua tem resposta, ela entra.
- [FrugalGPT](https://arxiv.org/abs/2305.05176) (Chen et al., 2023): cascata de modelos por
  confiança, o mais barato primeiro. A guarda é uma cascata ao contrário: o modelo decide
  onde tem certeza, e a regra de custo quase zero cobre onde ele hesita.

## 6. Compressão: destilação e poda de camadas

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (Hinton
  et al., 2015): o aluno aprende as probabilidades suavizadas do professor, com temperatura.
  É a perda que usamos.
- [DistilBERT](https://arxiv.org/abs/1910.01108) (Sanh et al., 2019) e
  [TinyBERT](https://arxiv.org/abs/1909.10351) (Jiao et al., 2019): destilação de BERT em
  modelos menores com a maior parte do desempenho.
- [On the Effect of Dropping Layers of Pre-trained Transformer Models](https://arxiv.org/abs/2004.03844)
  (Sajjad et al., 2020): podar até 40% das camadas de BERT, RoBERTa e XLNet mantém até 98% do
  desempenho, empatando com modelos destilados; as camadas de baixo são as mais importantes.
  O aluno A tira 41% das camadas do Gama e mantém as três primeiras.
- [Weight subcloning](https://arxiv.org/abs/2312.09299) (Samragh et al., 2023, Apple) e
  [LayerDrop](https://arxiv.org/abs/1909.11556) (Fan et al., 2019): inicializar um modelo
  menor removendo blocos de um maior acelera o treino. É como o aluno A nasce.
- [Born-Again Neural Networks](https://arxiv.org/abs/1805.04770) (Furlanello et al., 2018):
  alunos destilados podem superar o professor.
- [Does Knowledge Distillation Really Work?](https://arxiv.org/abs/2106.05945) (Stanton et
  al., 2021): o aluno raramente copia o professor com fidelidade, e copiar mais de perto nem
  sempre generaliza melhor.

O que medimos ([destilação](../experimentos/2026-09-23_destilacao.md)): o aluno A empata com
o professor no estilo da organização, onde o professor tem confiança acima de 0,98, e o supera
em ementas reais (+0,022 no teste intocado), o padrão que os dois últimos papers descrevem. O
controle sem destilação perde 0,23 no texto real: sem as probabilidades do professor, o aluno
aprende o rótulo mas não a incerteza.

- [Vocabulary Trimming](https://arxiv.org/abs/2305.15020) (Ushio et al., 2023) e
  [Load What You Need](https://arxiv.org/abs/2010.05609) (Abdaoui et al., 2020): cortar o
  vocabulário de um modelo multilíngue para uma língua preserva o desempenho e reduz muito o
  tamanho. Aqui os embeddings são 64% dos parâmetros e nossos textos usam 7,1% do
  vocabulário, mas todo documento do estresse traz token que o treino nunca viu: a economia
  existe, a garantia de saída idêntica não.

## 7. Quantização e aceleração de decodificação

- [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., 2025) e
  [PolarQuant](https://arxiv.org/abs/2502.02617) (Han et al., 2025): quantização de vetores do
  KV cache e de busca vetorial, com rotação aleatória e quantizador escalar (o TurboQuant
  preserva a qualidade com 3,5 bits por canal). Encoder que lê cada janela uma vez não tem KV
  cache.
- [QuaRot](https://arxiv.org/abs/2404.00456) e [SpinQuant](https://arxiv.org/abs/2405.16406):
  rotações antes de quantizar pesos, contra valores extremos. Poderiam salvar o int8 que
  mudou 16 de 68 documentos aqui, ao custo de adaptar LayerNorm, GLU e RoPE.
- [Multi-token Prediction](https://arxiv.org/abs/2404.19737) (Gloeckle et al., 2024) e
  [Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan et al., 2022): aceleram
  geração autorregressiva. O Gama rotula todos os tokens de uma janela numa passada; não há
  laço de geração para encurtar.

## O que não achamos

Nenhum trabalho no arXiv verifica citações jurídicas brasileiras contra um acervo fechado,
com extração por trecho e classificação em real, inventada e incompleta. Os mais próximos
verificam referências científicas (CiteTracer), medem LLMs gerando citações americanas
(LegalCiteBench) ou classificam decisões brasileiras (LegalBench-BR). A combinação que o
desafio pede, e a que o Gama implementa, é a lacuna.
