# Gama: manifesto do modelo

Gama é o extrator de citações da solução, nomeado em homenagem a Luiz Gama
(1830–1882), advogado abolicionista que libertou centenas de pessoas nos tribunais
citando a lei com precisão.

Preenchido a cada versão submetida. `scripts/baixar_pesos.sh` lê as duas primeiras chaves.

repo: vinimlo/gama
revisao: 5f924ca2fa77c2aae6afe6d770c78ca4438f52f3

| Campo | Valor |
|---|---|
| Versão | v1.3 ([D-009](wiki/decisoes/D-009_gama-v1-3-destilado.md)): as 12 primeiras das 22 camadas do v1.2, destiladas do v1.2 |
| Professor | Gama v1.2, `vinimlo/gama` na revisão `ad06ffd34e838bf3496645d9c16281dfc70cb871` (tag `v1.2`) |
| Modelo base do v1.2 | [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base) (MIT), revisão `c5955035435e2bf121cde7f3c8863ef52ff35d82` |
| Tarefa | classificação de tokens BIO (JURIS, LEI, VAGA) |
| Destilação | `treino/destilar.py`: perda `0,5·CE(ouro) + T²·KL(professor/T ‖ aluno/T) + 0,5·KL(professor ‖ aluno)`, T = 2; o professor roda no próprio treino, em FP32 |
| Pesos publicados | https://huggingface.co/vinimlo/gama (revisão acima, tag `v1.3`) |
| Dados de treino | dataset [`vinimlo/gama-goldenset`](https://huggingface.co/datasets/vinimlo/gama-goldenset), revisão `ec430c0373f6ef96ba3bb91a3f11b24e391c5e6a`: subpasta `final_v3` (6.000 docs sintéticos com ouro; 590 de reserva) e 1.818 ementas reais sem rótulo (`reais/amostra.jsonl`, ids em `destilacao/reais_ids.json`; de `celsowm/jurisprudencias_br`, CC-BY-4.0, fora de qualquer teste), onde só vale o termo do professor |
| Semente | 13 |
| Hardware de treino | HF Jobs, 1× A100 80 GB; 3 épocas, max_len 2048, lote 8, lr 5e-5, 18 min |
| Execução | offline, `eval()`, sem amostragem, algoritmos determinísticos do torch |

O v1.2 foi treinado com `treino/treinar.py` no mesmo dataset, revisão
`31474b1f7db9096c4ca4f2a4eae2e9b82852d7a7`, subpasta `final_v3`: 3 épocas, max_len 1024,
lote 8, lr 5e-5, semente 13, 1× A100 80 GB.

## Versões

| Versão | Revisão | O que mudou | Estresse difícil (erros de borda, 4.447 spans) | Sondas |
|---|---|---|---|---|
| v1 | `2aba5d1` | 4.000 docs, org + LLM v1 | 2 | 5/6 (falha "Terna") |
| v1.1 | rejeitada; pesos apagados (treino: `final_v2`, fora da versão atual do dataset e presente na revisão `ec430c0`) | Tema de 1 a 2.999 | 2 | 4/6 (regrediu) |
| v1.2 | `ad06ffd` | ruído dirigido à palavra-chave, 6.000 docs | 0 | 6/6 ("Reclarnação" no limite, IoU 0,5) |
| v1.3 | `5f924ca` | 12 primeiras camadas do v1.2, destiladas dele; JSON final igual ao do v1.2 em 4.626 docs | 0 | 6/6, os mesmos casos do v1.2 |

## Histórico de submissões

| Data | Commit | Revisão dos pesos | Score público |
|---|---|---|---|
| 22/09 | | não se aplica (régua, sem modelo) | 1,09830 (fase de treino, dev) |
| 23/09 | | `vinimlo/gama@ad06ffd` (v1.2) | 1,09999 (fase de treino, dev; local 1,0999975) |
| 23/09 | `077d761` | `vinimlo/gama@5f924ca` (v1.3) | 1,09999 (fase de treino, dev; CSV idêntico ao do v1.2) |

As submissões da fase de treino pontuam contra o dev set e não contam para o ranking. A
submissão final cita o commit deste repositório que produziu as saídas.
