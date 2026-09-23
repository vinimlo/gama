# Gama: manifesto do modelo

Gama é o extrator de citações da solução, nomeado em homenagem a Luiz Gama
(1830–1882), advogado abolicionista que libertou centenas de pessoas nos tribunais
citando a lei com precisão.

Preenchido a cada versão submetida. `scripts/baixar_pesos.sh` lê as duas primeiras chaves.

repo: vinimlo/gama
revisao: ad06ffd34e838bf3496645d9c16281dfc70cb871

| Campo | Valor |
|---|---|
| Modelo base | [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base) (MIT), revisão `c5955035435e2bf121cde7f3c8863ef52ff35d82` |
| Fine-tune | classificação de tokens BIO (JURIS, LEI, VAGA), `treino/treinar.py` |
| Pesos publicados | https://huggingface.co/vinimlo/gama (revisão acima) |
| Dados de treino | goldenset sintético v3 (`geracao/`, 6.000 docs, ruído dirigido à palavra-chave), dataset `vinimlo/gama-goldenset` (privado), subpasta `final_v3`, revisão `31474b1f7db9096c4ca4f2a4eae2e9b82852d7a7` |
| Semente | 13 |
| Hardware de treino | HF Jobs, 1× A100 80 GB; 3 épocas, max_len 1024, lote 8, lr 5e-5 |
| Execução | offline, `eval()`, sem amostragem, algoritmos determinísticos do torch |

## Versões

| Versão | Revisão | O que mudou | Estresse difícil (erros de borda, 4.447 spans) | Sondas |
|---|---|---|---|---|
| v1 | `2aba5d1` | 4.000 docs, org + LLM v1 | 2 | 5/6 (falha "Terna") |
| v1.1 | (`vinimlo/gama-v1-1@7ea7a67`, rejeitada) | Tema de 1 a 2.999 | 2 | 4/6 (regrediu) |
| v1.2 | `ad06ffd` | ruído dirigido à palavra-chave, 6.000 docs | 0 | 6/6 ("Reclarnação" no limite, IoU 0,5) |

## Histórico de submissões

| Data | Commit | Revisão dos pesos | Score público |
|---|---|---|---|
| 22/09 | | não se aplica (régua, sem modelo) | 1,09830 (fase de treino, dev) |
| 23/09 | | `vinimlo/gama@ad06ffd` (v1.2) | 1,09999 (fase de treino, dev; local 1,0999975) |

As submissões da fase de treino pontuam contra o dev set e não contam para o ranking. A
submissão final cita o commit deste repositório que produziu as saídas.
