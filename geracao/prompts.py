# -*- coding: utf-8 -*-
"""Prompts da expansão dos bancos. Os exemplos vêm do banco da organização.

Regra comum a todos: o LLM escreve TEXTO SEM CITAÇÃO. Citação só entra pelo
renderizador (rótulo exato por construção); o verificador rejeita qualquer saída
em que a régua encontre citação.
"""
from __future__ import annotations

SISTEMA = ("Você é um redator jurídico brasileiro experiente. Escreve em português do "
           "Brasil, no registro forense formal de petições, pareceres, memoriais e decisões. "
           "Responde sempre em JSON válido, sem comentários.")

ALVO = {
    "JURIS": "um PRECEDENTE (acórdão, recurso, reclamação ou súmula — por exemplo "
             "'o REsp 1.234.567/SP' ou 'a Súmula 211 do STJ')",
    "LEI": "um DISPOSITIVO DE LEI (por exemplo 'o art. 5º, LV, da Constituição Federal' ou "
           "'o art. 373, I, do CPC')",
}

CARREGADORAS = """Preciso de MOLDES de frase que introduzem {alvo} numa peça jurídica.

Em cada molde, o lugar da citação é marcado por um artigo entre chaves seguido de {{CIT}}:
{{o}} {{CIT}}, {{no}} {{CIT}}, {{do}} {{CIT}}, {{ao}} {{CIT}} ou {{pelo}} {{CIT}} — sempre na forma
masculina singular (a concordância é feita depois). Se a frase começar pela citação, use {{O}} {{CIT}}.

Exemplos reais do estilo desejado:
{exemplos}

Regras:
- uma frase completa por molde, terminada em ponto, com 8 a 30 palavras;
- exatamente um {{CIT}}, precedido de {{o}}, {{no}}, {{do}}, {{ao}}, {{pelo}} ou {{O}};
- nenhuma outra citação, número, data, nome de pessoa ou de tribunal com número;
- varie a função retórica: reforço, remissão, analogia, distinção, crítica ao acórdão
  recorrido, fundamento da pretensão, incidência na espécie;
- varie o começo e o fim das frases; não copie os exemplos.

Responda em JSON: {{"moldes": [ ... ]}} com {n} moldes."""

ENCHIMENTO = """Preciso de frases de ENCHIMENTO argumentativo: frases genéricas que aparecem
entre as citações em peças jurídicas brasileiras, sem citar nada.

Exemplos reais:
{exemplos}

Regras:
- uma frase por item, terminada em ponto, com 6 a 22 palavras;
- nada de precedente, lei, artigo, súmula, número, data, nome próprio ou tribunal;
- aplicável a qualquer matéria (cível, penal, trabalhista, eleitoral, militar);
- varie: transição, delimitação da controvérsia, reforço, refutação, conclusão parcial;
- não copie os exemplos.

Responda em JSON: {{"frases": [ ... ]}} com {n} frases."""

GENERICAS = """Preciso de REFERÊNCIAS GENÉRICAS SEM FONTE: expressões que aludem a jurisprudência
ou a legislação SEM identificá-la — sem número, sem ano, sem relator, sem nome de lei.

Exemplos reais (sem o artigo inicial):
{exemplos}

Para cada referência informe:
- "ref": a expressão SEM o artigo inicial (ex.: "jurisprudência pacífica desta Corte");
- "genero": "m" ou "f" (gênero do núcleo, no SINGULAR — não use plural);
- "tipo": "JURIS" se alude a jurisprudência, "LEI" se alude a norma.

Responda em JSON: {{"refs": [{{"ref": "...", "genero": "m", "tipo": "JURIS"}}, ... ]}} com {n} itens."""

NARRATIVAS = """Escreva {n} blocos NARRATIVOS para peças de matéria {materia} ({descricao}).
Cada bloco tem 2 ou 3 frases que relatam os fatos e o andamento de um caso hipotético
(quem são as partes, o que aconteceu, o que decidiu a instância anterior), como nos exemplos:

{exemplos}

Regras:
- nada de precedente, lei, artigo, súmula ou número de processo;
- datas SÓ pelo marcador {{DATA}} (ex.: "em {{DATA}}"); folhas dos autos SÓ por {{FLS}}
  (ex.: "às {{FLS}}"); nenhum outro número;
- sem nomes próprios de pessoas;
- 25 a 80 palavras por bloco; cada bloco sobre um caso diferente.

Responda em JSON: {{"blocos": [["frase 1", "frase 2"], ["frase 1", "frase 2", "frase 3"], ...]}}"""

DESCRICAO_MATERIA = {
    "civel": "direito civil, contratos, consumidor, responsabilidade civil",
    "penal": "direito penal e processual penal comum",
    "trabalhista": "direito do trabalho e processo do trabalho",
    "eleitoral": "direito eleitoral: registro, propaganda, prestação de contas, abuso de poder",
    "militar": "direito penal militar e processo penal militar",
}

PECAS = """Crie {n} MOLDES DE PEÇA do tipo "{tipo}", matéria {materia} ({descricao}).

Cada molde tem:
- "cabecalho": linhas do topo separadas por \\n — órgão ou destinatário, identificação do
  processo com o marcador {{CNJ}} no lugar do número, partes (nomes FICTÍCIOS em caixa-alta);
- "titulo": o título da peça em caixa-alta;
- "abertura": um parágrafo de 25 a 60 palavras que apresenta a peça;
- "secoes": 4 títulos de seção no formato "I — DO ...", "II — DA ...", "III — ...", "IV — ...";
- "fecho": lista com 2 ou 3 blocos — pedido ou conclusão, fórmula final e "Cidade, {{DATA}}.".

Exemplos reais:
{exemplos}

Regras: nenhum número real (use {{CNJ}}, {{OAB}}, {{PROTOCOLO}}, {{DATA}}); nenhuma citação de
precedente ou de lei; varie órgão, cidade, partes e fórmulas.

Responda em JSON: {{"pecas": [ ... ]}}"""

TIPOS_DE_PECA = {
    "civel": ["petição inicial", "contestação", "razões de apelação", "contrarrazões ao recurso especial",
              "agravo interno", "embargos de declaração", "decisão monocrática", "parecer jurídico"],
    "penal": ["habeas corpus", "memorial", "razões de apelação criminal", "recurso especial",
              "parecer do Ministério Público", "agravo regimental", "resposta à acusação"],
    "trabalhista": ["recurso ordinário", "contrarrazões ao recurso de revista", "acórdão de TRT",
                    "embargos de declaração", "agravo de instrumento em recurso de revista", "reclamação trabalhista"],
    "eleitoral": ["parecer da Procuradoria Regional Eleitoral", "recurso especial eleitoral",
                  "representação eleitoral", "agravo regimental", "memorial"],
    "militar": ["parecer da Procuradoria de Justiça Militar", "razões de apelação", "memorial",
                "habeas corpus", "embargos infringentes"],
}
