# -*- coding: utf-8 -*-
"""O gerador sintético entrega rótulo exato por construção.

Cada teste trava uma propriedade que, quebrada, contamina o goldenset em silêncio:
offset que escorrega, quebra de linha que muda o tamanho do texto, inventada que
existe no acervo (τ no nosso próprio treino), documento que não se reproduz.
"""
import pathlib
import random
import re

import pytest

from gama.indice import construir
from geracao import fichas as F
from geracao import render as R
from geracao.bancos import extrair
from geracao.expandir import val_carregadora, val_enchimento, val_generica
from geracao.gerar import gerar_um
from geracao.montar import Montador, mistura_do_dev, quebrar

RAIZ = pathlib.Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"
DB = str(DADOS / "desafio1_bracis.db")


@pytest.fixture(scope="module")
def mundo():
    idx = construir(DB)
    fichas, sumulas, disps = F.carregar(DB)
    m = Montador(extrair(DADOS), fichas, sumulas, disps, mistura_do_dev(DADOS), idx)
    return idx, m


@pytest.mark.dados
def test_quebra_reproduz_a_organizacao_byte_a_byte():
    """A regra 'termina a linha na palavra que chega ao limite (86-91)' refaz o dev.

    Medido em 22/09: 60 dos 69 parágrafos de prosa do nível 1 saem idênticos. Os 9
    restantes não seguem nenhuma regra simples testada (citação atômica, marcador
    substituído depois) — ficam como resíduo conhecido, não como meta.
    """
    total = iguais = 0
    for arq in sorted((DADOS / "txt").glob("gen_n1_*.txt")):
        texto = arq.read_text(encoding="utf-8")
        for b in re.split(r"\n\s*\n", texto):
            linhas = b.split("\n")
            if len(linhas) < 3 or not all(len(x) >= 80 for x in linhas[:-1]):
                continue                      # cabeçalho não é quebrado
            total += 1
            iguais += any(quebrar(b.replace("\n", " "), t) == b for t in range(86, 96))
    assert total >= 60 and iguais / total >= 0.85, (iguais, total)


def test_quebra_so_troca_espaco_por_quebra():
    s = "palavra " * 60
    q = quebrar(s, 91)
    assert len(q) == len(s)
    assert all(a == b or (a == " " and b == "\n") for a, b in zip(s, q))


@pytest.mark.dados
@pytest.mark.parametrize("semente", range(12))
def test_documento_gerado_passa_nos_portoes(mundo, semente):
    idx, m = mundo
    for nivel in (1, 2):
        doc, laudo = gerar_um(m, semente, 99, nivel, idx)
        assert not laudo.fatal, laudo.fatal
        for a, b, c in doc.spans:
            if nivel == 1:
                assert re.sub(r"\s", " ", doc.texto[a:b]) == c.texto


@pytest.mark.dados
def test_gerador_e_deterministico(mundo):
    idx, m = mundo
    d1, _ = gerar_um(m, 5, 42, 2, idx)
    d2, _ = gerar_um(m, 5, 42, 2, idx)
    assert d1.texto == d2.texto and [(a, b) for a, b, _ in d1.spans] == [(a, b) for a, b, _ in d2.spans]


@pytest.mark.dados
def test_inventada_nunca_existe_no_acervo(mundo):
    """Número inventado com candidato no acervo viraria τ no nosso próprio treino."""
    idx, m = mundo
    rng = random.Random(3)
    from gama.normalizar import nucleo_numerico
    for _ in range(300):
        c = m.citacao("proc_inv", "civel", rng.choice([1, 2]), rng)
        assert not idx.candidatos_processo(nucleo_numerico(c.texto)), c.texto


@pytest.mark.dados
def test_referencia_vaga_nunca_tem_numero_de_processo(mundo):
    idx, m = mundo
    rng = random.Random(4)
    for _ in range(200):
        c = m.citacao("vaga", rng.choice(["civel", "penal", "militar"]), 1, rng)
        sem_ano = re.sub(r"\b(?:19|20)\d{2}\b", "", c.texto)
        assert not re.search(r"\d", sem_ano), c.texto


def test_validadores_da_expansao():
    assert val_carregadora("Vale lembrar que {o} {CIT} consolidou entendimento diverso.")
    assert not val_carregadora("Vale lembrar {o} {CIT} e {o} {CIT}.")               # dois slots
    assert not val_carregadora("Vale lembrar {o} {CIT}, julgado em 2021.")          # número
    assert not val_carregadora("Aplica-se {o} {CIT} e o art. 5º da CF.")             # citação solta
    assert val_enchimento("A questão de fundo comporta solução singela.")
    assert not val_enchimento("Incide a Súmula 7 do STJ na espécie.")
    assert val_generica({"ref": "jurisprudência pacífica desta Corte", "genero": "f", "tipo": "JURIS"})
    assert not val_generica({"ref": "julgado de 2021, Rel. Min. Fux", "genero": "m", "tipo": "JURIS"})
