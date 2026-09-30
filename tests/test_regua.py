# -*- coding: utf-8 -*-
"""A régua peça por peça: cabeçalho, fronteira de sentença, extensão do prefixo e da UF,
referências vagas, distratores e a ordem de registro.

Os documentos começam com um cabeçalho curto terminado em linha tripla, para o corpo não
cair no cabeçalho inferido. Os casos contra o dev set estão em test_convencoes.py (dados).
"""
import pytest

from gama.extrair import (_e_sigla, _estende_direita, _estende_esquerda, _fim_cabecalho, _fronteira_sentenca,
                          _tem_classe, _vagas, extrair)

CAB = "CABEÇALHO\n\n\n"


def _trechos(corpo):
    return [(s.trecho, s.tipo, s.forma) for s in extrair(CAB + corpo)]


# ------------------------------------------------------------------ cabeçalho

@pytest.mark.parametrize("texto,fim", [
    ("abc\n\ndef\n\n\nghi" + "x" * 400, 8),         # a linha tripla vence a dupla
    ("abc\n\ndef" + "x" * 400, 3),
    ("a" * 50, 80),                                  # sem linha em branco: piso de 80
    ("a" * 1000, 250),                               # um quarto do texto
    ("a" * 4000, 700),                               # teto de 700
    ("x" * 500 + "\n\n" + "y" * 3000, 500),
    ("x" * 300 + "\n\n" + "y" * 100, 100),          # linha em branco depois do teto: teto
])
def test_fim_do_cabecalho(texto, fim):
    assert _fim_cabecalho(texto) == fim


# ------------------------------------------------------------------ prefixo

@pytest.mark.parametrize("bruto,sigla", [
    ("REsp", True), ("TST-ED-E-ED-RR", True), ("AgR-REspe", True), ("STJ", True), ("RR.", True),
    ("AbCdefghij", False),                           # maiúsculas de menos para o tamanho
    ("a", False), ("parte", False), ("A", False), ("Ab", False), ("abc", False), ("12", False),
])
def test_e_sigla(bruto, sigla):
    assert _e_sigla(bruto) == sigla


@pytest.mark.parametrize("janela,pos", [
    ("fim da frase. Começo", 14),
    ("no Rec. Esp. n. ", -1),                        # ponto de abreviação não encerra
    ("Rel. Min. Xavier; e o ", 18),
    ("Rel. Min. X; e o ", -1),                       # "X" de uma letra conta como abreviação
    ("a. b. ", -1),                                  # palavra de até 2 letras: abreviação
    ("Dito isso: o ", 11),
    ("primeiro\n\nsegundo", 10),                     # linha dupla sempre encerra
    ("x.\n\nno ", 4),
    ("sem fronteira", -1),
])
def test_fronteira_de_sentenca(janela, pos):
    assert _fronteira_sentenca(janela) == pos


def _esquerda(texto, numero, limite=80):
    return texto[_estende_esquerda(texto, texto.index(numero), limite):texto.index(numero)]


@pytest.mark.parametrize("texto,prefixo", [
    ("Cita-se o AgInt no EDcl no REsp nº 123", "AgInt no EDcl no REsp nº "),
    ("Observa-se o AgRg no Rec. Esp. n. 123", "AgRg no Rec. Esp. n. "),
    ("Conforme a TST-ED-E-ED-RR-123", "TST-ED-E-ED-RR-"),
    ("Julgado. REsp 123", "REsp "),                          # fronteira de sentença
    ("a parte autora 123", ""),
    ("o Agravo Regimental no Recurso Especial 123", "Agravo Regimental no Recurso Especial "),
    ("o REsp (ver) 123", ""),
    ("o REsp !!!! 123", ""),                                 # lixo demais entre o prefixo e o número
    ("o REsp !! 123", "REsp !! "),
    ("Ver o processo. AgInt 123", "AgInt "),                 # "processo" é vocabulário, mas a frase acabou
])
def test_estende_esquerda(texto, prefixo):
    assert _esquerda(texto, "123") == prefixo


def test_estende_esquerda_respeita_o_limite():
    texto = "o Agravo Regimental no Recurso Especial 123"
    assert _esquerda(texto, "123", limite=17) == "Recurso Especial "


@pytest.mark.parametrize("resto,uf", [
    ("/SP, no ponto", "/SP"), (" - PR.", " - PR"), (" (RS)", " (RS)"), ("– MG", "– MG"),
    (" e o", ""), ("/sp", ""), ("/SPX", "/SP"),
])
def test_estende_direita(resto, uf):
    texto = "REsp 1" + resto
    assert texto[6:_estende_direita(texto, 6)] == uf


@pytest.mark.parametrize("prefixo,classe", [
    ("Processo nº ", False), ("Processo nº TST-RR-", True), ("AgInt no ", True), ("n. ", False), ("", False),
])
def test_tem_classe(prefixo, classe):
    assert _tem_classe(prefixo) == classe


# ------------------------------------------------------------------ vagas

def _vaga(texto):
    return [texto[a:b] for a, b in _vagas(texto, _fim_cabecalho(texto))]


def test_vaga_da_abertura_ao_relator():
    t = CAB + "Conforme precedente do STF de 2020, da relatoria de Cármen Lúcia, no ponto."
    assert _vaga(t) == ["precedente do STF de 2020, da relatoria de Cármen Lúcia"]


def test_vaga_pega_a_ultima_abertura_antes_do_ano():
    t = CAB + "Precedente citado; julgado do STJ de 2021, Rel. Min. Nancy Andrighi."
    assert _vaga(t) == ["julgado do STJ de 2021, Rel. Min. Nancy Andrighi"]


@pytest.mark.parametrize("corpo", [
    "Conforme precedente do STF, da relatoria de Cármen Lúcia.",                # sem ano
    "Conforme o de 2020, da relatoria de Cármen Lúcia.",                         # sem abertura
    "Conforme precedente 1.234.567 de 2020, da relatoria de Cármen Lúcia.",     # tem número
    "Conforme precedente 0600316-49.2020.6.16.0182, da relatoria de Cármen Lúcia.",
    "Precedente. Em 2020, da relatoria de Cármen Lúcia.",                      # a abertura ficou na frase anterior
])
def test_nao_e_vaga(corpo):
    assert _vaga(CAB + corpo) == []


def test_relator_do_proprio_documento_no_cabecalho_nao_e_vaga():
    t = "ACÓRDÃO 2020\nRelator: Min. Fulano de Tal\n\n\nCorpo do documento."
    assert _vaga(t) == []


def test_relatoria_no_comeco_de_linha_do_corpo_ainda_e_vaga():
    t = CAB + "Conforme julgado do STJ de 2021,\nda relatoria de Nancy Andrighi, no ponto."
    assert _vaga(t) == ["julgado do STJ de 2021,\nda relatoria de Nancy Andrighi"]


# ------------------------------------------------------------------ extrair

def test_distratores_bloqueiam():
    assert _trechos("Ver fls. 123/456 e OAB/SP nº 12345 e R$ 1.234,56 no ponto. O REsp 1.234.567/SP.") == [
        ("REsp 1.234.567/SP", "jurisprudencia", "processo")]
    assert _trechos("Ver CPF 123.456.789-00 e o REsp 1.234.567/SP.") == []       # CPF leva 30 caracteres


def test_sumula_e_tema():
    assert _trechos("A Súmula 211 do STJ, a Súmula Vinculante nº 10, o Tema 725 da repercussão geral.") == [
        ("Súmula 211", "jurisprudencia", "sumula"), ("Súmula Vinculante nº 10", "jurisprudencia", "sumula"),
        ("Tema 725 da repercussão geral", "jurisprudencia", "tema")]


def test_artigos_com_alinea_paragrafo_e_lei():
    assert _trechos('Nos termos do art. 1º, I, "g", da LC 64/1990, e do art. 373, § 1º, do CPC; '
                    'e art. 5º da Lei n 13.105/2015.') == [
        ('art. 1º, I, "g", da LC 64/1990', "lei", "artigo"), ("art. 373, § 1º, do CPC", "lei", "artigo"),
        ("art. 5º da Lei n 13.105/2015", "lei", "artigo")]


def test_processos_cnj_e_uf():
    spans = extrair(CAB + "Conforme o RR-1835-06.2010.5.15.0042 e o AgInt no EDcl no REsp nº 21737l8 - SP "
                          "e a Rcl 88.178 (RS).")
    assert [(s.trecho, s.forma, s.digitos) for s in spans] == [
        ("RR-1835-06.2010.5.15.0042", "cnj", "18350620105150042"),
        ("AgInt no EDcl no REsp nº 21737l8 - SP", "processo", "2173718"),
        ("Rcl 88.178 (RS", "processo", "88178")]                               # o ")" sai com o lixo de borda


def test_cnj_cru_e_numero_curto_ou_ano_solto():
    assert _trechos("Processo 0600316-4920206160182 e em 2024 e nº 123 e o 5.432 no ponto.") == [
        ("Processo 0600316-4920206160182", "jurisprudencia", "cnj"), ("5.432", "jurisprudencia", "processo")]


def test_numero_dos_autos_no_cabecalho_sem_classe_e_distrator():
    t = "TRIBUNAL\nProcesso nº 8133385-26.2020.5.05.4913\nTST-RR-79500-10.2019.5.01.0001\n\n\nCorpo."
    assert [s.trecho for s in extrair(t)] == ["TST-RR-79500-10.2019.5.01.0001"]


def test_vaga_registra_antes_do_numero():
    """O ano da vaga ("de 2021") não pode virar número de processo."""
    assert _trechos("Conforme o REsp 1.234.567, julgado em 2021, Rel. Min. Nancy Andrighi, no ponto.") == [
        ("REsp 1.234.567", "jurisprudencia", "processo"),
        ("julgado em 2021, Rel. Min. Nancy Andrighi", "jurisprudencia", "vaga")]


def test_sumula_registra_antes_do_numero():
    """A súmula lê até 4 dígitos e para no ponto; o número inteiro já não entra como processo."""
    assert _trechos("Súmula 1.234 do STJ.") == [("Súmula 1", "jurisprudencia", "sumula")]


def test_saida_ordenada_e_sem_sobreposicao():
    texto = CAB + "O REsp 1.234.567/SP e a Súmula 7 e o art. 5º da CF. Conforme julgado de 2020, Rel. Min. Xavier Yuri."
    spans = extrair(texto)
    assert len(spans) == 4
    assert [s.inicio for s in spans] == sorted(s.inicio for s in spans)
    assert all(a.fim <= b.inicio for a, b in zip(spans, spans[1:]))
    assert all(s.trecho == texto[s.inicio:s.fim] for s in spans)


def test_texto_sem_citacao():
    assert extrair("") == [] and extrair(CAB + "Nada a citar aqui.") == []
