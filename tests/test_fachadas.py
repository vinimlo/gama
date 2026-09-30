# -*- coding: utf-8 -*-
"""Nomes antigos que o laboratório (bench/, treino/, geracao/, avaliacao/) ainda importa, vivos até a leva 8.

Cada fachada só delega: aqui se confere que ela chega à classe com os mesmos limites.
Sai inteiro junto com as fachadas, quando os consumidores migrarem.
"""
from gama import cabecalho, classificar, formas, indice, normalizar, pipeline, resolver, span
from gama.cabecalho import CadeiaDeClasse, LeitorDeNumeroProprio
from gama.extratores import carregar, guarda
from gama.extratores.regua import ExtratorRegua
from gama.leis import IdentificadorDeLei
from gama.normalizar import OCR, Normalizador, NumeroDeProcesso
from gama.resolver import ClasseProcessual, Resolvedor
from gama.span import Span


def _em(inicio, fim, forma="processo", conf=1.0):
    return Span(inicio, fim, "x" * (fim - inicio), "jurisprudencia", forma, "", conf)


def test_span():
    a, b = _em(0, 10), _em(12, 20)
    assert (span.cruza(a, b), span.distancia(a, b)) == (False, 2)
    t = " REsp. "
    s = Span(0, 7, t, "jurisprudencia", "processo")
    assert span.aparar(s, t) == s.aparado(t) == pipeline.aparar(s, t)
    assert span.aparar_todos([s], t) == Span.aparar_todos([s], t)


def test_guarda(monkeypatch):
    fraco, vaga = _em(0, 10, conf=0.7), _em(12, 30, forma="vaga")
    regua = [_em(0, 12, conf=None)]
    assert guarda.confiante(fraco, 0.5) and not guarda.confiante(fraco, 0.95)
    assert guarda.sem_vaga_colada([fraco, vaga], 2) == [fraco]
    assert guarda.sem_vaga_colada([fraco, vaga], 1) == [fraco, vaga]
    assert guarda.trocar_fracos_pela_regua([fraco], regua, 0.95) == regua
    assert guarda.trocar_fracos_pela_regua([fraco], regua, 0.5) == [fraco]
    assert guarda.guardar([fraco], regua) == regua
    monkeypatch.setattr(guarda, "CONFIANCA_MINIMA", 0.5)
    assert guarda.guardar([fraco], regua) == [fraco]


def test_carregar():
    assert isinstance(carregar("regua"), ExtratorRegua)


def test_normalizar():
    t = "AgInt no REsp 1.45g.779/SP Códig0"
    assert normalizar.sem_acento(t) == Normalizador.sem_acento(t)
    assert normalizar.achatar(t) == Normalizador.achatar(t)
    assert normalizar.esqueleto(t) == Normalizador.esqueleto(t)
    assert normalizar.so_digitos(t) == OCR.so_digitos(t)
    assert normalizar.chave_processo("0001.459") == NumeroDeProcesso.do_bruto("0001.459").chave == "1459"
    assert normalizar.nucleo_numerico(t) == NumeroDeProcesso.do_trecho(t).chave == "1459779"


def test_cabecalho():
    texto = "AgRg no RECURSO ESPECIAL Nº 1.205.500 - SC (2010⁄0146585-7)"
    assert cabecalho.cadeia_de_classe("AgRg no REsp") == CadeiaDeClasse.ler("AgRg no REsp")
    assert cabecalho.numero_proprio(texto, "STJ") == LeitorDeNumeroProprio().ler(texto, "STJ")
    assert cabecalho.Proprio is cabecalho.NumeroProprio


def test_indice(tmp_path):
    import sqlite3
    db = tmp_path / "a.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE documentos (id TEXT, natureza TEXT, tribunal TEXT, ano INTEGER, "
                "relator TEXT, texto TEXT)")
    con.execute("INSERT INTO documentos VALUES ('a', 'acordao', 'STJ', 2020, 'R', "
                "'AgRg no RECURSO ESPECIAL Nº 1.205.500 - SC')")
    con.execute("INSERT INTO documentos VALUES ('s', 'sumula', 'STJ', 2020, 'R', 'Súmula n. 7 do STJ')")
    con.execute("INSERT INTO documentos VALUES ('d', 'dispositivo', NULL, NULL, NULL, "
                "'Artigo 5 da Constituição Federal de 1988 Art. 5.')")
    con.commit()
    con.close()
    idx = indice.construir(db)
    novo = indice.Indice.do_banco(db)
    assert (idx.processos.por_chave, idx.sumulas, idx.meta) == (novo.processos.por_chave, novo.sumulas, novo.meta)
    assert idx.por_processo == {"1205500": ["a"]} and idx.cadeia == {"a": ("AgR", "REsp")}
    assert idx.resolve_sumula("7", "STJ", False) == ["s"]
    assert idx.resolve_dispositivo("5", "CF") == ["d"]
    assert indice._chave_lei("Lei nº 13.105") == IdentificadorDeLei.do_cabecalho("Lei nº 13.105")
    assert indice.APELIDOS_LEI["cpc"] == "13105"


def test_indice_aceita_dicionarios_simples():
    idx = indice.Indice(sumulas={("7", "STJ", False): "s"}, dispositivos={("5", "CF"): "d"})
    assert idx.sumulas.resolver("7", None, False) == ["s"] and idx.dispositivos.resolver("5", "CF") == ["d"]


def test_resolver():
    idx = indice.Indice(sumulas={("7", "STJ", False): "s"})
    sp = Span(0, 16, "Súmula 7 do STJ", "jurisprudencia", "sumula")
    assert resolver.resolver(sp, idx) == Resolvedor(idx).resolver(sp)
    assert resolver._numero_ocr("2l1") == OCR.numero("2l1")
    assert resolver._chave_lei_da_citacao("art. 5 da CF") == IdentificadorDeLei().da_citacao("art. 5 da CF")
    assert resolver.TRIBUNAL_DA_CLASSE is ClasseProcessual.TRIBUNAL_DA_CLASSE
    assert resolver.CLASSE_POR_EXTENSO is ClasseProcessual.POR_EXTENSO
    assert resolver._ART_NUM.search("art. 5").group(1) == "5"


def test_classificar_e_formas(monkeypatch):
    sp = formas.span_de("Ver REsp 1/SP", 4, 13, "JURIS", 0.99)
    assert sp == formas.DetectorDeForma().span("Ver REsp 1/SP", 4, 13, "JURIS", 0.99)
    assert formas.forma("Súmula 7") == formas.DetectorDeForma().forma("Súmula 7") == "sumula"
    assert classificar.faixa(0.99) == classificar.TabelaDeConfianca.faixa(0.99) == "alta"
    res = resolver.Resolucao(["1"], "processo")
    assert classificar.classificar(sp, res) == classificar.Classificador().classificar(sp, res)
    monkeypatch.setattr(classificar, "TABELA", {"processo|real|alta": 0.12})
    assert classificar.classificar(sp, res).confianca == 0.12                    # lida a cada chamada


def test_pipeline(tmp_path, monkeypatch):
    arq = tmp_path / "d.txt"
    arq.write_bytes("REsp 1\r\n".encode())
    assert pipeline.ler_texto(arq) == pipeline.Documento.ler(arq).texto
    a, b = _em(0, 10), _em(5, 15)
    assert pipeline._iou(a, b) == a.iou(b)
    assert pipeline.sem_sobreposicao([a, _em(0, 9)]) == pipeline.Pipeline.sem_sobreposicao([a, _em(0, 9)]) == [a]
    idx = indice.Indice()
    assert pipeline.processar("art. 1 da CF", idx) == pipeline.Pipeline(idx).processar("art. 1 da CF")
    chamado = []
    monkeypatch.setattr(pipeline.Aplicacao, "executar", lambda self, argv=None: chamado.append(argv) or 7)
    assert pipeline.main(["x"]) == 7 and chamado == [["x"]]
