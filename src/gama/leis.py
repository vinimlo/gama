# -*- coding: utf-8 -*-
"""Qual lei: a chave do diploma (dígitos do número, 'CF', 'LC64'), lida no cabeçalho de um
dispositivo do acervo ou numa citação."""
from __future__ import annotations

import re

from .normalizar import OCR, OCR_PARA_DIGITO, Normalizador, TabelaOCR

# Apelidos de codigo -> chave da lei (digitos do numero da lei).
# Conjunto fechado: os 13 dispositivos do acervo cobrem 9 diplomas.
APELIDOS_LEI = {
    "cpc": "13105", "codigo de processo civil": "13105", "lei 13105": "13105",
    "cc": "10406", "codigo civil": "10406",
    "clt": "5452", "consolidacao das leis do trabalho": "5452",
    "cpp": "3689", "codigo de processo penal": "3689",
    "cpm": "1001", "codigo penal militar": "1001",
    "cdc": "8078", "codigo de defesa do consumidor": "8078",
    "codigo eleitoral": "4737",
    "cf": "CF", "constituicao federal": "CF", "constituicao da republica": "CF",
    "constituicao": "CF", "carta magna": "CF", "cf/88": "CF",
    "lc 64": "LC64", "lei complementar 64": "LC64",
    "lei complementar n 64": "LC64", "lc 64/1990": "LC64",
}

_OCR = "".join(re.escape(c) for c in sorted(OCR_PARA_DIGITO))
_LEI_EXPL = re.compile(
    # O número da lei aceita qualquer letra da tabela de OCR ("131O5/2015", "473T/1965") e
    # espaço depois do ponto ("13. 105"); vai para TabelaOCR.numero, nunca direto. Entre "Lei" e
    # o número cabe UMA palavra qualquer: "Complementar" ruidoso ("Cornplernentar") não
    # pode derrubar a leitura — o tipo complementar sai do esqueleto, não da grafia.
    r"\bl\s*[eéêc]\s*[iíl1]\s*(?:(?P<pula>(?=\S*[^\W\d_]{3})\S{4,})\s+)?n?[ºo°.\s]*"
    r"(?P<num>[\d" + _OCR + r"][\d" + _OCR + r".\s]{0,10}?)\s*(?:/|,|;|\bd[eao]\b|$)", re.I)


class IdentificadorDeLei:
    """A chave da lei a partir do texto, com os apelidos de código ('CPC' -> '13105')."""

    def __init__(self, apelidos: dict = APELIDOS_LEI, ocr: TabelaOCR = OCR):
        self.apelidos = apelidos
        self.ocr = ocr
        # apelidos mais longos primeiro, senão 'cc' casa dentro de outra palavra.
        # Pelo esqueleto: "Consolldacao das Leis do Trabalho" ainda é a CLT.
        self._por_esqueleto = [
            (re.compile(r"\b" + re.escape(Normalizador.esqueleto(a)) + r"\b"), apelidos[a])
            for a in sorted(apelidos, key=len, reverse=True)]

    @staticmethod
    def do_cabecalho(descricao: str) -> str:
        """Cabeçalho de dispositivo do acervo: 'Lei n 13.105' -> '13105';
        'Constituicao Federal de 1988' -> 'CF'."""
        plano = Normalizador.achatar(descricao)
        if "constitui" in plano:
            return "CF"
        if "complementar" in plano:
            return "LC" + re.sub(r"\D", "", plano.split("complementar")[1][:8])
        digitos = re.sub(r"\D", "", plano.split(",")[0])
        return digitos.lstrip("0") or digitos

    def da_citacao(self, trecho: str) -> str | None:
        """'art. 373, I, do CPC' -> '13105'; 'da Lei n 13.105/2015' -> '13105'."""
        plano = Normalizador.achatar(trecho)
        esq = Normalizador.esqueleto(trecho)
        # Sobre o texto ORIGINAL: achatar() minusculiza, e no OCR a caixa carrega
        # informação ("807B" é 8078; "807b" seria 8076).
        m = _LEI_EXPL.search(trecho)
        # A palavra pulada entre "Lei" e o número só vale se for "Complementar" (ruidoso ou
        # não): "Lei Municipal nº 8.078" virando o CDC federal seria falso real — τ
        # (revisão independente, rodada 3, achado 2).
        if m and m.group("pula") and "omplementar" not in Normalizador.esqueleto(m.group("pula")):
            return None
        if m:
            digitos = self.ocr.numero(re.sub(r"[.\s]", "", m.group("num"))) or ""
            if "omplementar" in esq:
                return "LC" + digitos
            if digitos:
                return digitos.lstrip("0") or digitos
        m = re.search(r"\blc\s*n?[º°o.\s]*(\d+)", plano)       # "LC nº 64/1990", "LC 64/90"
        if m:
            return "LC" + m.group(1)
        if "eonstltul" in esq or "earta magna" in esq:
            return "CF"
        for padrao, chave in self._por_esqueleto:
            if padrao.search(esq):
                return chave
        return None
