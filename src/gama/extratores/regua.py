# -*- coding: utf-8 -*-
"""A régua: o extrator por regras do baseline.

Não é a solução — é o número que o extrator neural precisa bater (1,09848 no dev
set, ajustado a ele; falha em 5 de 6 sondas de ruído documentado).

Extracao dos spans de citacao.

Estrategia em duas etapas, em vez de enumerar prefixos:

1. acha o NUCLEO (CNJ, numero de processo, sumula, artigo) ou a ANCORA
   (o relator, no caso das referencias vagas);
2. ESTENDE A ESQUERDA por vocabulario de citacao ("AgInt no EDcl no REsp n"),
   o que cobre cadeias arbitrarias sem precisar lista-las;
3. estende a direita para pegar a UF (/PR, - PR, (PR)).

Enumerar prefixo a prefixo quebraria no conjunto cego: a amostra de
desenvolvimento ja traz 96 formas distintas em 192 citacoes.

O fonte e mantido em ASCII: os caracteres nao-ASCII entram por escape
\\uXXXX (que o modulo `re` interpreta) ou por chr(), para o arquivo
sobreviver a qualquer pipeline de edicao.
"""
from __future__ import annotations

import re

from ..normalizar import Normalizador, OCR
from ..span import Span
from .base import Extrator


class PadroesDaRegua:
    """Vocabulário e expressões regulares da régua."""

    VOCAB_PREFIXO = {
        "no", "nos", "na", "nas", "em", "de", "do", "da", "e", "n", "no.",
        "resp", "respe", "aresp", "arespel", "agint", "agrg", "agr", "ag",
        "agresp", "agarr", "rcl", "recl", "reclamacao", "re", "rhc", "hc",
        "apl", "rse", "rr", "arr", "airr", "ed", "eds", "edcl", "rms", "ms",
        "ar", "rp", "ai", "ede", "sl", "sls",
        "agravo", "interno", "regimental", "recurso", "especial", "eleitoral",
        "embargos", "declaracao", "habeas", "corpus", "mandado", "seguranca",
        "suspensao", "liminar", "sentenca", "instrumento", "processo", "terceiro",
        "primeiro", "segundo", "quarto", "reg", "int", "esp", "rec", "acordao",
        "peticao", "acao", "rescisoria", "conflito", "competencia", "inquerito",
    }

    TOKEN = re.compile(r"[\wºª°.\-]+", re.UNICODE)

    # ------------------------------------------------------------------ nucleos
    # Letras que o OCR usa no lugar de digitos. A organizacao garante que um
    # digito NUNCA vira outro digito, so letra vira digito -- por isso da para
    # normalizar sem risco de trocar a identidade da citacao.
    DIGITOIDE = r"[0-9OoDQlIiZzASsGbTBgq]"
    SEP0 = r"[-.\s ]*"   # separador opcional
    SEP1 = r"[-.\s ]+"   # separador obrigatorio (>=1)

    # CNJ: NNNNNNN-DD.AAAA.J.TR.OOOO, tolerando quebra de linha e hifen duplicado.
    CNJ = re.compile(
        r"\d{1,7}" + SEP0 + r"-" + SEP0 + r"\d{2}" + SEP1 + r"\d{4}" + SEP1
        + r"\d" + SEP1 + r"\d{2}" + SEP1 + r"\d{4}"
    )
    CNJ_CRU = re.compile(r"\d{7}\s*-\s*\d{13}\b")
    PROCESSO = re.compile(
        r"\d" + DIGITOIDE + r"{0,2}(?:" + SEP1 + DIGITOIDE + r"{3})+(?![.\d])"
        r"|\d" + DIGITOIDE + r"{3,}"
    )
    ANO_SOLTO = re.compile(r"^(?:19|20)\d{2}$")

    SUMULA = re.compile(
        r"\b[s5S]\s*[uúÚ]\s*m(?:ula|\.)?\s*(?:vinculante\s*)?"
        r"n?[ºo°.\s]{0,3}\d{1,4}\b",
        re.I,
    )
    TEMA = re.compile(
        r"\bTem[aãáâ]\s*n?[ºo°.\s]{0,3}\d[\d.]{0,6}"
        r"(?:\s+d[ae]\s+repercuss[aã]o\s+geral)?", re.I)

    # Alinea entre aspas: art. 1o, I, 'g', da LC 64/1990. Classe montada com
    # chr() para nao gravar aspas tipograficas no fonte.
    _ASPAS = "".join(chr(c) for c in (0x27, 0x22, 0x2018, 0x2019, 0x201C, 0x201D))
    _ALINEA = "[" + re.escape(_ASPAS) + r"]\w[" + re.escape(_ASPAS) + "]"
    ARTIGO = re.compile(
        r"\bart(?:igo|\.|\b)\s*\d{1,4}(?:\.\d{3})?\s*(?:[ºo°])?"
        r"(?:\s*,\s*(?:[IVXLC]+|§\s*\d+[ºo°]?(?:-[A-Z])?|caput"
        r"|inciso\s+[IVXLC]+|al[ií]nea\s+\w|" + _ALINEA + r"))*"
        # Cauda "da Lei n 13.105/2015": precisa aceitar ponto DENTRO do numero da
        # lei, senao o extrator corta em "Lei n 13" e o "105" vira citacao propria.
        r"(?:\s*,?\s*d[oae]s?\s+[\wÀ-ÿ.º°/\s-]{2,60}?"
        r"(?=\s*[,;)]|\.\s|\.$|\n\n|$))?",
        re.I,
    )

    # ------------------------------------------------------------------ vagas
    TRIBUNAL = r"(?:STF|STJ|TSE|TST|STM)"
    _NOME = (r"[A-ZÀ-Ý][A-Za-zÀ-ÿ]+"
             r"(?:\s+(?:d[aeo]s?\s+)?[A-ZÀ-Ý][A-Za-zÀ-ÿ]+){0,3}")
    RELATOR = re.compile(
        r"(?:Rel\.?\s*Min\.?|(?:sob|pela|da|de)\s+relatoria|relator[ae]?)"
        r"\s*:?\s*(?:d[eoac]\s+)?(" + _NOME + r")",
        re.I,
    )
    ANO = re.compile(r"\b(?:19|20)\d{2}\b")
    ABERTURA = re.compile(
        r"\b(?:julgad[oa]|precedente|ac[oó]rd[aã]o|ementa|decis[aã]o"
        r"|Reclama[cç][aã]o|Rcl|APL|RSE|RE|HC|RHC|REsp|AREsp"
        r"|Recurso\s+em\s+Habeas\s+Corpus|Agravo\s+em\s+Recurso\s+Especial"
        r"|Recurso\s+Especial)\b",
        re.I,
    )

    # ------------------------------------------------------------------ distratores
    DISTRATOR = re.compile(
        r"fls?\.\s*\d+[/\d\s-]*"
        r"|OAB[/\s]*[A-Z]{2}[\s.nº°]*\d+"
        r"|R\$\s*[\d.,]+"
        r"|\bprotocolo\b[^\n]{0,40}"
        r"|\bCPF\b[^\n]{0,30}|\bCNPJ\b[^\n]{0,30}",
        re.I,
    )

    LIXO_BORDA = " .,:-()nN–—º° "

    # Abreviacoes cujo ponto NAO encerra sentenca. Sem esta lista, o corte de
    # fronteira dispara dentro de "AgRg no Rec. Esp. n. 1.522.200" e decepa o
    # prefixo inteiro da citacao -- foi o que derrubou o recall de 89% para 80%.
    ABREVIACAO = {
        "rec", "esp", "min", "rel", "art", "n", "no", "nos", "ed", "edcl", "ag",
        "int", "reg", "des", "dr", "sr", "sra", "proc", "fls", "cf", "lc", "s",
        "h", "c", "hc", "re", "resp", "aresp", "rcl", "recl", "ap", "apl", "ar",
        "ex", "vol", "pag", "p", "j", "dj", "dje", "un", "obs",
    }
    PONTO_SENTENCA = re.compile(r"[.;:]\s")

    UF_DIREITA = re.compile(r"\s*[/\-–—(]\s*[A-Z]{2}\s*\)?")

    # Tokens que caracterizam uma classe processual (subconjunto de VOCAB_PREFIXO
    # sem os conectores e sem as palavras genericas de cabecalho).
    CLASSES_PROCESSUAIS = VOCAB_PREFIXO - {
        "no", "nos", "na", "nas", "em", "de", "do", "da", "e", "n", "no.",
        "processo", "acordao", "peticao", "primeiro", "segundo", "terceiro",
        "quarto", "int", "reg",
    }


P = PadroesDaRegua


class ExtensorDeCitacao:
    """Leva o núcleo achado até a borda da citação: o prefixo de classe à esquerda
    ("AgInt no EDcl no REsp n") e a UF à direita (/PR, - PR, (PR))."""

    def __init__(self, limite: int = 80):
        self.limite = limite

    @staticmethod
    def e_sigla(bruto: str) -> bool:
        """REsp, TST-ED-E-ED-RR, AgR-REspe sao sigla; 'a' e 'parte' nao sao.

        Sem o piso de 2 caracteres e a exigencia de maiuscula, um 'a' solto
        passava por acronimo e a extensao atravessava a frase inteira.
        """
        limpo = bruto.strip(".-")
        if len(limpo) < 2 or not any(c.isupper() for c in limpo):
            return False
        letras = [c for c in limpo if c.isalpha()]
        return bool(letras) and sum(c.isupper() for c in letras) >= max(2, len(letras) // 2)

    @staticmethod
    def fronteira_sentenca(janela: str) -> int:
        """Posicao logo apos a ultima fronteira REAL de sentenca, ou -1.

        Um ponto depois de abreviacao ("Esp.", "n.", "Min.") nao encerra frase.
        Quebra de linha dupla sempre encerra.
        """
        melhor = -1
        pos = janela.rfind("\n\n")
        if pos != -1:
            melhor = pos + 2
        for m in P.PONTO_SENTENCA.finditer(janela):
            antes = janela[:m.start()]
            palavra = Normalizador.achatar(re.split(r"[^\wÀ-ÿ]+", antes)[-1] if antes else "")
            if palavra in P.ABREVIACAO or (len(palavra) <= 2 and palavra.isalpha()):
                continue                      # abreviacao: o ponto nao encerra
            melhor = max(melhor, m.end())
        return melhor

    def esquerda(self, texto: str, inicio: int) -> int:
        janela_ini = max(0, inicio - self.limite)
        janela = texto[janela_ini:inicio]
        pos = self.fronteira_sentenca(janela)
        if pos != -1:
            janela = janela[pos:]
            janela_ini += pos
        corte = inicio
        for tok in reversed(list(P.TOKEN.finditer(janela))):
            bruto = tok.group(0)
            # Tira ordinal/grau: "nº" e "n°" precisam virar "n", senao a
            # extensao morre no primeiro token e todo prefixo se perde.
            t = Normalizador.achatar(bruto).strip(".-º°ª")
            entre = janela[tok.end():corte - janela_ini]
            if len(entre.strip(P.LIXO_BORDA)) > 2:
                break
            if t in P.VOCAB_PREFIXO or self.e_sigla(bruto):
                corte = janela_ini + tok.start()
            else:
                break
        return corte

    @staticmethod
    def direita(texto: str, fim: int) -> int:
        m = P.UF_DIREITA.match(texto, fim)
        return m.end() if m else fim

    @staticmethod
    def tem_classe(prefixo: str) -> bool:
        plano = Normalizador.achatar(prefixo)
        return any(t.strip(".,-") in P.CLASSES_PROCESSUAIS
                   for t in plano.replace("-", " ").split())


class DetectorDeVagas:
    """Referencias sem numero: ancora no relator, expande a esquerda.

    A forma da classe `incompleta` depois da limpeza final da organizacao e
    sempre a mesma: ABERTURA ... ANO ... RELATOR, sem numero de processo.
    Pegamos a ultima ABERTURA antes do ultimo ANO -- a primeira engoliria o
    paragrafo inteiro.
    """
    JANELA = 130

    def encontrar(self, texto: str, fim_cabecalho: int) -> list:
        saida = []
        for m in P.RELATOR.finditer(texto):
            # "Relator: X" do proprio documento e um ROTULO DE CAMPO: comeca a
            # linha. Uma citacao vaga vem embutida em prosa. Testar isso, e nao a
            # posicao, evita que um documento sem linha em branco -- onde o
            # cabecalho inferido engole o texto todo -- perca todas as vagas.
            # Os DOIS sinais juntos: sozinho, "comeca a linha" derruba citacao
            # legitima cujo "da relatoria de X" caiu no inicio de uma linha
            # quebrada (custou 2 citacoes e 0,009 de score); sozinha, a posicao
            # engole o documento inteiro quando nao ha linha em branco.
            inicio_linha = texto.rfind("\n", 0, m.start()) + 1
            rotulo_de_campo = not texto[inicio_linha:m.start()].strip()
            if rotulo_de_campo and m.start() < fim_cabecalho:
                continue
            janela_ini = max(0, m.start() - self.JANELA)
            janela = texto[janela_ini:m.start()]
            corte = max(janela.rfind(". "), janela.rfind(".\n"), janela.rfind("\n\n"))
            if corte != -1:
                janela = janela[corte + 1:]
                janela_ini += corte + 1
            anos = list(P.ANO.finditer(janela))
            if not anos:
                continue                      # sem ano nao e a forma da classe
            limite = anos[-1].start()
            aberturas = [a for a in P.ABERTURA.finditer(janela) if a.start() <= limite]
            if not aberturas:
                continue
            ini = janela_ini + aberturas[-1].start()
            miolo = texto[ini:m.end()]
            if P.PROCESSO.search(P.ANO.sub(" ", miolo)) or P.CNJ.search(miolo):
                continue                      # tem numero: nao e vaga
            saida.append((ini, m.end()))
        return saida


class Achados:
    """Os spans de um documento, registrados em ordem de prioridade: quem registra
    primeiro bloqueia o resto (`registrar` recusa sobreposição), e os distratores
    bloqueiam desde o início."""

    def __init__(self, texto: str):
        self.texto = texto
        self.bloqueado = [(m.start(), m.end()) for m in P.DISTRATOR.finditer(texto)]
        self.spans: list = []

    @staticmethod
    def _sobrepoe(a, b) -> bool:
        return max(0, min(a[1], b[1]) - max(a[0], b[0])) > 0

    def registrar(self, ini: int, fim: int, tipo: str, forma: str, digitos: str = "") -> None:
        while fim > ini and self.texto[fim - 1] in P.LIXO_BORDA:
            fim -= 1
        if fim <= ini:
            return
        if any(self._sobrepoe((ini, fim), b) for b in self.bloqueado):
            return
        if any(self._sobrepoe((ini, fim), (s.inicio, s.fim)) for s in self.spans):
            return
        self.spans.append(Span(ini, fim, self.texto[ini:fim], tipo, forma, digitos))

    def ordenados(self) -> list:
        return sorted(self.spans, key=lambda s: s.inicio)


class ExtratorRegua(Extrator):
    """Todos os spans de citacao de um documento, sem sobreposicao.

    A ORDEM importa: o mais especifico registra primeiro e bloqueia o resto,
    porque `Achados.registrar` recusa sobreposicao. Vagas antes de numeros, senao o
    ano ("em 2024") e capturado como numero de processo.
    """
    nome = "regua"

    def __init__(self, extensor: ExtensorDeCitacao | None = None, vagas: DetectorDeVagas | None = None):
        self.extensor = extensor or ExtensorDeCitacao()
        self.vagas = vagas or DetectorDeVagas()

    @staticmethod
    def fim_do_cabecalho(texto: str) -> int:
        """Fronteira cabecalho/corpo.

        O numero dos autos e o relator DO PROPRIO documento moram no cabecalho e
        sao distratores; as mesmas formas no corpo sao citacao (secao 2 do enunciado).
        """
        corte = texto.find("\n\n\n")
        if corte == -1:
            corte = texto.find("\n\n")
        # Sem linha em branco o cabecalho nao pode engolir o documento: um texto
        # curto ou mal formatado perderia TODAS as referencias vagas, porque
        # o detector de vagas ignora o que cai no cabecalho.
        teto = min(700, max(80, len(texto) // 4))
        if corte == -1 or corte > teto:
            corte = teto
        return corte

    def extrair(self, texto: str) -> list[Span]:
        fim_cabecalho = self.fim_do_cabecalho(texto)
        achados = Achados(texto)

        for ini, fim in self.vagas.encontrar(texto, fim_cabecalho):
            achados.registrar(ini, fim, "jurisprudencia", "vaga")

        for rx, forma in ((P.SUMULA, "sumula"), (P.TEMA, "tema")):
            for m in rx.finditer(texto):
                achados.registrar(m.start(), self.extensor.direita(texto, m.end()),
                                  "jurisprudencia", forma, OCR.so_digitos(m.group(0)[-6:]))

        for m in P.ARTIGO.finditer(texto):
            achados.registrar(m.start(), m.end(), "lei", "artigo")

        for rx, forma in ((P.CNJ, "cnj"), (P.CNJ_CRU, "cnj"), (P.PROCESSO, "processo")):
            for m in rx.finditer(texto):
                bruto = m.group(0)
                if forma == "processo" and P.ANO_SOLTO.match(bruto.strip()):
                    continue
                digitos = OCR.so_digitos(bruto)
                if len(digitos) < 4:
                    continue
                ini = self.extensor.esquerda(texto, m.start())
                # Numero dos autos do PROPRIO documento, no cabecalho: distrator.
                # O teste e a ausencia de classe processual no prefixo -- "Processo
                # n 8133385-..." e distrator, "Processo n TST-RR-79500-..." nao e.
                if m.start() < fim_cabecalho and not self.extensor.tem_classe(texto[ini:m.start()]):
                    continue
                achados.registrar(ini, self.extensor.direita(texto, m.end()),
                                  "jurisprudencia", forma, digitos)

        return achados.ordenados()
