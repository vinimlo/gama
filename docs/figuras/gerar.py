# -*- coding: utf-8 -*-
"""Desenha as figuras da documentação em SVG, versões clara e escura.

Quatro figuras: o pipeline com um exemplo atravessando as etapas, o que o run.sh faz (com a
fronteira entre rede e sem rede), a linhagem de dados e pesos com as revisões fixas, e a
guarda. Mesma paleta e mesma fonte do gráfico do benchmark (bench/grafico.py), para o README
e o card do modelo lerem como um conjunto só. Só biblioteca padrão.

    make figuras

Os números das figuras são os do README e do MODELO.md. Mudou lá, muda aqui.
"""
from __future__ import annotations

import pathlib
from xml.sax.saxutils import escape

from bench.grafico import FONTE, TEMAS as BASE

PASTA = pathlib.Path(__file__).parent
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
EXTRA = {
    "claro": {"cartao": "#f4f3ee", "realce": "#e3eefb"},
    "escuro": {"cartao": "#242422", "realce": "#1c2f49"},
}
TEMAS = {nome: {**BASE[nome], **EXTRA[nome]} for nome in BASE}
W, M = 880, 28


def abrir(h: int, titulo: str, sub: str, desc: str, t: dict) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" '
            f'role="img" aria-labelledby="t d" font-family="{FONTE}">',
            f'<title id="t">{escape(titulo)}</title>',
            f'<desc id="d">{escape(desc)}</desc>',
            f'<rect x="0.5" y="0.5" width="{W - 1}" height="{h - 1}" rx="10" fill="{t["fundo"]}" '
            f'stroke="{t["borda"]}"/>',
            texto(M, 40, titulo, t["tinta"], 17, 600),
            texto(M, 62, sub, t["tinta2"], 12.5)]


def texto(x: float, y: float, s: str, cor: str, tam: float = 12, peso: int = 400,
          ancora: str = "start", mono: bool = False) -> str:
    fam = f' font-family="{MONO}"' if mono else ""
    anc = f' text-anchor="{ancora}"' if ancora != "start" else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{tam}" font-weight="{peso}" fill="{cor}"'
            f'{anc}{fam}>{escape(s)}</text>')


def caixa(x: float, y: float, w: float, h: float, t: dict, destaque: bool = False,
          tracejada: bool = False, vazia: bool = False) -> str:
    preench = "none" if vazia else (t["realce"] if destaque else t["cartao"])
    borda = t["destaque"] if destaque else t["eixo"]
    traco = ' stroke-dasharray="6 5"' if tracejada else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" fill="{preench}" '
            f'stroke="{borda}" stroke-width="1.2"{traco}/>')


def cartao(x: float, y: float, w: float, h: float, titulo: str, linhas: list[str], t: dict,
           destaque: bool = False) -> list[str]:
    s = [caixa(x, y, w, h, t, destaque), texto(x + 12, y + 24, titulo, t["tinta"], 13, 600)]
    for i, linha in enumerate(linhas):
        s.append(texto(x + 12, y + 44 + i * 15, linha, t["tinta2"], 11))
    return s


def seta(pontos: list[tuple[float, float]], t: dict) -> str:
    """Linha por pontos (só trechos retos), com ponta no último."""
    (xa, ya), (xb, yb) = pontos[-2], pontos[-1]
    p = 5.0
    if abs(xb - xa) >= abs(yb - ya):                      # chega na horizontal
        d = 1 if xb > xa else -1
        ponta = f"{xb},{yb} {xb - d * p * 1.6},{yb - p} {xb - d * p * 1.6},{yb + p}"
    else:                                                 # chega na vertical
        d = 1 if yb > ya else -1
        ponta = f"{xb},{yb} {xb - p},{yb - d * p * 1.6} {xb + p},{yb - d * p * 1.6}"
    caminho = " ".join(f"{x:.1f},{y:.1f}" for x, y in pontos)
    return (f'<polyline points="{caminho}" fill="none" stroke="{t["muda"]}" stroke-width="1.4"/>'
            f'<polygon points="{ponta}" fill="{t["muda"]}"/>')


# ------------------------------------------------------------------ 1. pipeline
ETAPAS = [
    ("1 · Extrator Gama", ["acha cada citação", "e o tipo dela: JURIS,", "LEI ou referência vaga"]),
    ("2 · Guarda", ["onde o modelo hesita", "(confiança < 0,95)", "vale o trecho da régua"]),
    ("3 · Normalização", ["desfaz o ruído de OCR", "só no número:", "l→1, S→5, O→0"]),
    ("4 · Índice do acervo", ["montado do .db recebido,", "a cada execução;", "consulta exata"]),
    ("5 · Classificação", ["1 registro: real", "nenhum: inventada", "sem número: incompleta"]),
]
TRILHA = [  # (marca, trecho, número lido, acervo, saída)
    ("①", "REsp nº l.234.S67/SP", "1234567  (l→1, S→5)", "1 registro", "real, com o doc_id dele"),
    ("②", "HC 987.654/RJ", "987654", "nenhum registro", "inventada"),
    ("③", "julgado do STJ proferido em 2021 pela relatoria de…", "sem número", "não há o que consultar",
     "incompleta"),
]


def pipeline(t: dict) -> str:
    h = 478
    s = abrir(h, "Do texto ao veredito",
              "O modelo só acha onde a citação está. Quem decide se ela existe é o acervo.",
              "Um trecho de documento com três citações passa por cinco etapas: extrator Gama, guarda, "
              "normalização do ruído de OCR, índice do acervo e classificação. A primeira citação, com "
              "ruído de OCR, casa com um registro e sai como real; a segunda não casa com nenhum e sai "
              "como inventada; a terceira não tem número e sai como incompleta.", t)

    # o documento
    s.append(caixa(M, 84, W - 2 * M, 84, t))
    s.append(texto(M + 16, 106, "documento.txt", t["muda"], 11, mono=True))
    dest = f'font-weight="600" fill="{t["tinta"]}" text-decoration="underline"'
    marca = f'font-weight="600" fill="{t["destaque"]}"'
    s.append(f'<text x="{M + 16}" y="130" font-size="13" fill="{t["tinta2"]}">'
             f'(…) na linha do que decidiu o <tspan {dest}>REsp nº l.234.S67/SP</tspan>'
             f'<tspan {marca}> ①</tspan>, e ainda o <tspan {dest}>HC 987.654/RJ</tspan>'
             f'<tspan {marca}> ②</tspan>, além de</text>')
    s.append(f'<text x="{M + 16}" y="152" font-size="13" fill="{t["tinta2"]}">'
             f'<tspan {dest}>julgado do STJ proferido em 2021 pela relatoria de Fulano de Tal</tspan>'
             f'<tspan {marca}> ③</tspan>, que reconhece o direito (…)</text>')
    s.append(seta([(M + 76, 168), (M + 76, 190)], t))

    # as etapas
    larg, vao, y0 = 152, 16, 192
    for i, (titulo, linhas) in enumerate(ETAPAS):
        x = M + i * (larg + vao)
        s += cartao(x, y0, larg, 96, titulo, linhas, t, destaque=(i == 0))
        if i:
            s.append(seta([(x - vao + 1, y0 + 48), (x - 1, y0 + 48)], t))

    # as três citações, etapa por etapa
    y = 322
    colunas = [(M + 24, "Trecho no documento"), (372, "Número lido"), (532, "No acervo"), (692, "Saída")]
    for x, nome in colunas:
        s.append(texto(x, y, nome, t["muda"], 11, 600))
    for i, (m, trecho, numero, acervo, saida) in enumerate(TRILHA):
        yl = y + 28 + i * 30
        s.append(f'<line x1="{M}" y1="{yl - 19}" x2="{W - M}" y2="{yl - 19}" stroke="{t["grade"]}"/>')
        s.append(texto(M, yl, m, t["destaque"], 13, 600))
        s.append(texto(M + 24, yl, trecho, t["tinta"], 12, mono=(i < 2)))
        s.append(texto(372, yl, numero, t["tinta2"], 12))
        s.append(texto(532, yl, acervo, t["tinta2"], 12))
        s.append(texto(692, yl, saida, t["tinta"], 12, 600))
    s.append(texto(M, h - 34, "Exemplo ilustrativo, com números fictícios. Cada saída leva também uma "
                   "confiança: a taxa de acerto medida para aquele tipo de caso.", t["tinta2"], 11))
    s.append(texto(M, h - 18, "Um modelo que decidisse sozinho se a citação existe estaria chutando com "
                   "fluência, que é a alucinação que o desafio quer pegar.", t["tinta2"], 11))
    s.append('</svg>')
    return "\n".join(s) + "\n"


# ------------------------------------------------------------------ 2. execução
def execucao(t: dict) -> str:
    h = 508
    s = abrir(h, "O que o run.sh faz",
              "A rede só entra na preparação. A execução roda com --network none.",
              "Preparação, com rede e uma vez: bash run.sh --preparar constrói a imagem Docker e baixa os "
              "pesos vinimlo/gama na revisão 5f924ca para a pasta modelos. Execução, sem rede: bash run.sh "
              "com o caminho do .db, a pasta de .txt e o arquivo de saída monta o índice a partir do .db "
              "recebido, extrai, aplica a guarda, resolve, grava um JSON por documento e converte para o "
              "CSV da submissão.", t)

    # 1. preparação
    s.append(caixa(M, 84, W - 2 * M, 140, t, tracejada=True, vazia=True))
    s.append(texto(M + 16, 108, "1 · Preparação", t["tinta"], 13, 600))
    s.append(texto(W - M - 16, 108, "com rede, uma vez", t["muda"], 11.5, ancora="end"))
    s.append(texto(M + 16, 131, "bash run.sh --preparar", t["tinta"], 12.5, mono=True))
    s += cartao(M + 16, 146, 388, 64, "Imagem Docker",
                ["Python 3.12, torch 2.14.0 (CUDA 12.6), versões fixadas"], t)
    s += cartao(M + 420, 146, 388, 64, "Pesos do extrator",
                ["vinimlo/gama na revisão 5f924ca, 1,03 GB, em ./modelos"], t)
    s.append(seta([(W / 2, 224), (W / 2, 246)], t))

    # 2. execução
    s.append(caixa(M, 248, W - 2 * M, 232, t, destaque=True, vazia=True))
    s.append(texto(M + 16, 272, "2 · Execução", t["tinta"], 13, 600))
    s.append(texto(W - M - 16, 272, "sem rede (--network none)", t["muda"], 11.5, ancora="end"))
    s.append(texto(M + 16, 295, "bash run.sh <caminho_db> <pasta_txt> <arquivo_saida.csv>", t["tinta"], 12.5,
                   mono=True))

    entradas = [("acervo .db", "o da avaliação, qualquer caminho"), ("pasta com os .txt", "um documento por arquivo"),
                ("./modelos", "os pesos baixados antes")]
    for i, (nome, det) in enumerate(entradas):
        y = 312 + i * 48
        s.append(caixa(M + 16, y, 204, 40, t))
        s.append(texto(M + 28, y + 17, nome, t["tinta"], 12, 600, mono=True))
        s.append(texto(M + 28, y + 32, det, t["muda"], 10.5))
        s.append(seta([(M + 220, y + 20), (M + 250, y + 20)], t))
    s.append(texto(M + 16, 468, "montados só para leitura", t["muda"], 10.5))

    s.append(caixa(M + 252, 312, 320, 136, t, destaque=True))
    s.append(texto(M + 264, 336, "container gama", t["tinta"], 13, 600))
    for i, linha in enumerate(["1. monta o índice a partir do .db recebido", "2. extrai, aplica a guarda e resolve",
                               "3. grava um JSON por documento", "4. converte para CSV com o conversor oficial"]):
        s.append(texto(M + 264, 360 + i * 20, linha, t["tinta2"], 11.5))

    saidas = [("<arquivo_saida>.csv", "o formato da submissão"), ("<arquivo_saida>.json/", "um JSON por documento")]
    for i, (nome, det) in enumerate(saidas):
        y = 324 + i * 64
        s.append(seta([(M + 572, y + 24), (M + 602, y + 24)], t))
        s.append(caixa(M + 604, y, 204, 48, t))
        s.append(texto(M + 616, y + 20, nome, t["tinta"], 12, 600, mono=True))
        s.append(texto(M + 616, y + 36, det, t["muda"], 10.5))

    s.append(texto(M, h - 12, "Usa a GPU se o Docker tiver o runtime NVIDIA. Sem ela roda em CPU, em cerca de "
                   "1 s por documento, dentro do teto de 60 s.", t["tinta2"], 11))
    s.append('</svg>')
    return "\n".join(s) + "\n"


# ------------------------------------------------------------------ 3. linhagem
def linhagem(t: dict) -> str:
    h = 532
    s = abrir(h, "De onde vêm os dados e os pesos",
              "Cada caixa tem revisão fixa. Os dados da organização não são redistribuídos.",
              "Dados: o dev set da organização, que não sai da máquina, é desmontado em bancos de frases; "
              "LLMs abertos só ampliam as frases; o gerador por moldes produz o gama-goldenset, com 6.000 "
              "documentos na pasta final_v3. Modelos: o mmBERT-base na revisão c595503 é ajustado nesses "
              "documentos e vira o Gama v1.2, revisão ad06ffd; o v1.2 é destilado nas 12 primeiras camadas, "
              "com os mesmos documentos e 1.818 ementas reais sem rótulo do celsowm/jurisprudencias_br, e "
              "vira o Gama v1.3, revisão 5f924ca, que é o extrator da solução.", t)
    larg, vao = 194, 16
    xs = [M + i * (larg + vao) for i in range(4)]

    s.append(texto(M, 96, "DADOS", t["muda"], 11, 600))
    dados = [
        ("Dev set da organização", ["26 documentos e o acervo;", "não saem da máquina", "nem entram no repositório"]),
        ("Bancos de frases", ["os documentos desmontados", "em peças; LLMs abertos", "só ampliam as frases"]),
        ("Gerador por moldes", ["cita registros do acervo,", "rótulo exato por construção,", "ruído de OCR calibrado"]),
        ("gama-goldenset", ["final_v3: 6.000 documentos", "(5.410 de treino, 590 de", "reserva); 44.234 citações"]),
    ]
    for i, (titulo, linhas) in enumerate(dados):
        s += cartao(xs[i], 106, larg, 100, titulo, linhas, t, destaque=(i == 3))
        if i:
            s.append(seta([(xs[i] - vao + 1, 156), (xs[i] - 1, 156)], t))

    # goldenset -> treino do v1.2 e destilação do v1.3
    cx = [x + larg / 2 for x in xs]
    s.append(f'<polyline points="{cx[3]:.1f},206 {cx[3]:.1f},236 {cx[1]:.1f},236" fill="none" '
             f'stroke="{t["muda"]}" stroke-width="1.4"/>')
    s.append(seta([(cx[1], 236), (cx[1], 270)], t))
    s.append(seta([(cx[2], 236), (cx[2], 270)], t))
    s.append(texto(cx[1] + 10, 256, "treino, dataset em 31474b1", t["muda"], 10.5))
    s.append(texto(cx[2] + 10, 256, "destilação, dataset em ec430c0", t["muda"], 10.5))

    s.append(texto(M, 262, "MODELOS", t["muda"], 11, 600))
    modelos = [
        ("mmBERT-base", ["jhu-clsp/mmBERT-base", "revisão c595503, MIT", "encoder multilíngue"]),
        ("Gama v1.2", ["22 camadas, 307,5M", "revisão ad06ffd (tag v1.2)", "o professor"]),
        ("Gama v1.3", ["12 camadas, 257,4M", "revisão 5f924ca (tag v1.3)", "o que a solução usa"]),
        ("Solução", ["extrator + guarda +", "consulta ao acervo;", "um comando, sem rede"]),
    ]
    for i, (titulo, linhas) in enumerate(modelos):
        s += cartao(xs[i], 272, larg, 100, titulo, linhas, t, destaque=(i == 2))
        if i:
            s.append(seta([(xs[i] - vao + 1, 322), (xs[i] - 1, 322)], t))

    # ementas reais -> destilação
    s += cartao(xs[2], 404, 2 * larg + vao, 64, "1.818 ementas reais, sem rótulo",
                ["celsowm/jurisprudencias_br, CC-BY-4.0; nelas só vale a imitação do v1.2"], t)
    s.append(seta([(cx[2], 404), (cx[2], 373)], t))

    s.append(texto(M, h - 34, "Os LLMs de pesos abertos (DeepSeek-V4-Pro, Kimi-K3, GLM-5.3) ficam no preparo dos "
                   "dados: ampliam frases e anotam ementas de teste.", t["tinta2"], 11))
    s.append(texto(M, h - 18, "Nenhum deles escreve citação, decide rótulo ou roda na solução avaliada.",
                   t["tinta2"], 11))
    s.append('</svg>')
    return "\n".join(s) + "\n"


# ------------------------------------------------------------------ 4. guarda
def guarda(t: dict) -> str:
    h = 420
    x0, x1, lo, hi = 60.0, 820.0, 0.5, 1.0
    pos = lambda c: x0 + (c - lo) / (hi - lo) * (x1 - x0)
    lim, fp, acerto = pos(0.95), pos(0.83), pos(0.9993)
    s = abrir(h, "A guarda: onde o modelo hesita, vale a régua",
              "Confiança do trecho: a média da probabilidade que o modelo deu ao rótulo escolhido, do Gama v1.3.",
              "Eixo de confiança de 0,50 a 1,00 com o limiar em 0,95. Abaixo do limiar o trecho do modelo sai "
              "e entra o da régua, se ela achou algo ali; acima vale o modelo. A mediana dos falsos positivos "
              "em ementa real fica em 0,83, abaixo do limiar. O menor dos 4.447 acertos no estresse difícil "
              "fica em 0,9993, acima dele. No estilo da organização, 4.626 documentos saem iguais com e sem a "
              "guarda. Em texto real, 305 ementas, o F1 de extração vai de 0,620 a 0,818.", t)

    # os dois pontos medidos, rotulados acima da faixa
    yb, alt = 148, 44
    ym, ye = yb + alt / 2, yb + alt + 8
    s.append(texto(fp + 20, 96, "0,83", t["tinta"], 13, 600, ancora="end"))
    s.append(texto(fp + 20, 112, "mediana dos falsos positivos", t["tinta2"], 11, ancora="end"))
    s.append(texto(fp + 20, 127, "em ementa real", t["tinta2"], 11, ancora="end"))
    s.append(texto(x1, 96, "0,9993", t["tinta"], 13, 600, ancora="end"))
    s.append(texto(x1, 112, "o menor dos 4.447 acertos", t["tinta2"], 11, ancora="end"))
    s.append(texto(x1, 127, "no estresse difícil", t["tinta2"], 11, ancora="end"))

    # faixas: hesita (cinza) e confia (azul)
    s.append(f'<rect x="{x0}" y="{yb}" width="{lim - x0 - 1:.1f}" height="{alt}" fill="{t["cartao"]}"/>')
    s.append(f'<rect x="{lim + 1:.1f}" y="{yb}" width="{x1 - lim - 1:.1f}" height="{alt}" fill="{t["realce"]}"/>')
    s.append(f'<line x1="{lim:.1f}" y1="{yb - 10}" x2="{lim:.1f}" y2="{ye + 5}" stroke="{t["tinta"]}" '
             f'stroke-width="1.5" stroke-dasharray="4 3"/>')
    for x, cor, rot in ((fp, t["outro"], "0,83, mediana dos falsos positivos em ementa real"),
                        (acerto, t["destaque"], "0,9993, o menor dos 4.447 acertos no estresse difícil")):
        s.append(f'<g><title>{rot}</title><line x1="{x:.1f}" y1="134" x2="{x:.1f}" y2="{ym - 7}" '
                 f'stroke="{t["muda"]}" stroke-width="1"/>'
                 f'<circle cx="{x:.1f}" cy="{ym}" r="6" fill="{cor}" stroke="{t["fundo"]}" stroke-width="2"/></g>')

    # eixo
    s.append(f'<line x1="{x0}" y1="{ye}" x2="{x1}" y2="{ye}" stroke="{t["eixo"]}"/>')
    for c in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        s.append(f'<line x1="{pos(c):.1f}" y1="{ye}" x2="{pos(c):.1f}" y2="{ye + 5}" stroke="{t["eixo"]}"/>')
        s.append(texto(pos(c), ye + 19, f"{c:.2f}".replace(".", ","), t["muda"], 11, ancora="middle"))
    s.append(texto(lim, ye + 19, "limiar 0,95", t["tinta"], 11, 600, ancora="middle"))
    s.append(texto(x0, ye + 46, "abaixo do limiar: sai o trecho do modelo e entra o da régua, se ela achou algo ali",
                   t["tinta2"], 11.5))
    s.append(texto(x1, ye + 46, "do limiar para cima: vale o modelo", t["tinta"], 11.5, 600, ancora="end"))

    # o efeito
    yp = 270
    meio = (W - 2 * M - 16) / 2
    s.append(caixa(M, yp, meio, 96, t))
    s.append(texto(M + 16, yp + 24, "No estilo da organização", t["tinta2"], 11.5))
    s.append(texto(M + 16, yp + 54, "4.626 documentos", t["tinta"], 22, 600))
    s.append(texto(M + 16, yp + 78, "saem iguais com e sem a guarda: a nota oficial não muda", t["tinta2"], 11.5))

    xd = M + meio + 16
    s.append(caixa(xd, yp, meio, 96, t))
    s.append(texto(xd + 16, yp + 24, "Em texto real, 305 ementas, F1 de extração", t["tinta2"], 11.5))
    escala = (meio - 190) / 1.0
    for i, (nome, v, cor, peso) in enumerate((("modelo sozinho", 0.620, t["outro"], 400),
                                              ("com a guarda", 0.818, t["destaque"], 600))):
        y = yp + 38 + i * 26
        s.append(texto(xd + 16, y + 13, nome, t["tinta"], 12, peso))
        s.append(f'<g><title>{nome}: {f"{v:.3f}".replace(".", ",")}</title>'
                 f'<rect x="{xd + 120:.1f}" y="{y}" width="{v * escala:.1f}" height="18" rx="4" fill="{cor}"/></g>')
        s.append(texto(xd + 126 + v * escala, y + 13, f"{v:.3f}".replace(".", ","), t["tinta"], 12, peso))

    s.append(texto(M, h - 34, "A guarda tem mais uma regra: referência vaga a até 2 caracteres de outra citação "
                   "sai, porque colada a um precedente ela é o rabo dele.", t["tinta2"], 11))
    s.append(texto(M, h - 18, "No estilo da organização o modelo não hesita, então nenhuma das duas regras dispara.",
                   t["tinta2"], 11))
    s.append('</svg>')
    return "\n".join(s) + "\n"


FIGURAS = {"pipeline": pipeline, "execucao": execucao, "linhagem": linhagem, "guarda": guarda}


def main() -> int:
    for nome, desenhar in FIGURAS.items():
        for tema, t in TEMAS.items():
            arq = PASTA / f"{nome}-{tema}.svg"
            arq.write_text(desenhar(t), encoding="utf-8")
            print("->", arq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
