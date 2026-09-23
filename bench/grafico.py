# -*- coding: utf-8 -*-
"""Desenha o benchmark (bench/resultados.json) em SVG, versões clara e escura.

Dois painéis com eixo próprio, porque as medidas têm escalas diferentes: métrica oficial
no estresse difícil (teto 1,10) e F1 de extração no texto real (teto 1,0). O Gama leva a
cor de destaque; os demais ficam em cinza, com rótulo e valor escritos, para a
identidade nunca depender só da cor. Só biblioteca padrão.

    docker compose run --rm gama python -m bench.grafico
"""
from __future__ import annotations

import json
import pathlib
from xml.sax.saxutils import escape

PASTA = pathlib.Path(__file__).parent
TEMAS = {
    "claro": {"fundo": "#fcfcfb", "tinta": "#0b0b0b", "tinta2": "#52514e", "muda": "#898781",
              "grade": "#e1e0d9", "eixo": "#c3c2b7", "destaque": "#2a78d6", "outro": "#aeaca4",
              "borda": "rgba(11,11,11,0.10)"},
    "escuro": {"fundo": "#1a1a19", "tinta": "#ffffff", "tinta2": "#c3c2b7", "muda": "#898781",
               "grade": "#2c2c2a", "eixo": "#383835", "destaque": "#3987e5", "outro": "#5f5e59",
               "borda": "rgba(255,255,255,0.10)"},
}
LINHAS = [  # (chave, nome, descrição)
    ("gama", "Gama v1.2", "mmBERT fine-tunado"),
    ("regua", "Régua", "regex, sem modelo"),
    ("qwen", "Qwen3-8B", "LLM, zero-shot"),
    ("gliner", "GLiNER multi v2.1", "NER, zero-shot"),
]
PAINEIS = [  # (título, subtítulo, extrai valor, domínio, marcas, casas decimais)
    ("Estresse difícil · métrica oficial", "600 documentos com redação nunca treinada, N1 + N2",
     lambda r: r.get("estresse", {}).get("oficial", {}).get("final"), 1.1,
     [(0, "0"), (0.5, "0,5"), (1.0, "1,0")], 4),
    ("Texto real · F1 de extração", "305 ementas reais do STF, STJ e TJRJ",
     lambda r: r.get("reais", {}).get("extracao", {}).get("f1"), 1.0,
     [(0, "0"), (0.5, "0,5"), (1.0, "1,0")], 3),
]
RODAPE = [
    "Mesmo resolver e mesma métrica oficial para os quatro. GLiNER e Qwen3-8B têm pesos abertos e rodam em zero-shot,",
    "sem ajuste nos conjuntos avaliados. O Gama foi treinado só com documentos sintéticos, nunca com os avaliados.",
]

W, M, ROTULO, VAO = 880, 28, 168, 44
PAINEL = (W - 2 * M - ROTULO - VAO) // 2
Y_LINHAS, ALTURA_LINHA, BARRA = 138, 42, 20
FONTE = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def num(v: float, casas: int = 3) -> str:
    return f"{v:.{casas}f}".replace(".", ",")


def barra(x: float, y: float, comp: float, alt: float, cor: str) -> str:
    """Barra horizontal: base reta no eixo, ponta arredondada (4 px) no valor."""
    r = min(4.0, comp)
    return (f'<path d="M{x:.1f},{y:.1f} h{comp - r:.1f} a{r},{r} 0 0 1 {r},{r} v{alt - 2 * r:.1f} '
            f'a{r},{r} 0 0 1 -{r},{r} h-{comp - r:.1f} z" fill="{cor}"/>')


def desenhar(res: dict, t: dict) -> str:
    n = len(LINHAS)
    y_eixo = Y_LINHAS + n * ALTURA_LINHA
    H = y_eixo + 86
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'role="img" aria-labelledby="t d" font-family="{FONTE}">',
         '<title id="t">Gama contra extratores sem treino</title>',
         '<desc id="d">' + escape("; ".join(
             f"{nome}: estresse {num(v1, 4) if v1 is not None else 'sem dado'}, texto real "
             f"{num(v2) if v2 is not None else 'sem dado'}"
             for k, nome, _ in LINHAS
             for v1, v2 in [(PAINEIS[0][2](res.get(k, {})), PAINEIS[1][2](res.get(k, {})))])) + '</desc>',
         f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="10" fill="{t["fundo"]}" '
         f'stroke="{t["borda"]}"/>',
         f'<text x="{M}" y="40" font-size="17" font-weight="600" fill="{t["tinta"]}">'
         'Gama contra extratores sem treino</text>',
         f'<text x="{M}" y="62" font-size="12.5" fill="{t["tinta2"]}">'
         'Quanto maior a barra, melhor. Barra do Gama em destaque.</text>']

    for i, (k, nome, desc) in enumerate(LINHAS):
        yc = Y_LINHAS + i * ALTURA_LINHA + ALTURA_LINHA / 2
        peso = "600" if k == "gama" else "400"
        s.append(f'<text x="{M + ROTULO - 14}" y="{yc - 3:.1f}" text-anchor="end" font-size="13" '
                 f'font-weight="{peso}" fill="{t["tinta"]}">{escape(nome)}</text>')
        s.append(f'<text x="{M + ROTULO - 14}" y="{yc + 12:.1f}" text-anchor="end" font-size="11" '
                 f'fill="{t["muda"]}">{escape(desc)}</text>')

    for p, (titulo, sub, valor, dom, marcas, casas) in enumerate(PAINEIS):
        x0 = M + ROTULO + p * (PAINEL + VAO)
        util = PAINEL - 52                               # sobra para o valor na ponta
        s.append(f'<text x="{x0}" y="{Y_LINHAS - 34}" font-size="13" font-weight="600" '
                 f'fill="{t["tinta"]}">{escape(titulo)}</text>')
        s.append(f'<text x="{x0}" y="{Y_LINHAS - 17}" font-size="11" fill="{t["muda"]}">{escape(sub)}</text>')
        for v, rot in marcas:
            x = x0 + v / dom * util
            cor = t["eixo"] if v == 0 else t["grade"]
            s.append(f'<line x1="{x:.1f}" y1="{Y_LINHAS - 4}" x2="{x:.1f}" y2="{y_eixo}" '
                     f'stroke="{cor}" stroke-width="1"/>')
            s.append(f'<text x="{x:.1f}" y="{y_eixo + 16}" text-anchor="middle" font-size="11" '
                     f'fill="{t["muda"]}" style="font-variant-numeric: tabular-nums">{rot}</text>')
        for i, (k, nome, _) in enumerate(LINHAS):
            y = Y_LINHAS + i * ALTURA_LINHA + (ALTURA_LINHA - BARRA) / 2
            v = valor(res.get(k, {}))
            if v is None:
                s.append(f'<text x="{x0 + 8}" y="{y + BARRA / 2 + 4:.1f}" font-size="11" '
                         f'fill="{t["muda"]}">sem dado</text>')
                continue
            comp = max(1.0, min(v, dom) / dom * util)
            cor = t["destaque"] if k == "gama" else t["outro"]
            s.append(f'<g><title>{escape(nome)} · {escape(titulo)}: {num(v, casas)}</title>'
                     + barra(x0, y, comp, BARRA, cor) + '</g>')
            peso = "600" if k == "gama" else "400"
            s.append(f'<text x="{x0 + comp + 6:.1f}" y="{y + BARRA / 2 + 4:.1f}" font-size="12" '
                     f'font-weight="{peso}" fill="{t["tinta"]}" style="font-variant-numeric: tabular-nums">'
                     f'{num(v, casas)}</text>')

    for j, linha in enumerate(RODAPE):
        s.append(f'<text x="{M}" y="{y_eixo + 46 + j * 16}" font-size="11" fill="{t["tinta2"]}">'
                 f'{escape(linha)}</text>')
    s.append('</svg>')
    return "\n".join(s) + "\n"


def main() -> int:
    res = json.loads((PASTA / "resultados.json").read_text(encoding="utf-8"))["resultados"]
    for nome, tema in TEMAS.items():
        arq = PASTA / f"grafico-{nome}.svg"
        arq.write_text(desenhar(res, tema), encoding="utf-8")
        print("->", arq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
