# -*- coding: utf-8 -*-
"""Desenha risco x cobertura (bench/controles/offline/risco_cobertura.json) em SVG, claro e escuro.

Três painéis com o mesmo eixo: o estresse com a confiança calibrada e as 305 e 172 ementas com a
confiança do extrator. Cada painel usa uma única variável de confiança; elas não se misturam. No
texto real, o Gama com a guarda leva a cor de destaque e o modelo antes da guarda fica em cinza,
com legenda e rótulo na ponta, para a identidade não depender só da cor. Estilo e temas de
`bench/grafico.py` (importados; aquele arquivo não muda). Só biblioteca padrão.

    docker compose run --rm gama python -m bench.controles.offline.grafico_risco
"""
from __future__ import annotations

import copy
import json
import pathlib
from xml.sax.saxutils import escape

from bench.grafico import FONTE, TEMAS, num

AQUI = pathlib.Path(__file__).parent
W, M, VAO = 880, 28, 30
PW = (W - 2 * M - 2 * VAO) // 3          # largura de cada painel
ESQ, DIR = 34, 44                        # rótulos do eixo y; folga para o rótulo na ponta
Y_TOPO, PH = 150, 210                    # topo e altura da área de plotagem
Y_MAX = 0.6
MARCAS_Y = [(0.0, "0"), (0.2, "0,2"), (0.4, "0,4"), (0.6, "0,6")]
MARCAS_X = [(0.0, "0"), (0.5, "0,5"), (1.0, "1")]


def _mil(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def esperados(curva: list[dict]) -> list[tuple[float, float]]:
    """(cobertura, risco) ao longo de todas as posições: dentro de um empate de confiança os erros entram
    por igual (a expectativa sob ordem aleatória, a mesma que o AURC integra). Até 12 amostras por empate."""
    n = curva[-1]["mantidas"]
    pts, k0, e0 = [], 0, 0
    for q in curva:
        k1, e1 = q["mantidas"], q["erros"]
        m = k1 - k0
        passos = sorted({k0 + 1, k1} | {k0 + max(1, round(m * j / 12)) for j in range(1, 13)})
        for k in passos:
            pts.append((k / n, (e0 + (e1 - e0) * (k - k0) / m) / k))
        k0, e0 = k1, e1
    return pts


def _pontos(c: dict) -> list[dict]:
    return [dict(zip(c["colunas"], p)) for p in c["pontos"]]


def paineis(r: dict) -> list[dict]:
    r = copy.deepcopy(r)                   # desenhar() chama isto uma vez por tema
    e = r["estresse_confianca_calibrada"]
    for v in [e] + [x for k in ("reais_305_confianca_extrator", "novas_172_confianca_extrator")
                    for x in r[k]["variantes"].values()]:
        v["curva"] = _pontos(v["curva"])
    out = [{"titulo": "Estresse difícil", "sub": "600 docs · confiança calibrada (Brier)",
            "series": [{"chave": "cal", "nome": "Gama v1.3 + guarda", "curva": e["curva"], "destaque": True}],
            "nota": [f"0 erros em {_mil(e['previsoes'])} previsões: a curva fica",
                     f"no zero e não há erro para a confiança ordenar."],
            "marcos": []}]
    for chave, titulo, sub in (("reais_305_confianca_extrator", "305 ementas reais", "validação · confiança do extrator"),
                               ("novas_172_confianca_extrator", "172 ementas novas", "confirmação · confiança do extrator")):
        g, c = r[chave]["variantes"]["com_guarda"], r[chave]["variantes"]["cru"]
        n_mod = g["por_origem"]["modelo"]["n"]
        fronteira = next(p for p in g["curva"] if p["mantidas"] == n_mod)
        lim = c["no_limiar_da_guarda"]
        out.append({"titulo": titulo, "sub": sub,
                    "series": [{"chave": "cru", "nome": "antes da guarda", "curva": c["curva"], "destaque": False},
                               {"chave": "gua", "nome": "com a guarda", "curva": g["curva"], "destaque": True}],
                    "nota": [f"AUROC com a guarda {num(g['auroc'], 2)} · antes {num(c['auroc'], 2)}",
                             f"AURC {num(g['aurc'], 3)} (oráculo {num(g['aurc_oraculo'], 3)}) · "
                             f"antes {num(c['aurc'], 3)}"],
                    "marcos": [
                        {"serie": "cru", "x": lim["cobertura"], "y": lim["risco"], "rotulo": "0,95",
                         "titulo": f"Antes da guarda, confiança >= 0,95: cobertura {num(lim['cobertura'], 3)}, "
                                   f"risco {num(lim['risco'], 3)}, recall {num(lim['recall'], 3)}"},
                        {"serie": "gua", "x": fronteira["cobertura"], "y": fronteira["risco"], "rotulo": "régua",
                         "titulo": f"Com a guarda: {_mil(n_mod)} spans do modelo (>= 0,95), risco "
                                   f"{num(fronteira['risco'], 3)}; depois deles, {g['por_origem']['regua']['n']} spans "
                                   f"da régua, risco {num(g['por_origem']['regua']['taxa_erro'], 3)} entre eles"}]})
    return out


def desenhar(r: dict, t: dict) -> str:
    ps = paineis(r)
    y_eixo = Y_TOPO + PH
    H = y_eixo + 150

    def px(x0, v):
        return x0 + ESQ + v * (PW - ESQ - DIR)

    def py(v):
        return y_eixo - min(v, Y_MAX) / Y_MAX * PH

    desc = []
    for p in ps:
        for s in p["series"]:
            fim = s["curva"][-1]
            desc.append(f"{p['titulo']}, {s['nome']}: risco {num(fim['risco'], 3)} com cobertura total")
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'role="img" aria-labelledby="t d" font-family="{FONTE}">',
         '<title id="t">Risco x cobertura do Gama v1.3</title>',
         '<desc id="d">' + escape("; ".join(desc)) + '</desc>',
         f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="10" fill="{t["fundo"]}" stroke="{t["borda"]}"/>',
         f'<text x="{M}" y="40" font-size="17" font-weight="600" fill="{t["tinta"]}">Risco x cobertura do Gama v1.3</text>',
         f'<text x="{M}" y="62" font-size="12.5" fill="{t["tinta2"]}">Previsões ordenadas pela confiança, da maior '
         'para a menor. Quanto mais baixa a curva, menos erro entre as mantidas.</text>']
    # legenda (uma vez, acima dos painéis)
    lx = M
    for nome, cor in (("Gama v1.3 com a guarda (como roda na solução)", t["destaque"]),
                      ("o mesmo modelo antes da guarda", t["muda"])):
        s.append(f'<line x1="{lx}" y1="86" x2="{lx + 18}" y2="86" stroke="{cor}" stroke-width="2" stroke-linecap="round"/>')
        s.append(f'<text x="{lx + 24}" y="90" font-size="12" fill="{t["tinta2"]}">{escape(nome)}</text>')
        lx += 24 + 7.0 * len(nome) + 26

    for i, p in enumerate(ps):
        x0 = M + i * (PW + VAO)
        s.append(f'<text x="{x0}" y="{Y_TOPO - 36}" font-size="13" font-weight="600" fill="{t["tinta"]}">'
                 f'{escape(p["titulo"])}</text>')
        s.append(f'<text x="{x0}" y="{Y_TOPO - 19}" font-size="11" fill="{t["muda"]}">{escape(p["sub"])}</text>')
        for v, rot in MARCAS_Y:
            y = py(v)
            s.append(f'<line x1="{px(x0, 0):.1f}" y1="{y:.1f}" x2="{px(x0, 1):.1f}" y2="{y:.1f}" '
                     f'stroke="{t["eixo"] if v == 0 else t["grade"]}" stroke-width="1"/>')
            s.append(f'<text x="{px(x0, 0) - 6:.1f}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
                     f'fill="{t["muda"]}" style="font-variant-numeric: tabular-nums">{rot}</text>')
        for v, rot in MARCAS_X:
            x = px(x0, v)
            s.append(f'<line x1="{x:.1f}" y1="{y_eixo}" x2="{x:.1f}" y2="{y_eixo + 4}" stroke="{t["eixo"]}" stroke-width="1"/>')
            s.append(f'<text x="{x:.1f}" y="{y_eixo + 17}" text-anchor="middle" font-size="11" fill="{t["muda"]}" '
                     f'style="font-variant-numeric: tabular-nums">{rot}</text>')
        s.append(f'<text x="{px(x0, 0.5):.1f}" y="{y_eixo + 33}" text-anchor="middle" font-size="11" '
                 f'fill="{t["tinta2"]}">cobertura (fração mantida)</text>')
        if i == 0:
            s.append(f'<text transform="translate({x0 - 2},{Y_TOPO + PH / 2}) rotate(-90)" text-anchor="middle" '
                     f'font-size="11" fill="{t["tinta2"]}">risco (erros / mantidas)</text>')
        pontos_fim = {}
        for se in p["series"]:
            cor = t["destaque"] if se["destaque"] else t["muda"]
            c = se["curva"]
            d = "M" + " L".join(f"{px(x0, x):.1f},{py(y):.1f}" for x, y in esperados(c))
            fim = c[-1]
            s.append(f'<g><title>{escape(p["titulo"])} · {escape(se["nome"])}: risco {num(fim["risco"], 3)} com todas '
                     f'as {_mil(fim["mantidas"])} previsões ({_mil(fim["erros"])} erros)</title>'
                     f'<path d="{d}" fill="none" stroke="{cor}" stroke-width="2" stroke-linejoin="round" '
                     f'stroke-linecap="round"/></g>')
            xe, ye = px(x0, fim["cobertura"]), py(fim["risco"])
            s.append(f'<circle cx="{xe:.1f}" cy="{ye:.1f}" r="4" fill="{cor}" stroke="{t["fundo"]}" stroke-width="2"/>')
            pontos_fim[se["chave"]] = (xe, ye, fim["risco"], se["destaque"])
        for xe, ye, v, dest in pontos_fim.values():
            s.append(f'<text x="{xe + 8:.1f}" y="{ye + 4:.1f}" font-size="12" font-weight="{600 if dest else 400}" '
                     f'fill="{t["tinta"]}" style="font-variant-numeric: tabular-nums">{num(v, 3)}</text>')
        for mk in p["marcos"]:
            se = next(x for x in p["series"] if x["chave"] == mk["serie"])
            cor = t["destaque"] if se["destaque"] else t["muda"]
            x, y = px(x0, mk["x"]), py(mk["y"])
            s.append(f'<g><title>{escape(mk["titulo"])}</title>'
                     f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{cor}" stroke="{t["fundo"]}" stroke-width="2"/>'
                     f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="transparent"/></g>')
            if mk["serie"] == "cru":        # curva sobe para a direita: o canto de cima à esquerda fica livre
                s.append(f'<text x="{x - 8:.1f}" y="{y - 6:.1f}" text-anchor="end" font-size="11" '
                         f'fill="{t["tinta2"]}">{escape(mk["rotulo"])}</text>')
            else:                           # abaixo e à direita do ponto, sob o trecho final
                s.append(f'<text x="{x - 2:.1f}" y="{y + 19:.1f}" text-anchor="start" font-size="11" '
                         f'fill="{t["tinta2"]}">{escape(mk["rotulo"])}</text>')
        for j, linha in enumerate(p["nota"]):
            s.append(f'<text x="{x0}" y="{y_eixo + 56 + j * 15}" font-size="11" fill="{t["tinta2"]}">'
                     f'{escape(linha)}</text>')

    rodape = [
        "Erro no estresse: par casado pela métrica oficial com classe ou link errado, ou previsão espúria. No texto real: "
        "previsão sem par no casamento",
        "1 para 1 (mesmo tipo, IoU ≥ 0,5). Confiança calibrada (a do Brier) e confiança do extrator (a da guarda) são "
        "variáveis diferentes; cada painel usa uma.",
        "Com a guarda, o span da régua que entra no lugar de um span fraco leva a confiança do extrator naquele trecho "
        "(< 0,95). Marcas: 0,95 = limiar da guarda.",
    ]
    for j, linha in enumerate(rodape):
        s.append(f'<text x="{M}" y="{y_eixo + 100 + j * 15}" font-size="10.5" fill="{t["muda"]}">{escape(linha)}</text>')
    s.append('</svg>')
    return "\n".join(s) + "\n"


def main() -> int:
    r = json.loads((AQUI / "risco_cobertura.json").read_text(encoding="utf-8"))
    for nome, tema in TEMAS.items():
        arq = AQUI / f"risco-cobertura-{nome}.svg"
        arq.write_text(desenhar(r, tema), encoding="utf-8")
        print("->", arq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
