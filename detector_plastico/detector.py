"""Aplica o modelo treinado em uma imagem e localiza as regiões com plástico."""

from __future__ import annotations

import math

import numpy as np

from . import descritores, filtros

MIN_SUBCELULAS = 4  # menor região reportada (~16x16 px): descarta ruído


class Regiao:
    __slots__ = ("x", "y", "largura", "altura", "confianca", "subcelulas")

    def __init__(self, x, y, largura, altura, confianca, subcelulas):
        self.x = x
        self.y = y
        self.largura = largura
        self.altura = altura
        self.confianca = confianca
        self.subcelulas = subcelulas

    @property
    def caixa(self):
        return (self.x, self.y, self.largura, self.altura)


class Resultado:
    def __init__(self, largura, altura, escala, gw, gh, prob, incerteza):
        self.largura = largura      # da imagem original
        self.altura = altura
        self.escala = escala        # imagem de trabalho / original
        self.gw = gw
        self.gh = gh
        self.prob = prob            # (gh, gw) probabilidade por sub-célula
        self.incerteza = incerteza

    def regioes(self, limiar):
        m = self.prob >= limiar
        comps, rot = filtros.componentes(m, com_rotulos=True)
        n = len(comps)
        soma = np.zeros(n + 1, dtype=np.float64)
        if n:
            idx = rot.ravel()
            np.add.at(soma, idx, self.prob.ravel())
        px = descritores.P / self.escala
        out = []
        for k, c in enumerate(comps):
            minx, miny, maxx, maxy, area = c
            if area < MIN_SUBCELULAS:
                continue
            x0 = int(math.floor(minx * px))
            y0 = int(math.floor(miny * px))
            x1 = min(self.largura, int(math.ceil((maxx + 1) * px)))
            y1 = min(self.altura, int(math.ceil((maxy + 1) * px)))
            conf = soma[k + 1] / area
            out.append(Regiao(x0, y0, x1 - x0, y1 - y0, float(conf), area))
        out.sort(key=lambda r: r.subcelulas, reverse=True)
        return out

    def area_percentual(self, limiar):
        P = descritores.P
        w = round(self.largura * self.escala)
        h = round(self.altura * self.escala)
        ys, xs = np.nonzero(self.prob >= limiar)
        if len(xs) == 0:
            return 0.0
        larg = np.minimum(P, w - xs * P)
        alt = np.minimum(P, h - ys * P)
        area = float(np.sum(np.clip(larg, 0, None) * np.clip(alt, 0, None)))
        return 100.0 * area / (w * h)

    def imagem_calor(self, minimo):
        """Mapa de calor RGBA (gh, gw): verde->vermelho conforme a probabilidade."""
        faixa = max(0.05, 1.0 - minimo)
        p = self.prob
        t = np.clip((p - minimo) / faixa, 0.0, 1.0)
        alfa = (90 + 110 * t).astype(np.uint8)
        verde = (220 * (1 - t)).astype(np.uint8)
        vermelho = np.full_like(verde, 255)
        azul = np.zeros_like(verde)
        visivel = p >= minimo
        rgba = np.dstack([vermelho, verde, azul, np.where(visivel, alfa, 0)]).astype(np.uint8)
        return rgba

    def largura_grade(self):
        return self.gw * descritores.P / self.escala

    def altura_grade(self):
        return self.gh * descritores.P / self.escala


def detectar(modelo, img: np.ndarray) -> Resultado:
    """Detecta plástico numa imagem RGB float32 (H, W, 3)."""
    h0, w0 = img.shape[:2]
    esc = filtros.escala_para(w0, h0, modelo.lado_base)
    w = max(32, round(w0 * esc))
    h = max(32, round(h0 * esc))
    im = filtros.redimensionar(img, w, h)
    g = descritores.calcular(im)
    gw, gh, cel = g.gw, g.gh, descritores.CEL

    # extrai descritores de todas as posições de célula e prevê em lote
    celulas = [(cx, cy) for cy in range(g.ny()) for cx in range(g.nx())]
    if not celulas:
        return Resultado(w0, h0, w / w0, gw, gh, np.zeros((gh, gw), dtype=np.float32), 0.0)

    X = descritores.extrair_muitas(g, celulas)
    probs = modelo.prever(X)  # (N,)

    soma = np.zeros((gh, gw), dtype=np.float64)
    cont = np.zeros((gh, gw), dtype=np.float64)
    duvidosas = 0
    for (cx, cy), p in zip(celulas, probs):
        if 0.3 < p < 0.7:
            duvidosas += 1
        soma[cy:cy + cel, cx:cx + cel] += p
        cont[cy:cy + cel, cx:cx + cel] += 1

    prob = np.where(cont > 0, soma / np.where(cont > 0, cont, 1.0), 0.0).astype(np.float32)
    incerteza = duvidosas / len(celulas) if celulas else 0.0
    return Resultado(w0, h0, w / w0, gw, gh, prob, incerteza)
