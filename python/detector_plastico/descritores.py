"""Descritores: transformam cada pedaço da imagem em um vetor de números.

A imagem é dividida em sub-células de 8x8 px. Uma "célula" (unidade classificada
como plástico ou não) tem 2x2 sub-células (16x16 px). Para cada célula olhamos
três janelas centradas nela — 16 px (objeto), 48 px (vizinhança) e 128 px (cena) —
e extraímos cor, forma/contorno (tipo HOG), textura (LBP), brilho especular e o
contraste do objeto com o entorno.

Mantém o mesmo tamanho de descritor do projeto Java para compatibilidade de
raciocínio. Usa imagens integrais (soma acumulada) por canal para extrair
qualquer janela em tempo constante.
"""

from __future__ import annotations

import numpy as np

from . import filtros

P = 8            # lado da sub-célula, em pixels
CEL = 2          # lado da célula classificada, em sub-células (2 x 8 = 16 px)
JANELAS = (2, 6, 16)   # janelas de contexto, em sub-células: 16, 48 e 128 px

NC = 20          # faixas de cor (4 neutras + 8 matizes x 2 saturações)
NO = 8           # orientações do gradiente (forma/contornos)
NL = 10          # padrões de textura LBP

# índices dos canais acumulados por sub-célula
C_COR = 0
C_ORI = NC
C_LBP = C_ORI + NO
C_SOMA = C_LBP + NL
C_QUAD = C_SOMA + 5
C_MAG = C_QUAD + 5
C_BORDA = C_MAG + 1
C_ESPEC = C_BORDA + 1
C_N = C_ESPEC + 1
NCANAIS = C_N + 1

POR_JANELA = NC + 10 + NO + 2 + NL + 1
TAMANHO = POR_JANELA * len(JANELAS) + 10  # tamanho do descritor

_LIMIAR_BORDA = 0.15

# tabela LBP uniforme (igual ao Java): rótulos 0..8 por nº de bits, 9 = não-uniforme
_TABELA_LBP = np.empty(256, dtype=np.int32)
for _b in range(256):
    _rot = ((_b >> 1) | ((_b & 1) << 7)) & 0xFF
    _trans = bin(_b ^ _rot).count("1")
    _TABELA_LBP[_b] = bin(_b).count("1") if _trans <= 2 else 9


class Grade:
    """Somas acumuladas (imagem integral) por canal sobre a grade de sub-células."""

    __slots__ = ("gw", "gh", "w", "h", "integral")

    def __init__(self, w, h, gw, gh, integral):
        self.w, self.h = w, h
        self.gw, self.gh = gw, gh
        self.integral = integral  # (gh+1, gw+1, NCANAIS)

    def soma(self, c, x0, y0, x1, y1):
        I = self.integral
        return (I[y1, x1, c] - I[y0, x1, c] - I[y1, x0, c] + I[y0, x0, c])

    def nx(self):
        return max(0, self.gw - CEL + 1)

    def ny(self):
        return max(0, self.gh - CEL + 1)


def _faixa_cor(r, g, b, mx, mn, s, v):
    if v < 0.18:
        return 0  # preto
    if s < 0.18:
        return 1 if v < 0.45 else (2 if v < 0.75 else 3)  # cinza esc, cinza claro, branco
    h = filtros.matiz(r, g, b, mx, mn)
    hb = int(((h + 22.5) % 360.0) / 45.0)
    return 4 + min(7, hb) * 2 + (0 if s < 0.5 else 1)


def _acumular_por_subcelula(im: np.ndarray):
    """Monta o array (gh, gw, NCANAIS) de somas por sub-célula (vetorizado)."""
    h, w = im.shape[:2]
    gw = (w + P - 1) // P
    gh = (h + P - 1) // P

    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    mx = np.maximum.reduce([r, g, b])
    mn = np.minimum.reduce([r, g, b])
    s = np.where(mx > 1e-6, (mx - mn) / np.where(mx > 1e-6, mx, 1.0), 0.0)
    v = mx

    c = filtros.cinza(im)
    gx, gy = filtros.sobel(c)
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)

    # --- faixa de cor por pixel (vetorizado) ---
    faixa = np.zeros((h, w), dtype=np.int32)
    hgraus, _, _ = filtros.hsv(im)
    hb = (((hgraus + 22.5) % 360.0) / 45.0).astype(np.int32)
    hb = np.minimum(7, hb)
    cor_viva = 4 + hb * 2 + (s >= 0.5).astype(np.int32)
    cinza_idx = np.where(v < 0.45, 1, np.where(v < 0.75, 2, 3))
    faixa = np.where(v < 0.18, 0, np.where(s < 0.18, cinza_idx, cor_viva)).astype(np.int32)

    # --- orientação do gradiente (HOG) ---
    ang = np.arctan2(gy, gx)
    ang = np.where(ang < 0, ang + np.pi, ang)
    bo = (ang / np.pi * NO).astype(np.int32)
    bo = np.minimum(NO - 1, bo)
    tem_mag = mag > 1e-4

    # --- LBP ---
    lbp = _lbp(c)

    acc = np.zeros((gh, gw, NCANAIS), dtype=np.float64)
    sy = (np.arange(h) // P)
    sx = (np.arange(w) // P)
    SX, SY = np.meshgrid(sx, sy)
    plano = (SY * gw + SX).ravel()

    def soma_em(canal_base, valores, indices_extra=None):
        """Acumula ``valores`` nas sub-células; opcionalmente num sub-canal variável."""
        if indices_extra is None:
            np.add.at(acc.reshape(-1, NCANAIS)[:, canal_base:canal_base + 1],
                      (plano, np.zeros_like(plano)), valores.ravel())
        else:
            canais = canal_base + indices_extra.ravel()
            np.add.at(acc.reshape(-1, NCANAIS), (plano, canais), valores.ravel())

    flat = acc.reshape(-1, NCANAIS)

    # cor: +1 na faixa correspondente
    np.add.at(flat, (plano, C_COR + faixa.ravel()), 1.0)
    # orientação: +mag na orientação (só onde há gradiente)
    np.add.at(flat, (plano, C_ORI + bo.ravel()), np.where(tem_mag, mag, 0.0).ravel())
    # LBP: +1 no padrão
    np.add.at(flat, (plano, C_LBP + lbp.ravel()), 1.0)
    # somas e quadrados de r,g,b,s,v
    for k, canal in enumerate((r, g, b, s, v)):
        np.add.at(flat, (plano, np.full_like(plano, C_SOMA + k)), canal.ravel())
        np.add.at(flat, (plano, np.full_like(plano, C_QUAD + k)), (canal * canal).ravel())
    np.add.at(flat, (plano, np.full_like(plano, C_MAG)), mag.ravel())
    np.add.at(flat, (plano, np.full_like(plano, C_BORDA)), (mag > _LIMIAR_BORDA).astype(np.float64).ravel())
    espec = (v > 0.88) & (s < 0.15)
    np.add.at(flat, (plano, np.full_like(plano, C_ESPEC)), espec.astype(np.float64).ravel())
    np.add.at(flat, (plano, np.full_like(plano, C_N)), 1.0)

    return acc, gw, gh


def _lbp(c: np.ndarray) -> np.ndarray:
    """Padrão binário local (LBP) por pixel, com borda replicada."""
    cp = np.pad(c, 1, mode="edge")
    centro = cp[1:-1, 1:-1] + 0.02
    viz = [
        (cp[0:-2, 0:-2], 1), (cp[0:-2, 1:-1], 2), (cp[0:-2, 2:], 4),
        (cp[1:-1, 2:], 8), (cp[2:, 2:], 16), (cp[2:, 1:-1], 32),
        (cp[2:, 0:-2], 64), (cp[1:-1, 0:-2], 128),
    ]
    bits = np.zeros(c.shape, dtype=np.int32)
    for v, peso in viz:
        bits |= (v >= centro).astype(np.int32) * peso
    return _TABELA_LBP[bits]


def calcular(im: np.ndarray) -> Grade:
    """Calcula a grade de imagens integrais a partir de uma imagem RGB [0,1]."""
    h, w = im.shape[:2]
    acc, gw, gh = _acumular_por_subcelula(im)
    # imagem integral por canal: (gh+1, gw+1, NCANAIS)
    integ = np.zeros((gh + 1, gw + 1, NCANAIS), dtype=np.float64)
    integ[1:, 1:, :] = np.cumsum(np.cumsum(acc, axis=0), axis=1)
    return Grade(w, h, gw, gh, integ)


def extrair(g: Grade, cx: int, cy: int, out: np.ndarray, off: int) -> None:
    """Escreve em ``out[off:off+TAMANHO]`` o descritor da célula cujo canto
    superior esquerdo é a sub-célula (cx, cy)."""
    ccx = cx + CEL // 2
    ccy = cy + CEL // 2
    med = np.zeros(5 * len(JANELAS), dtype=np.float32)
    o = off
    for k, jan in enumerate(JANELAS):
        meia = jan // 2
        x0 = max(0, ccx - meia); x1 = min(g.gw, ccx + meia)
        y0 = max(0, ccy - meia); y1 = min(g.gh, ccy + meia)
        n = max(1.0, g.soma(C_N, x0, y0, x1, y1))

        for c in range(NC):
            out[o] = g.soma(C_COR + c, x0, y0, x1, y1) / n
            o += 1
        for c in range(5):
            m = g.soma(C_SOMA + c, x0, y0, x1, y1) / n
            med[k * 5 + c] = m
            out[o] = m
            o += 1
        for c in range(5):
            m = med[k * 5 + c]
            var = g.soma(C_QUAD + c, x0, y0, x1, y1) / n - m * m
            out[o] = np.sqrt(max(0.0, var))
            o += 1
        mag = g.soma(C_MAG, x0, y0, x1, y1)
        for c in range(NO):
            out[o] = g.soma(C_ORI + c, x0, y0, x1, y1) / (mag + 1e-6)
            o += 1
        out[o] = mag / n; o += 1
        out[o] = g.soma(C_BORDA, x0, y0, x1, y1) / n; o += 1
        for c in range(NL):
            out[o] = g.soma(C_LBP + c, x0, y0, x1, y1) / n
            o += 1
        out[o] = g.soma(C_ESPEC, x0, y0, x1, y1) / n; o += 1

    # contraste entre o objeto e o entorno (plástico costuma destoar do rio)
    for c in range(5):
        out[o] = med[c] - med[5 + c]; o += 1
    for c in range(5):
        out[o] = med[c] - med[10 + c]; o += 1


def extrair_muitas(g: Grade, celulas) -> np.ndarray:
    """Extrai os descritores de uma lista de (cx, cy) e devolve matriz (N, TAMANHO)."""
    out = np.zeros((len(celulas), TAMANHO), dtype=np.float32)
    linha = np.zeros(TAMANHO, dtype=np.float32)
    for i, (cx, cy) in enumerate(celulas):
        extrair(g, cx, cy, linha, 0)
        out[i] = linha
    return out
