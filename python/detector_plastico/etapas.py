"""Processamento digital de imagens passo a passo, para analisar os filtros e
ilustrar o trabalho escrito.

Cada etapa traz o resultado (imagem Pillow), uma explicação e um número medido.
A função ``salvar`` grava cada etapa como PNG e um ``etapas.txt`` com os textos.
"""

from __future__ import annotations

import colorsys
import re
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import filtros


class Etapa:
    __slots__ = ("titulo", "explicacao", "info", "imagem")

    def __init__(self, titulo, explicacao, info, imagem):
        self.titulo = titulo
        self.explicacao = explicacao
        self.info = info
        self.imagem = imagem  # PIL.Image


def _media(v):
    return float(np.mean(v))


def _desvio(v):
    return float(np.std(v))


def _contar(m):
    return int(np.count_nonzero(m))


def _cinza_img(v):
    return filtros.cinza_para_pil(v)


def _bin_img(m):
    return Image.fromarray(np.where(m, 255, 0).astype(np.uint8), mode="L")


def _mascara_img(im, m):
    """Pixels marcados em magenta sobre a foto escurecida."""
    r = np.rint(im[..., 0] * 255).astype(np.int32)
    g = np.rint(im[..., 1] * 255).astype(np.int32)
    b = np.rint(im[..., 2] * 255).astype(np.int32)
    rm = np.where(m, (r + 255) // 2, r * 35 // 100)
    gm = np.where(m, g // 3, g * 35 // 100)
    bm = np.where(m, (b + 255) // 2, b * 35 // 100)
    out = np.clip(np.dstack([rm, gm, bm]), 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _componentes_img(im, rot, comps, min_area):
    h, w = im.shape[:2]
    cores = {0: (0, 0, 0)}
    for k, c in enumerate(comps):
        if c[4] >= min_area:
            rgb = colorsys.hsv_to_rgb((k * 0.618034) % 1.0, 0.85, 1.0)
            cores[k + 1] = tuple(int(x * 255) for x in rgb)
        else:
            cores[k + 1] = (96, 96, 96)
    y = np.clip(np.rint(filtros.cinza(im) * 0.3 * 255), 0, 255).astype(np.uint8)
    base = np.dstack([y, y, y]).astype(np.uint8)
    paleta = np.zeros((len(comps) + 1, 3), dtype=np.uint8)
    for idx, cor in cores.items():
        if idx <= len(comps):
            paleta[idx] = cor
    mask = rot > 0
    base[mask] = paleta[rot[mask]]
    img = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(img)
    esp = max(1, round(w / 400.0))
    for k, c in enumerate(comps):
        if c[4] >= min_area:
            draw.rectangle([c[0], c[1], c[2], c[3]], outline=cores[k + 1], width=esp)
    return img


def calcular(im: np.ndarray):
    """Calcula as 12 etapas do PDI a partir de uma imagem RGB float32 [0,1]."""
    h, w = im.shape[:2]
    n = w * h
    e = []

    e.append(Etapa("1. Aquisição (original)",
                   "A foto como foi capturada, reduzida para no máximo 1024 px para a análise ser rápida.",
                   f"{w} × {h} pixels", filtros.rgb_para_pil(im)))

    c = filtros.cinza(im)
    e.append(Etapa("2. Tons de cinza",
                   "Luminância Y = 0,299·R + 0,587·G + 0,114·B. Os filtros de forma e borda usam um único canal.",
                   f"brilho médio: {_media(c) * 255:.0f} de 255", _cinza_img(c)))

    eq = filtros.equalizar(c)
    e.append(Etapa("3. Equalização de histograma",
                   "Redistribui os tons para ocupar toda a faixa 0–255. Realça garrafas e sacolas em água turva ou escura.",
                   f"contraste (desvio dos tons): {_desvio(c) * 255:.0f} → {_desvio(eq) * 255:.0f}",
                   filtros.rgb_para_pil(filtros.equalizar_cores(im))))

    med = filtros.mediana(c, 2)
    mudou = int(np.count_nonzero(np.abs(med - c) > 5 / 255))
    e.append(Etapa("4. Filtro de mediana 5×5",
                   "Cada pixel vira a mediana dos 25 vizinhos. Remove reflexos e ruído pontual sem borrar as bordas.",
                   f"{100.0 * mudou / n:.1f}% dos pixels mudaram mais de 5 tons", _cinza_img(med)))

    gau = filtros.desfoque(c)
    e.append(Etapa("5. Filtro gaussiano 5×5",
                   "Média ponderada pelo núcleo 1-4-6-4-1 (separável). Suaviza a textura da água antes de medir bordas.",
                   f"desvio dos tons: {_desvio(c) * 255:.1f} → {_desvio(gau) * 255:.1f}", _cinza_img(gau)))

    gx, gy = filtros.sobel(gau)
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    p99 = float(np.percentile(mag, 99))
    mag_vis = np.minimum(1.0, mag / max(1e-4, p99))
    e.append(Etapa("6. Sobel (gradiente)",
                   "Derivadas horizontal e vertical com os núcleos de Sobel. Quanto mais claro, mais forte a mudança de brilho.",
                   f"gradiente médio: {_media(mag):.3f}", _cinza_img(mag_vis)))

    bordas = filtros.bordas(im)
    e.append(Etapa("7. Canny – contornos P&B",
                   "Sobel + supressão de não-máximos + histerese: linhas finas que contornam os objetos (filtro P&B).",
                   f"{100.0 * _contar(bordas) / n:.1f}% dos pixels são borda", _bin_img(bordas)))

    t = filtros.limiar_otsu(med)
    otsu = med >= t
    e.append(Etapa("8. Limiarização de Otsu",
                   "Escolhe sozinho o tom que melhor separa claros e escuros (máxima variância entre as classes).",
                   f"limiar automático: {round(t * 255)} de 255 · {100.0 * _contar(otsu) / n:.1f}% claros",
                   _bin_img(otsu)))

    _, s, _ = filtros.hsv(im)
    e.append(Etapa("9. Canal S do HSV (saturação)",
                   "Plástico transparente ou branco tem pouca saturação (escuro aqui); rótulos e tampas, muita (claro).",
                   f"saturação média: {_media(s) * 100:.0f}%", _cinza_img(s)))

    faixas = filtros.faixas_hsv(im)
    e.append(Etapa("10. Segmentação por faixas HSV",
                   "Regra fixa: claro e pouco saturado (PET, isopor) OU cor viva quente (rótulos, tampas).",
                   f"{100.0 * _contar(faixas) / n:.1f}% da imagem marcada", _mascara_img(im, faixas)))

    limpo = filtros.fechar(filtros.abrir(faixas))
    e.append(Etapa("11. Morfologia: abertura + fechamento",
                   "Abertura apaga pontos soltos; fechamento fecha buracos e une partes do mesmo objeto (elemento 3×3).",
                   f"{100.0 * _contar(faixas) / n:.1f}% → {100.0 * _contar(limpo) / n:.1f}% da imagem",
                   _mascara_img(im, limpo)))

    comps, rot = filtros.componentes(limpo, com_rotulos=True)
    min_area = max(30, n // 2000)
    grandes = sum(1 for c in comps if c[4] >= min_area)
    e.append(Etapa("12. Componentes conexos",
                   "Cada região ligada (vizinhança-8) vira um objeto candidato com sua caixa. O filtro não sabe o que é.",
                   f"{grandes} regiões com pelo menos {min_area} px",
                   _componentes_img(im, rot, comps, min_area)))
    return e


def salvar(etapas, nome_imagem, pasta):
    """Grava cada etapa como PNG e um etapas.txt com as explicações."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    linhas = [f"Filtros passo a passo – {nome_imagem}", ""]
    for i, e in enumerate(etapas):
        arquivo = f"{i + 1:02d}_{_slug(e.titulo)}.png"
        e.imagem.save(pasta / arquivo)
        linhas.append(f"{e.titulo}  ({arquivo})")
        linhas.append(f"  {e.explicacao}")
        linhas.append(f"  Resultado: {e.info}")
        linhas.append("")
    (pasta / "etapas.txt").write_text("\n".join(linhas), encoding="utf-8")


def _slug(titulo):
    s = unicodedata.normalize("NFD", titulo)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"^\d+\.\s*", "", s).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")
