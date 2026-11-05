"""Processamento de muitas imagens de uma vez (detecção em lote)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import detector as det
from . import filtros, projeto as projeto_mod


def _fonte(tam):
    for nome in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(nome, tam)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def _altura_fonte(fonte):
    """Ascender e descender da fonte (altura de linha estável)."""
    try:
        asc, desc = fonte.getmetrics()
        return asc, desc
    except Exception:  # noqa: BLE001
        return 12, 3


def desenhar(img: np.ndarray, resultado, limiar) -> Image.Image:
    """Cópia da imagem com o mapa de calor e as caixas das regiões detectadas."""
    base = filtros.rgb_para_pil(img).convert("RGBA")
    w, h = base.size

    # mapa de calor ampliado para o tamanho da grade
    calor = resultado.imagem_calor(limiar)  # (gh, gw, 4)
    calor_img = Image.fromarray(calor, mode="RGBA")
    lg = max(1, round(resultado.largura_grade()))
    ag = max(1, round(resultado.altura_grade()))
    calor_img = calor_img.resize((lg, ag), Image.BILINEAR)
    # meia opacidade
    alfa = calor_img.split()[3].point(lambda a: int(a * 0.5))
    calor_img.putalpha(alfa)
    base.alpha_composite(calor_img, (0, 0))

    draw = ImageDraw.Draw(base)
    esp = max(2, round(max(w, h) / 320.0))
    fonte = _fonte(max(12, max(w, h) // 55))

    # altura de uma linha de texto para esta fonte (independe do conteúdo)
    asc, desc = _altura_fonte(fonte)
    lin = asc + desc

    regs = resultado.regioes(limiar)
    for reg in regs:
        x, y, rw, rh = reg.caixa
        draw.rectangle([x, y, x + rw, y + rh], outline=(255, 40, 40, 255), width=esp)
        txt = f"plástico {reg.confianca * 100:.0f}%"
        tw = draw.textlength(txt, font=fonte)
        cap = lin + 6
        ty = y - cap if y - cap >= 0 else y
        draw.rectangle([x, ty, x + tw + 8, ty + cap], fill=(255, 40, 40, 255))
        draw.text((x + 4, ty + 3), txt, fill=(255, 255, 255, 255), font=fonte)

    resumo = (f"IA: {len(regs)} região(ões) · "
              f"{resultado.area_percentual(limiar):.1f}% da imagem com plástico")
    cap = lin + 8
    draw.rectangle([0, h - cap, w, h], fill=(0, 0, 0, 170))
    draw.text((8, h - cap + 3), resumo, fill=(255, 255, 255, 255), font=fonte)
    return base.convert("RGB")


def detectar_pasta(modelo, imagens, saida, limiar, log):
    """Detecta plástico em todas as imagens; grava uma cópia marcada e relatorio.csv."""
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    linhas = []
    ok = 0
    total = len(imagens)
    for i, f in enumerate(imagens):
        f = Path(f)
        try:
            img = projeto_mod.Projeto.ler_imagem(f)
        except Exception as e:  # noqa: BLE001
            linhas.append([f.name, "erro", "", "", str(e)])
            continue
        r = det.detectar(modelo, img)
        regs = r.regioes(limiar)
        base = re.sub(r"\.[^.]+$", "", f.name)
        desenhar(img, r, limiar).save(saida / f"{base}_ia.jpg", quality=92)
        conf = max((g.confianca for g in regs), default=0.0)
        linhas.append([f.name, "sim" if regs else "nao", len(regs),
                       f"{r.area_percentual(limiar):.1f}", f"{conf * 100:.0f}"])
        ok += 1
        if (i + 1) % 10 == 0 or i == total - 1:
            log(f"  {i + 1}/{total} imagens analisadas")

    projeto_mod.Projeto.escrever_csv(
        saida / "relatorio.csv",
        ["imagem", "tem_plastico", "regioes", "area_plastico_%", "maior_confianca_%"],
        linhas, anexar=False)
    return ok


def rankear_incerteza(modelo, imagens, log):
    """Imagens em que a IA mais ficou em dúvida (as que mais ensinam se anotadas)."""
    out = []
    for f in imagens:
        f = Path(f)
        try:
            r = det.detectar(modelo, projeto_mod.Projeto.ler_imagem(f))
            out.append((f, r.incerteza, r.area_percentual(modelo.limiar)))
        except Exception:  # noqa: BLE001
            log(f"  não foi possível ler {f.name}")
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def etapa_ia(numero, img: np.ndarray, resultado, limiar):
    """Última etapa da tela de filtros: o que a rede neural vê na imagem."""
    from .etapas import Etapa

    regs = resultado.regioes(limiar)
    info = (f"{len(regs)} região(ões) · {resultado.area_percentual(limiar):.1f}% "
            f"da imagem com plástico (limiar {limiar * 100:.0f}%)")
    return Etapa(f"{numero}. Visão computacional (IA)",
                 "A rede neural treinada com as anotações dá a probabilidade de plástico "
                 "em cada célula; as regiões acima do limiar viram detecções.",
                 info, desenhar(img, resultado, limiar))
