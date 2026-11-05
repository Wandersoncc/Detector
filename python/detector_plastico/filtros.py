"""Filtros clássicos de Processamento Digital de Imagens (PDI).

Sem aprendizado: tons de cinza, HSV, equalização, mediana, gaussiano, Sobel,
bordas estilo Canny, limiarização de Otsu, morfologia, componentes conexos e a
"varinha" que usa os contornos para acelerar a anotação.

As imagens são representadas como arrays NumPy float32 no intervalo [0, 1]:
    RGB  -> array (H, W, 3)
    cinza/máscaras de intensidade -> array (H, W)
    máscaras binárias -> array (H, W) booleano

A I/O de arquivos é feita com Pillow (PIL); o processamento, com NumPy.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

LADO_MODELO = 640  # maior lado usado pela IA


# --------------------------------------------------------------------------- I/O

def abrir_rgb(caminho) -> np.ndarray:
    """Abre uma imagem com Pillow e devolve RGB float32 (H, W, 3) em [0, 1].

    Transparência é composta sobre fundo branco (como no projeto Java).
    """
    img = Image.open(caminho)
    return pil_para_rgb(img)


def pil_para_rgb(img: Image.Image) -> np.ndarray:
    """Converte uma imagem Pillow para array RGB float32 em [0, 1]."""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        fundo = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(fundo, img).convert("RGB")
    else:
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def rgb_para_pil(im: np.ndarray) -> Image.Image:
    """Converte RGB float32 [0, 1] para uma imagem Pillow de 8 bits."""
    dados = np.clip(np.rint(im * 255.0), 0, 255).astype(np.uint8)
    if dados.ndim == 2:
        return Image.fromarray(dados, mode="L")
    return Image.fromarray(dados, mode="RGB")


def cinza_para_pil(c: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(np.rint(c * 255.0), 0, 255).astype(np.uint8), mode="L")


def nivel(v: float) -> int:
    """Nível 0..255 de um valor 0..1."""
    q = int(round(v * 255.0))
    return 0 if q < 0 else (255 if q > 255 else q)


# ---------------------------------------------------------------- redimensionar

def escala_para(w: int, h: int, lado: int) -> float:
    """Escala que faz o maior lado medir ``lado`` pixels."""
    return lado / float(max(w, h))


def redimensionar(im: np.ndarray, w: int, h: int) -> np.ndarray:
    """Redimensiona (bilinear) um array RGB ou cinza para (h, w)."""
    img = rgb_para_pil(im)
    img = img.resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def espelhar(im: np.ndarray) -> np.ndarray:
    """Espelha horizontalmente (usado no aumento de dados)."""
    return im[:, ::-1].copy()


def ajustar_cores(im: np.ndarray, brilho: float, contraste: float, saturacao: float) -> np.ndarray:
    """Variação de brilho/contraste/saturação do aumento de dados."""
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    out = np.empty_like(im)
    for k, canal in enumerate((r, g, b)):
        c = y + (canal - y) * saturacao
        out[..., k] = (c - 0.5) * contraste + 0.5 + brilho
    return np.clip(out, 0.0, 1.0)


# ------------------------------------------------------------------- HSV e cinza

def cinza(im: np.ndarray) -> np.ndarray:
    """Luminância Y = 0.299 R + 0.587 G + 0.114 B."""
    return (0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]).astype(np.float32)


def hsv(im: np.ndarray):
    """Devolve (H em graus [0,360), S em [0,1], V em [0,1]) por pixel."""
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    mx = np.maximum.reduce([r, g, b])
    mn = np.minimum.reduce([r, g, b])
    d = mx - mn
    s = np.where(mx > 1e-6, d / np.where(mx > 1e-6, mx, 1.0), 0.0)
    v = mx

    h = np.zeros_like(mx)
    dz = d > 1e-6
    # evita divisão por zero fora das máscaras
    dd = np.where(dz, d, 1.0)
    mask = dz & (mx == r)
    h = np.where(mask, 60.0 * (((g - b) / dd) % 6.0), h)
    mask = dz & (mx == g)
    h = np.where(mask, 60.0 * ((b - r) / dd + 2.0), h)
    mask = dz & (mx == b)
    h = np.where(mask, 60.0 * ((r - g) / dd + 4.0), h)
    h = np.where(h < 0, h + 360.0, h)
    return h.astype(np.float32), s.astype(np.float32), v.astype(np.float32)


def matiz(r, g, b, mx, mn):
    """Matiz (H) em graus [0,360) para valores escalares (usado nos descritores)."""
    d = mx - mn
    if d < 1e-6:
        return 0.0
    if mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * ((b - r) / d + 2.0)
    else:
        h = 60.0 * ((r - g) / d + 4.0)
    return h + 360.0 if h < 0 else h


# ------------------------------------------------------------- filtragem espacial

_K5 = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0


def _conv1d_borda(c: np.ndarray, kernel: np.ndarray, eixo: int) -> np.ndarray:
    """Convolução 1D com borda replicada (clamp), ao longo de um eixo."""
    raio = len(kernel) // 2
    out = np.zeros_like(c)
    n = c.shape[eixo]
    for k, peso in enumerate(kernel):
        desl = k - raio
        idx = np.clip(np.arange(n) + desl, 0, n - 1)
        if eixo == 0:
            out += peso * c[idx, :]
        else:
            out += peso * c[:, idx]
    return out


def desfoque(c: np.ndarray) -> np.ndarray:
    """Desfoque gaussiano 5x5 separável (núcleo 1 4 6 4 1)."""
    return _conv1d_borda(_conv1d_borda(c, _K5, 1), _K5, 0)


def sobel(c: np.ndarray):
    """Gradientes de Sobel (gx, gy), normalizados para que um degrau 0->1 valha 1."""
    def desl(a, dy, dx):
        return np.roll(np.roll(a, -dy, axis=0), -dx, axis=1)

    # replica bordas antes de rolar para imitar o clamp do Java
    cp = np.pad(c, 1, mode="edge")
    a = cp[0:-2, 0:-2]; b = cp[0:-2, 1:-1]; d = cp[0:-2, 2:]
    e = cp[1:-1, 0:-2]; f = cp[1:-1, 2:]
    g = cp[2:, 0:-2]; hh = cp[2:, 1:-1]; k = cp[2:, 2:]
    gx = ((d + 2 * f + k) - (a + 2 * e + g)) / 4.0
    gy = ((g + 2 * hh + k) - (a + 2 * b + d)) / 4.0
    return gx.astype(np.float32), gy.astype(np.float32)


def equalizar(c: np.ndarray) -> np.ndarray:
    """Equalização de histograma: espalha os tons por toda a faixa 0..255."""
    q = np.clip(np.rint(c * 255.0), 0, 255).astype(np.int32)
    hist = np.bincount(q.ravel(), minlength=256).astype(np.float64)
    n = c.size
    cdf = np.cumsum(hist)
    nao_zero = hist > 0
    cdf_min = cdf[np.argmax(nao_zero)] if nao_zero.any() else 0
    if n == cdf_min:
        mapa = np.arange(256, dtype=np.float32) / 255.0
    else:
        mapa = np.maximum(0.0, (cdf - cdf_min) / float(n - cdf_min)).astype(np.float32)
    return mapa[q]


def equalizar_cores(im: np.ndarray) -> np.ndarray:
    """Equaliza só o brilho e mantém as cores (realça em água turva/escura)."""
    y = cinza(im)
    ye = equalizar(y)
    out = np.empty_like(im)
    escuro = y < 1e-3
    f = np.where(escuro, 1.0, ye / np.where(escuro, 1.0, y))
    for k in range(3):
        canal = np.where(escuro, ye, im[..., k] * f)
        out[..., k] = np.clip(canal, 0.0, 1.0)
    return out


def mediana(c: np.ndarray, raio: int = 2) -> np.ndarray:
    """Filtro de mediana em janela (2r+1)x(2r+1): remove ruído sem borrar bordas."""
    lado = 2 * raio + 1
    cp = np.pad(c, raio, mode="edge")
    janelas = []
    for dy in range(lado):
        for dx in range(lado):
            janelas.append(cp[dy:dy + c.shape[0], dx:dx + c.shape[1]])
    pilha = np.stack(janelas, axis=0)
    return np.median(pilha, axis=0).astype(np.float32)


def _percentil_grad(mag: np.ndarray, q: float) -> float:
    """Percentil via histograma de 1024 faixas (como no Java)."""
    mx = max(1e-6, float(mag.max()))
    idx = np.minimum(1023, (mag / mx * 1023).astype(np.int32))
    hist = np.bincount(idx.ravel(), minlength=1024)
    alvo = q * mag.size
    acc = np.cumsum(hist)
    pos = np.argmax(acc >= alvo)
    return pos / 1023.0 * mx


def bordas(im: np.ndarray) -> np.ndarray:
    """Detector de bordas estilo Canny (o filtro 'P&B contornos').

    Sobel sobre o cinza desfocado, supressão de não-máximos e histerese com
    limiares automáticos pelo percentil do gradiente. Devolve máscara booleana.
    """
    h, w = im.shape[:2]
    c = desfoque(cinza(im))
    gx, gy = sobel(c)
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    alto = max(0.06, _percentil_grad(mag, 0.85))
    baixo = alto * 0.45

    ang = np.degrees(np.arctan2(gy, gx))
    ang = np.where(ang < 0, ang + 180.0, ang)

    magp = np.pad(mag, 1, mode="constant")
    centro = magp[1:-1, 1:-1]
    # vizinhos conforme a direção do gradiente
    esq = magp[1:-1, 0:-2]; dir_ = magp[1:-1, 2:]
    cima = magp[0:-2, 1:-1]; baixo_ = magp[2:, 1:-1]
    d1a = magp[0:-2, 0:-2]; d1b = magp[2:, 2:]      # diagonal "\"
    d2a = magp[0:-2, 2:]; d2b = magp[2:, 0:-2]      # diagonal "/"

    horiz = (ang < 22.5) | (ang >= 157.5)
    diag1 = (ang >= 22.5) & (ang < 67.5)
    vert = (ang >= 67.5) & (ang < 112.5)
    diag2 = (ang >= 112.5) & (ang < 157.5)

    n1 = np.select([horiz, diag1, vert, diag2], [esq, d1a, cima, d2a], default=0.0)
    n2 = np.select([horiz, diag1, vert, diag2], [dir_, d1b, baixo_, d2b], default=0.0)

    pico = (centro >= n1) & (centro >= n2) & (centro >= baixo)
    forte = pico & (centro >= alto)
    fraco = pico & (centro >= baixo) & (centro < alto)
    # zera a moldura de 1 px (o Java ignora as bordas externas)
    forte[0, :] = forte[-1, :] = forte[:, 0] = forte[:, -1] = False
    fraco[0, :] = fraco[-1, :] = fraco[:, 0] = fraco[:, -1] = False

    return _histerese(forte, fraco)


def _histerese(forte: np.ndarray, fraco: np.ndarray) -> np.ndarray:
    """Bordas fracas só permanecem se conectadas (vizinhança-8) a uma forte."""
    from collections import deque

    h, w = forte.shape
    out = forte.copy()
    pilha = deque(zip(*np.nonzero(forte)))
    while pilha:
        y, x = pilha.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and fraco[yy, xx] and not out[yy, xx]:
                    out[yy, xx] = True
                    pilha.append((yy, xx))
    return out


def limiar_otsu(c: np.ndarray) -> float:
    """Limiar de Otsu: o tom (0..1) que maximiza a variância entre classes."""
    q = np.clip(np.rint(c * 255.0), 0, 255).astype(np.int32)
    hist = np.bincount(q.ravel(), minlength=256).astype(np.float64)
    total = c.size
    niveis = np.arange(256, dtype=np.float64)
    soma_total = float((niveis * hist).sum())
    n_esc = np.cumsum(hist)
    soma_esc = np.cumsum(niveis * hist)
    n_cla = total - n_esc
    valido = (n_esc > 0) & (n_cla > 0)
    m_esc = np.divide(soma_esc, n_esc, out=np.zeros_like(soma_esc), where=n_esc > 0)
    m_cla = np.divide(soma_total - soma_esc, n_cla, out=np.zeros_like(soma_esc), where=n_cla > 0)
    entre = n_esc * n_cla * (m_esc - m_cla) ** 2
    entre = np.where(valido, entre, -1.0)
    t = int(np.argmax(entre))
    return t / 255.0


def faixas_hsv(im: np.ndarray) -> np.ndarray:
    """Faixas fixas de HSV (sem aprendizado): claro e dessaturado OU cor viva quente."""
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    mx = np.maximum.reduce([r, g, b])
    mn = np.minimum.reduce([r, g, b])
    s = np.where(mx > 1e-6, (mx - mn) / np.where(mx > 1e-6, mx, 1.0), 0.0)
    claro = (s < 0.22) & (mx > 0.72)
    h, _, _ = hsv(im)
    cor_viva = (~claro) & (s > 0.5) & (mx > 0.45) & ((h < 50.0) | (h > 290.0))
    return claro | cor_viva


def filtro_hsv_classico(im: np.ndarray) -> np.ndarray:
    """Faixas de HSV já limpas por abertura morfológica. Comparação com a IA."""
    return abrir(faixas_hsv(im))


# --------------------------------------------------------------------- morfologia

def dilatar(m: np.ndarray) -> np.ndarray:
    """Dilatação com elemento 3x3 (vizinhança-8)."""
    mp = np.pad(m, 1, mode="constant", constant_values=False)
    out = np.zeros_like(m)
    for dy in range(3):
        for dx in range(3):
            out |= mp[dy:dy + m.shape[0], dx:dx + m.shape[1]]
    return out


def erodir(m: np.ndarray) -> np.ndarray:
    """Erosão 3x3 (implementada como a negação da dilatação do complemento)."""
    return ~dilatar(~m)


def abrir(m: np.ndarray) -> np.ndarray:
    """Abertura: erosão seguida de dilatação (apaga pontos soltos)."""
    return dilatar(erodir(m))


def fechar(m: np.ndarray) -> np.ndarray:
    """Fechamento: dilatação seguida de erosão (fecha buracos e une partes)."""
    return erodir(dilatar(m))


# ------------------------------------------------------------------ componentes

def componentes(m: np.ndarray, com_rotulos: bool = False):
    """Componentes conexos (vizinhança-8).

    Devolve uma lista de caixas ``[min_x, min_y, max_x, max_y, area]`` e, se
    ``com_rotulos`` for True, também uma matriz de rótulos (0 = fundo, 1.. = id).
    """
    h, w = m.shape
    rotulos = np.zeros((h, w), dtype=np.int32)
    comps = []
    estrutura = None
    try:
        from scipy import ndimage  # opcional, acelera muito

        estrutura = np.ones((3, 3), dtype=bool)
        lab, n = ndimage.label(m, structure=estrutura)
        rotulos = lab.astype(np.int32)
        if n:
            objs = ndimage.find_objects(lab)
            for i, sl in enumerate(objs, start=1):
                if sl is None:
                    continue
                ys, xs = sl
                area = int((lab[sl] == i).sum())
                comps.append([xs.start, ys.start, xs.stop - 1, ys.stop - 1, area])
    except Exception:
        # fallback em Python puro (BFS), sem dependência externa
        from collections import deque

        visto = np.zeros((h, w), dtype=bool)
        ys, xs = np.nonzero(m)
        id_atual = 0
        for y0, x0 in zip(ys, xs):
            if visto[y0, x0]:
                continue
            id_atual += 1
            fila = deque([(y0, x0)])
            visto[y0, x0] = True
            minx = maxx = x0
            miny = maxy = y0
            area = 0
            while fila:
                y, x = fila.popleft()
                rotulos[y, x] = id_atual
                area += 1
                minx = min(minx, x); maxx = max(maxx, x)
                miny = min(miny, y); maxy = max(maxy, y)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w and m[yy, xx] and not visto[yy, xx]:
                            visto[yy, xx] = True
                            fila.append((yy, xx))
            comps.append([minx, miny, maxx, maxy, area])
    if com_rotulos:
        return comps, rotulos
    return comps


# ------------------------------------------------------------------------ varinha

def varinha(im: np.ndarray, bordas_mask: np.ndarray, sx: int, sy: int,
            tolerancia: float, max_pixels: int):
    """Varinha mágica para acelerar a anotação.

    A partir do clique (sx, sy), preenche a região de cor parecida contida dentro
    dos contornos detectados. Devolve ``None`` se a região "vazar" (contorno
    aberto) e passar de ``max_pixels``.
    """
    from collections import deque

    h, w = im.shape[:2]
    barreira = dilatar(bordas_mask)

    semente = None
    for raio in range(5):
        for dy in range(-raio, raio + 1):
            for dx in range(-raio, raio + 1):
                x, y = sx + dx, sy + dy
                if 0 <= x < w and 0 <= y < h and not barreira[y, x]:
                    semente = (y, x)
                    break
            if semente:
                break
        if semente:
            break

    reg = np.zeros((h, w), dtype=bool)
    if semente is None:
        return reg
    y0, x0 = semente

    y1, y2 = max(0, y0 - 2), min(h, y0 + 3)
    x1, x2 = max(0, x0 - 2), min(w, x0 + 3)
    ref = im[y1:y2, x1:x2].reshape(-1, 3).mean(axis=0)
    tol2 = tolerancia * tolerancia

    fila = deque([semente])
    reg[y0, x0] = True
    contagem = 1
    while fila:
        if contagem > max_pixels:
            return None
        y, x = fila.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and not reg[yy, xx] and not barreira[yy, xx]:
                dif = im[yy, xx] - ref
                if float(dif @ dif) <= tol2:
                    reg[yy, xx] = True
                    contagem += 1
                    fila.append((yy, xx))

    # inclui o contorno e fecha buracos pequenos (reflexos, rótulos)
    for _ in range(2):
        d = dilatar(reg)
        reg |= d & barreira
    reg = erodir(dilatar(dilatar(reg)))
    return erodir(reg)
