"""Treina a rede neural a partir das imagens anotadas.

Mede a eficiência com validação cruzada (cada imagem é avaliada por um modelo que
não a viu no treino) e também compara com o filtro HSV clássico. Faz aumento de
dados (espelhamento, escala e variação de cor) e escolhe o limiar que maximiza o F1.
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np

from . import descritores, filtros, projeto as projeto_mod
from .metricas import Metricas
from .modelo import LADO_BASE, Modelo, normalizar
from .rede_neural import RedeNeural

# aumento de dados: (escala, espelhar, brilho, contraste, saturação)
VARIANTES = [
    (1.00, 0, 0.00, 1.00, 1.00),
    (1.00, 1, 0.06, 1.10, 0.85),
    (0.75, 0, -0.06, 0.90, 1.15),
    (1.30, 1, 0.00, 1.05, 1.05),
]
FRAC_POS = 0.4      # célula é plástico se >= 40% pintada
FRAC_NEG = 0.1      # fundo se <= 10%; o meio-termo é ignorado
FRAC_CLASSICO = 0.3
LOTE = 64
MAX_AMOSTRAS_EPOCA = 30000
MAX_EPOCAS = 40
MIN_EPOCAS = 6
PACIENCIA = 6
MIN_NEGATIVOS = 100
CAMADAS = [descritores.TAMANHO, 64, 32, 1]


class ResultadoTreino:
    def __init__(self):
        self.imagens = 0
        self.validado = False
        self.ia = Metricas()
        self.classico = Metricas()
        self.proprias = Metricas()
        self.importadas = Metricas()
        self.limiar = 0.5
        self.segundos = 0.0
        self.por_imagem = []  # (nome, Metricas)


class _DadosImagem:
    __slots__ = ("nome", "x_treino", "y_treino", "x_aval", "y_aval",
                 "classico", "prob_oof")

    def __init__(self, nome):
        self.nome = nome
        self.x_treino = None
        self.y_treino = None
        self.x_aval = None
        self.y_aval = None
        self.classico = None
        self.prob_oof = None


# --------------------------------------------------------------- preparação

def _contar_por_subcelula(m, gw, gh):
    """Por sub-célula: (pixels de plástico, pixels 'ignorar', total)."""
    P = descritores.P
    h, w = m.shape
    pl = np.zeros((gh, gw), dtype=np.int32)
    ig = np.zeros((gh, gw), dtype=np.int32)
    tot = np.zeros((gh, gw), dtype=np.int32)
    sy = np.arange(h) // P
    sx = np.arange(w) // P
    SX, SY = np.meshgrid(sx, sy)
    plano = (SY * gw + SX).ravel()
    np.add.at(tot.ravel(), plano, 1)
    np.add.at(pl.ravel(), plano, (m == projeto_mod.PLASTICO).astype(np.int32).ravel())
    np.add.at(ig.ravel(), plano, (m == projeto_mod.IGNORAR).astype(np.int32).ravel())
    return pl, ig, tot


def _janelas_soma(a, cel):
    """Soma deslizante cel x cel (passo 1) sobre uma matriz de sub-células."""
    integ = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.int64)
    integ[1:, 1:] = np.cumsum(np.cumsum(a.astype(np.int64), axis=0), axis=1)
    gh, gw = a.shape
    ny, nx = gh - cel + 1, gw - cel + 1
    if ny <= 0 or nx <= 0:
        return np.zeros((max(0, ny), max(0, nx)), dtype=np.int64)
    s = (integ[cel:cel + ny, cel:cel + nx] - integ[0:ny, cel:cel + nx]
         - integ[cel:cel + ny, 0:nx] + integ[0:ny, 0:nx])
    return s


def _rotulos_grade(pl, ig, tot, cel):
    """Matriz de rótulos por posição de célula: 1 plástico, 0 fundo, -1 ignorar."""
    spl = _janelas_soma(pl, cel)
    sig = _janelas_soma(ig, cel)
    stot = _janelas_soma(tot, cel)
    rot = np.full(spl.shape, -1, dtype=np.int8)
    denom = (stot - sig).astype(np.float64)
    valido = (stot > 0) & (sig * 2 <= stot) & (denom > 0)
    frac = np.divide(spl, denom, out=np.zeros_like(denom), where=denom > 0)
    rot = np.where(valido & (frac >= FRAC_POS), 1, rot)
    rot = np.where(valido & (frac <= FRAC_NEG), 0, rot)
    return rot, (np.divide(spl, np.where(stot > 0, stot, 1), out=np.zeros_like(denom), where=True))


def _preparar(projeto, arq, n_imagens, rng):
    img = projeto.ler_imagem(arq)
    h0, w0 = img.shape[:2]
    mascara = projeto.carregar_mascara(arq, w0, h0)
    if mascara is None:
        mascara = np.zeros((h0, w0), dtype=np.uint8)
    d = _DadosImagem(str(getattr(arq, "name", arq)))
    F, CEL = descritores.TAMANHO, descritores.CEL
    limite_pos = max(60, min(600, 2_000_000 // max(1, n_imagens * len(VARIANTES) * 3)))

    blocos_x, blocos_y = [], []

    for v, var in enumerate(VARIANTES):
        esc = filtros.escala_para(w0, h0, LADO_BASE) * var[0]
        w = max(32, round(w0 * esc))
        h = max(32, round(h0 * esc))
        iv = filtros.redimensionar(img, w, h)
        mv = projeto.redimensionar_mascara(mascara, w, h)
        if var[1] > 0:
            iv = filtros.espelhar(iv)
            mv = projeto.espelhar_mascara(mv)
        if v > 0:
            iv = filtros.ajustar_cores(iv, var[2], var[3], var[4])
        g = descritores.calcular(iv)
        pl, ig, tot = _contar_por_subcelula(mv, g.gw, g.gh)

        if v == 0:
            # avaliação: células sem sobreposição com rótulo definido
            rot_passo, _ = _rotulos_grade(pl, ig, tot, CEL)
            cl = filtros.filtro_hsv_classico(iv)
            cl_m = np.where(cl, projeto_mod.PLASTICO, projeto_mod.FUNDO).astype(np.uint8)
            plc, igc, totc = _contar_por_subcelula(cl_m, g.gw, g.gh)
            _, frac_cl = _rotulos_grade(plc, igc, totc, CEL)

            aval = []
            for cy in range(0, g.gh - CEL + 1, CEL):
                for cx in range(0, g.gw - CEL + 1, CEL):
                    r = int(rot_passo[cy, cx])
                    if r >= 0:
                        aval.append((cx, cy, r))
            limite_aval = None if n_imagens <= 300 else max(200, 300_000 // n_imagens)
            if limite_aval is not None and len(aval) > limite_aval:
                rng.shuffle(aval)
                aval = aval[:limite_aval]
            d.x_aval = descritores.extrair_muitas(g, [(a[0], a[1]) for a in aval])
            d.y_aval = np.array([a[2] for a in aval], dtype=np.int8)
            d.classico = np.array([frac_cl[a[1], a[0]] for a in aval], dtype=np.float32)

        # treino: plástico em todas as posições; fundo em passo de CEL
        rot_treino, _ = _rotulos_grade(pl, ig, tot, CEL)
        pos, neg = [], []
        ny, nx = rot_treino.shape
        for cy in range(ny):
            for cx in range(nx):
                r = int(rot_treino[cy, cx])
                if r == 1:
                    pos.append((cx, cy))
                elif r == 0 and cx % CEL == 0 and cy % CEL == 0:
                    neg.append((cx, cy))
        rng.shuffle(pos)
        rng.shuffle(neg)
        npos = min(len(pos), limite_pos)
        nneg = min(len(neg), max(MIN_NEGATIVOS, 2 * npos))
        sel = pos[:npos] + neg[:nneg]
        bx = descritores.extrair_muitas(g, sel)
        by = np.array([1] * npos + [0] * nneg, dtype=np.int8)
        blocos_x.append(bx)
        blocos_y.append(by)

    d.x_treino = np.concatenate(blocos_x, axis=0) if blocos_x else np.zeros((0, F), np.float32)
    d.y_treino = np.concatenate(blocos_y, axis=0) if blocos_y else np.zeros((0,), np.int8)
    return d


# ----------------------------------------------------------------- normalização

def _normalizacao(dados):
    F = descritores.TAMANHO
    s = np.zeros(F, dtype=np.float64)
    s2 = np.zeros(F, dtype=np.float64)
    n = 0
    for d in dados:
        if len(d.x_treino):
            s += d.x_treino.sum(axis=0)
            s2 += (d.x_treino.astype(np.float64) ** 2).sum(axis=0)
            n += len(d.x_treino)
    if n == 0:
        return np.zeros(F, np.float32), np.ones(F, np.float32)
    media = (s / n)
    var = s2 / n - media * media
    desvio = np.maximum(0.01, np.sqrt(np.maximum(0.0, var)))
    return media.astype(np.float32), desvio.astype(np.float32)


# --------------------------------------------------------------- treino da rede

def _treinar_rede(treino, parada, epocas_fixas, media, desvio, rng, log, nome):
    F = descritores.TAMANHO
    X = np.concatenate([d.x_treino for d in treino], axis=0) if treino else np.zeros((0, F), np.float32)
    y = np.concatenate([d.y_treino for d in treino], axis=0) if treino else np.zeros((0,), np.int8)
    total = len(X)
    positivas = int((y == 1).sum())
    if total == 0:
        return RedeNeural(CAMADAS, rng), (epocas_fixas or MAX_EPOCAS)

    Xn = normalizar(X, media, desvio)
    peso_pos = 1.0 if positivas == 0 else max(1.0, min(4.0, (total - positivas) / positivas))
    pesos = np.where(y == 1, peso_pos, 1.0).astype(np.float32)

    # conjunto de parada (early stopping)
    Xp = Yp = None
    if parada:
        Xp = np.concatenate([d.x_aval for d in parada if len(d.x_aval)], axis=0) if any(len(d.x_aval) for d in parada) else None
        Yp = np.concatenate([d.y_aval for d in parada if len(d.y_aval)], axis=0) if Xp is not None else None
        if Xp is not None:
            Xp = normalizar(Xp, media, desvio)

    rede = RedeNeural(CAMADAS, rng)
    melhor_rede = None
    melhor_perda = float("inf")
    melhor_epoca = epocas_fixas or MAX_EPOCAS
    sem_melhora = 0
    max_epocas = epocas_fixas if epocas_fixas > 0 else MAX_EPOCAS

    for ep in range(1, max_epocas + 1):
        ordem = rng.permutation(total)
        usar = min(total, MAX_AMOSTRAS_EPOCA)
        perda = 0.0
        nlotes = 0
        for ini in range(0, usar, LOTE):
            fim = min(usar, ini + LOTE)
            idx = ordem[ini:fim]
            perda += rede.passo_lote(Xn[idx], y[idx].astype(np.float64), pesos[idx], rng)
            nlotes += 1
        if log and (ep == 1 or ep % 5 == 0 or ep == max_epocas):
            log(f"  {nome} – época {ep}/{max_epocas}, perda {perda / max(1, nlotes):.4f}")

        if epocas_fixas <= 0 and Xp is not None:
            pv = _perda_balanceada(rede, Xp, Yp)
            if pv < melhor_perda - 1e-4:
                melhor_perda = pv
                melhor_rede = rede.copia()
                melhor_epoca = ep
                sem_melhora = 0
            else:
                sem_melhora += 1
                if sem_melhora >= PACIENCIA and ep >= MIN_EPOCAS:
                    break

    return (melhor_rede or rede), melhor_epoca


def _perda_balanceada(rede, Xp, Yp):
    p = np.clip(rede.prever(Xp), 1e-6, 1 - 1e-6)
    perda = np.where(Yp == 1, -np.log(p), -np.log(1 - p))
    a = perda[Yp == 0].mean() if (Yp == 0).any() else 0.0
    b = perda[Yp == 1].mean() if (Yp == 1).any() else 0.0
    if not (Yp == 0).any() or not (Yp == 1).any():
        return a + b
    return (a + b) / 2


def _rodar_dobra(dados, dobra, f, k, log):
    rng = np.random.default_rng(77 + f)
    treino = [dados[i] for i in range(len(dados)) if dobra[i] != f]
    teste = [dados[i] for i in range(len(dados)) if dobra[i] == f]
    emb = list(treino)
    rng.shuffle(emb)
    n_parada = max(1, round(len(treino) * 0.15)) if len(treino) >= 3 else 0
    parada = emb[:n_parada]
    efetivo = emb[n_parada:]
    media, desvio = _normalizacao(efetivo)
    rede, melhor = _treinar_rede(efetivo, parada, 0, media, desvio, rng, None, None)
    for d in teste:
        if len(d.x_aval):
            d.prob_oof = rede.prever(normalizar(d.x_aval, media, desvio))
        else:
            d.prob_oof = np.zeros(0, dtype=np.float32)
    if log:
        log(f"  parte {f + 1}/{k} concluída (melhor época: {melhor})")
    return melhor


# -------------------------------------------------------------------- avaliação

def _avaliar(dados, res, importadas):
    melhor_t, melhor_f1 = 0.5, -1.0
    for i in range(29):
        t = 0.15 + i * 0.025
        m = Metricas()
        for d in dados:
            if len(d.y_aval):
                m.somar_lote(d.prob_oof >= t, d.y_aval == 1)
        if m.f1() > melhor_f1 + 1e-9:
            melhor_f1 = m.f1()
            melhor_t = t
    res.limiar = melhor_t
    for d in dados:
        m = Metricas()
        if len(d.y_aval):
            real = d.y_aval == 1
            m.somar_lote(d.prob_oof >= melhor_t, real)
            res.classico.somar_lote(d.classico >= FRAC_CLASSICO, real)
        res.ia.somar_metricas(m)
        (res.importadas if d.nome in importadas else res.proprias).somar_metricas(m)
        res.por_imagem.append((d.nome, m))
    res.por_imagem.sort(key=lambda o: o[1].f1())


# ------------------------------------------------------------------- relatórios

def _gravar_relatorios(projeto, res):
    if not res.validado:
        if projeto.arquivo_desempenho().is_file():
            projeto.arquivo_desempenho().unlink()
        return
    linhas = []
    for nome, m in res.por_imagem:
        linhas.append([nome, f"{m.f1() * 100:.1f}", f"{m.precisao() * 100:.1f}",
                       f"{m.revocacao() * 100:.1f}", m.vp + m.fn, m.total(),
                       m.vp, m.fp, m.fn, m.vn])
    projeto.escrever_csv(
        projeto.arquivo_desempenho(),
        ["imagem", "f1_%", "precisao_%", "revocacao_%", "celulas_plastico",
         "celulas_avaliadas", "vp", "fp", "fn", "vn"],
        linhas, anexar=False)

    def pct(m):
        return "" if m.total() == 0 else f"{m.f1() * 100:.1f}"

    hist = projeto.arquivo_historico()
    cab = ["data", "imagens_anotadas", "f1_ia_%", "precisao_%", "revocacao_%", "iou_%",
           "f1_filtro_hsv_%", "limiar", "segundos", "f1_imagens_proprias_%", "f1_datasets_importados_%"]
    linha = [datetime.now().strftime("%d/%m/%Y %H:%M"), res.imagens,
             f"{res.ia.f1() * 100:.1f}", f"{res.ia.precisao() * 100:.1f}",
             f"{res.ia.revocacao() * 100:.1f}", f"{res.ia.iou() * 100:.1f}",
             f"{res.classico.f1() * 100:.1f}", f"{res.limiar:.3f}", f"{res.segundos:.0f}",
             pct(res.proprias), pct(res.importadas)]
    projeto.escrever_csv(hist, cab, [linha], anexar=hist.is_file())


# ------------------------------------------------------------------------ treino

def treinar(projeto, log):
    t0 = time.time()
    anotadas = projeto.imagens_anotadas()
    n = len(anotadas)
    if n == 0:
        raise RuntimeError("Nenhuma imagem anotada ainda. Anote pelo menos 1 imagem "
                           "(o ideal é 10 ou mais) e salve.")

    log(f"Preparando {n} imagens anotadas ({len(VARIANTES)} variações de cada uma)...")
    dados = []
    amostras = positivas = 0
    for i, arq in enumerate(anotadas):
        d = _preparar(projeto, arq, n, np.random.default_rng(1000 + i))
        dados.append(d)
        amostras += len(d.y_treino)
        positivas += int((d.y_treino == 1).sum())
        if (i + 1) % max(1, n // 4) == 0 or i == n - 1:
            log(f"  {i + 1}/{n} imagens preparadas")
    if positivas == 0:
        raise RuntimeError("As anotações não têm nenhuma área marcada como plástico.")
    celulas = sum(len(d.y_aval) for d in dados)
    log(f"Amostras de treino: {amostras} ({positivas} de plástico). Células de avaliação: {celulas}.")

    res = ResultadoTreino()
    res.imagens = n
    epocas_final = 20

    if n >= 4:
        k = min(5, n)
        ordem = list(range(n))
        np.random.default_rng(7).shuffle(ordem)
        dobra = [0] * n
        for i in range(n):
            dobra[ordem[i]] = i % k
        log(f"Validação cruzada em {k} partes (cada imagem é testada por um modelo "
            f"que não a viu no treino)...")
        melhores = sorted(_rodar_dobra(dados, dobra, f, k, log) for f in range(k))
        epocas_final = melhores[k // 2]
        _avaliar(dados, res, projeto.imagens_de_datasets())
        res.validado = True
        log(f"Eficiência na validação: F1 = {res.ia.f1() * 100:.1f}% "
            f"(precisão {res.ia.precisao() * 100:.1f}%, revocação {res.ia.revocacao() * 100:.1f}%)."
            f" Filtro HSV clássico: F1 = {res.classico.f1() * 100:.1f}%.")
        if res.proprias.total() > 0 and res.importadas.total() > 0:
            log(f"  por origem: imagens próprias F1 = {res.proprias.f1() * 100:.1f}% · "
                f"datasets importados F1 = {res.importadas.f1() * 100:.1f}%")
    else:
        log("Com menos de 4 imagens anotadas o modelo é treinado, mas a eficiência não é medida.")

    epocas_final = max(MIN_EPOCAS, min(MAX_EPOCAS, epocas_final))
    log(f"Treinando o modelo final com todas as {n} imagens ({epocas_final} épocas)...")
    media, desvio = _normalizacao(dados)
    rede, _ = _treinar_rede(dados, [], epocas_final, media, desvio,
                            np.random.default_rng(99), log, "modelo final")

    m = Modelo()
    m.media, m.desvio, m.rede = media, desvio, rede
    m.limiar = res.limiar
    m.data_treino = time.time()
    m.n_imagens = n
    if res.validado:
        m.f1 = res.ia.f1()
        m.precisao = res.ia.precisao()
        m.revocacao = res.ia.revocacao()
        m.iou = res.ia.iou()
        m.f1_classico = res.classico.f1()
        m.f1_proprias = res.proprias.f1() if res.proprias.total() > 0 else -1
        m.f1_importadas = res.importadas.f1() if res.importadas.total() > 0 else -1
    m.salvar(projeto.arquivo_modelo())
    res.segundos = time.time() - t0
    _gravar_relatorios(projeto, res)
    log(f"Modelo salvo em {projeto.arquivo_modelo().name} ({res.segundos:.0f} s).")
    return res
