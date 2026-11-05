"""Autoteste: confere se os filtros, descritores, rede e dados estão coerentes."""

from __future__ import annotations

import numpy as np

from . import descritores, detector, filtros
from .modelo import Modelo
from .rede_neural import RedeNeural


def rodar(projeto, log=print) -> bool:
    ok = True

    def checar(cond, msg):
        nonlocal ok
        log(("  [OK] " if cond else "  [FALHOU] ") + msg)
        ok = ok and cond

    log("Autoteste do Detector de Plástico (Python)")

    # imagem sintética: metade esquerda escura (água), um quadrado claro (plástico)
    rng = np.random.default_rng(0)
    im = (rng.random((128, 192, 3)).astype(np.float32) * 0.2 + 0.1)
    im[40:80, 120:170] = 0.95  # região clara

    log("Filtros de PDI:")
    c = filtros.cinza(im)
    checar(c.shape == (128, 192), "conversão para tons de cinza")
    b = filtros.bordas(im)
    checar(b.dtype == bool and b.any(), "detecção de bordas (Canny)")
    t = filtros.limiar_otsu(c)
    checar(0.0 <= t <= 1.0, f"limiar de Otsu = {t:.2f}")
    fx = filtros.faixas_hsv(im)
    checar(fx[50, 140], "faixa HSV marca a região clara como plástico")
    comps = filtros.componentes(filtros.abrir(fx))
    checar(len(comps) >= 1, f"componentes conexos ({len(comps)} região[ões])")

    log("Descritores:")
    g = descritores.calcular(im)
    x = np.zeros(descritores.TAMANHO, dtype=np.float32)
    descritores.extrair(g, 0, 0, x, 0)
    checar(len(x) == descritores.TAMANHO and np.isfinite(x).all(),
           f"descritor com {descritores.TAMANHO} números finitos")

    log("Rede neural:")
    rede = RedeNeural([descritores.TAMANHO, 8, 1], np.random.default_rng(1))
    p = rede.prever(x)
    checar(0.0 <= p <= 1.0, f"saída sigmoide = {p:.3f}")
    # um passo de treino não pode gerar NaN
    X = rng.random((16, descritores.TAMANHO)).astype(np.float32)
    y = (rng.random(16) > 0.5).astype(np.float64)
    perda = rede.passo_lote(X, y, np.ones(16, np.float32), np.random.default_rng(2))
    checar(np.isfinite(perda), f"passo de treino (perda = {perda:.3f})")

    log("Modelo + detecção:")
    m = Modelo()
    m.media = np.zeros(descritores.TAMANHO, np.float32)
    m.desvio = np.ones(descritores.TAMANHO, np.float32)
    m.rede = rede
    r = detector.detectar(m, im)
    checar(r.prob.shape == (r.gh, r.gw), "mapa de probabilidades da detecção")

    log("Dados do projeto:")
    imgs = projeto.listar_imagens()
    checar(True, f"{len(imgs)} imagem(ns) em {projeto.pasta_imagens}")
    anot = projeto.imagens_anotadas()
    log(f"  {len(anot)} imagem(ns) anotada(s)")
    if projeto.arquivo_modelo().is_file():
        try:
            mm = Modelo.carregar(projeto.arquivo_modelo())
            log(f"  modelo treinado encontrado (F1 = {mm.f1 * 100:.1f}%)"
                if mm.f1 >= 0 else "  modelo treinado encontrado (sem validação)")
        except Exception as ex:  # noqa: BLE001
            checar(False, f"carregar modelo: {ex}")
    else:
        log("  nenhum modelo treinado ainda (rode 'treinar')")

    log("TUDO OK" if ok else "HOUVE FALHAS")
    return ok
