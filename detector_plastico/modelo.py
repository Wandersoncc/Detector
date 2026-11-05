"""O "cérebro" treinado: normalização dos descritores + rede neural + limiar.

Salvo em formato NumPy ``.npz`` (portável e simples). Guarda a média/desvio de
normalização, os pesos da rede e as métricas da validação cruzada.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import descritores
from .rede_neural import RedeNeural

LADO_BASE = 640
_LIMITE_Z = 8.0
VERSAO = 2


def normalizar(X: np.ndarray, media: np.ndarray, desvio: np.ndarray) -> np.ndarray:
    """Normaliza (z-score) e satura em +/- LIMITE_Z. Aceita vetor ou lote."""
    z = (X - media) / desvio
    return np.clip(z, -_LIMITE_Z, _LIMITE_Z)


class Modelo:
    def __init__(self):
        self.lado_base = LADO_BASE
        self.media = None
        self.desvio = None
        self.rede: RedeNeural | None = None
        self.limiar = 0.5
        self.data_treino = 0.0
        self.n_imagens = 0
        self.f1 = -1.0
        self.precisao = -1.0
        self.revocacao = -1.0
        self.iou = -1.0
        self.f1_classico = -1.0
        self.f1_proprias = -1.0
        self.f1_importadas = -1.0

    def prever(self, X: np.ndarray) -> np.ndarray:
        """Probabilidade de plástico para um descritor (ou lote de descritores)."""
        z = normalizar(X, self.media, self.desvio)
        return self.rede.prever(z)

    def salvar(self, arquivo):
        arquivo = Path(arquivo)
        estado = self.rede.estado()
        dados = {
            "versao": np.int32(VERSAO),
            "lado_base": np.int32(self.lado_base),
            "tamanho_descritor": np.int32(descritores.TAMANHO),
            "media": self.media.astype(np.float32),
            "desvio": self.desvio.astype(np.float32),
            "limiar": np.float32(self.limiar),
            "data_treino": np.float64(self.data_treino),
            "n_imagens": np.int32(self.n_imagens),
            "metricas": np.array(
                [self.f1, self.precisao, self.revocacao, self.iou,
                 self.f1_classico, self.f1_proprias, self.f1_importadas],
                dtype=np.float32),
            "rede_tam": estado["tam"],
            "rede_nc": np.int32(len(estado["w"])),
        }
        for i, (w, b) in enumerate(zip(estado["w"], estado["b"])):
            dados[f"w{i}"] = w
            dados[f"b{i}"] = b
        tmp = arquivo.with_suffix(arquivo.suffix + ".tmp")
        np.savez(tmp, **dados)
        # np.savez acrescenta .npz se não houver; normaliza o nome
        gerado = tmp if tmp.exists() else Path(str(tmp) + ".npz")
        gerado.replace(arquivo)

    @staticmethod
    def carregar(arquivo) -> "Modelo":
        d = np.load(arquivo, allow_pickle=False)
        versao = int(d["versao"])
        if versao < 1 or versao > VERSAO:
            raise IOError(f"Versão de modelo não suportada: {versao}")
        if int(d["tamanho_descritor"]) != descritores.TAMANHO:
            raise IOError("Modelo feito com outra versão dos descritores; treine novamente.")
        m = Modelo()
        m.lado_base = int(d["lado_base"])
        m.media = d["media"].astype(np.float32)
        m.desvio = d["desvio"].astype(np.float32)
        m.limiar = float(d["limiar"])
        m.data_treino = float(d["data_treino"])
        m.n_imagens = int(d["n_imagens"])
        mt = d["metricas"]
        (m.f1, m.precisao, m.revocacao, m.iou,
         m.f1_classico, m.f1_proprias, m.f1_importadas) = [float(x) for x in mt]
        nc = int(d["rede_nc"])
        w = [d[f"w{i}"].astype(np.float32) for i in range(nc)]
        b = [d[f"b{i}"].astype(np.float32) for i in range(nc)]
        m.rede = RedeNeural.de_estado({"tam": d["rede_tam"], "w": w, "b": b})
        return m
