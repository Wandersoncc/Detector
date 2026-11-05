"""Rede neural (perceptron multicamadas) escrita do zero com NumPy.

Camadas ocultas ReLU, saída sigmoide (probabilidade de ser plástico), treino com
retropropagação, otimizador Adam, dropout e decaimento de pesos (weight decay).

Diferente do Java (que processa uma amostra por vez), aqui a propagação e a
retropropagação são vetorizadas por lote (mini-batch), o que torna o treino
muito mais rápido mantendo a mesma matemática.
"""

from __future__ import annotations

import numpy as np


def _sigmoide(s):
    return 1.0 / (1.0 + np.exp(-s))


class RedeNeural:
    def __init__(self, tamanhos, rng: np.random.Generator | None = None):
        self.tam = list(tamanhos)
        self.taxa = 0.0015
        self.decaimento = 0.01
        self.dropout = 0.15
        self.w = []
        self.b = []
        L = len(self.tam) - 1
        for l in range(L):
            n_in, n_out = self.tam[l], self.tam[l + 1]
            if rng is not None:
                desvio = np.sqrt(1.0 / n_in) if l == L - 1 else np.sqrt(2.0 / n_in)
                self.w.append((rng.standard_normal((n_out, n_in)) * desvio).astype(np.float32))
            else:
                self.w.append(np.zeros((n_out, n_in), dtype=np.float32))
            self.b.append(np.zeros(n_out, dtype=np.float32))
        # estado do Adam
        self._mw = [np.zeros_like(x) for x in self.w]
        self._vw = [np.zeros_like(x) for x in self.w]
        self._mb = [np.zeros_like(x) for x in self.b]
        self._vb = [np.zeros_like(x) for x in self.b]
        self._passos = 0

    def entradas(self):
        return self.tam[0]

    # ---------------------------------------------------------------- inferência

    def prever(self, x: np.ndarray) -> np.ndarray:
        """Probabilidade(s) 0..1 para um vetor (TAMANHO,) ou um lote (N, TAMANHO)."""
        unidim = x.ndim == 1
        a = x.reshape(1, -1) if unidim else x
        L = len(self.w)
        for l in range(L):
            z = a @ self.w[l].T + self.b[l]
            a = np.maximum(0.0, z) if l < L - 1 else _sigmoide(z)
        saida = a[:, 0]
        return float(saida[0]) if unidim else saida

    # -------------------------------------------------------------------- treino

    def passo_lote(self, X: np.ndarray, y: np.ndarray, peso: np.ndarray,
                   rng: np.random.Generator) -> float:
        """Um passo de treino sobre um lote: propaga, retropropaga e aplica Adam.

        X: (N, entradas), y: (N,) em {0,1}, peso: (N,) peso de cada amostra.
        Devolve a perda média (entropia cruzada ponderada) do lote.
        """
        L = len(self.w)
        n = X.shape[0]
        escala_drop = 1.0 / (1.0 - self.dropout) if self.dropout > 0 else 1.0

        # forward guardando ativações e máscaras de dropout
        ativ = [X]
        mascaras = []
        a = X
        for l in range(L):
            z = a @ self.w[l].T + self.b[l]
            if l < L - 1:
                a = np.maximum(0.0, z)
                if self.dropout > 0:
                    manter = rng.random(a.shape) >= self.dropout
                    a = a * manter * escala_drop
                    mascaras.append(manter)
                else:
                    mascaras.append(None)
            else:
                a = _sigmoide(z)
                mascaras.append(None)
            ativ.append(a)

        p = np.clip(ativ[-1][:, 0], 1e-6, 1 - 1e-6)
        perda = float(np.sum(peso * np.where(y == 1, -np.log(p), -np.log(1 - p))) / max(1, n))

        # backward
        delta = ((p - y) * peso).reshape(-1, 1)  # (N, 1) na saída
        gw = [None] * L
        gb = [None] * L
        for l in range(L - 1, -1, -1):
            a_prev = ativ[l]
            gw[l] = (delta.T @ a_prev) / n
            gb[l] = delta.sum(axis=0) / n
            if l > 0:
                dprev = delta @ self.w[l]
                relu = ativ[l] > 0
                dprev = dprev * relu
                if self.dropout > 0 and mascaras[l - 1] is not None:
                    dprev = dprev * mascaras[l - 1] * escala_drop
                delta = dprev

        self._adam(gw, gb)
        return perda

    def _adam(self, gw, gb):
        b1, b2, eps = 0.9, 0.999, 1e-8
        self._passos += 1
        t = self._passos
        corr = self.taxa * np.sqrt(1 - b2 ** t) / (1 - b1 ** t)
        for l in range(len(self.w)):
            self._mw[l] = b1 * self._mw[l] + (1 - b1) * gw[l]
            self._vw[l] = b2 * self._vw[l] + (1 - b2) * (gw[l] * gw[l])
            self.w[l] -= (corr * self._mw[l] / (np.sqrt(self._vw[l]) + eps)
                          + self.taxa * self.decaimento * self.w[l]).astype(np.float32)
            self._mb[l] = b1 * self._mb[l] + (1 - b1) * gb[l]
            self._vb[l] = b2 * self._vb[l] + (1 - b2) * (gb[l] * gb[l])
            self.b[l] -= (corr * self._mb[l] / (np.sqrt(self._vb[l]) + eps)).astype(np.float32)

    # ---------------------------------------------------------------------- util

    def copia(self) -> "RedeNeural":
        c = RedeNeural(self.tam)
        c.w = [x.copy() for x in self.w]
        c.b = [x.copy() for x in self.b]
        c.taxa = self.taxa
        c.decaimento = self.decaimento
        c.dropout = self.dropout
        return c

    def estado(self) -> dict:
        return {
            "tam": np.array(self.tam, dtype=np.int32),
            "w": self.w,
            "b": self.b,
        }

    @staticmethod
    def de_estado(d: dict) -> "RedeNeural":
        r = RedeNeural([int(x) for x in d["tam"]])
        r.w = [np.asarray(x, dtype=np.float32) for x in d["w"]]
        r.b = [np.asarray(x, dtype=np.float32) for x in d["b"]]
        return r
