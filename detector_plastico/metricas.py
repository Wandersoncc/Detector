"""Matriz de confusão e as medidas de eficiência derivadas dela."""

from __future__ import annotations


class Metricas:
    __slots__ = ("vp", "fp", "fn", "vn")

    def __init__(self):
        self.vp = self.fp = self.fn = self.vn = 0

    def somar(self, previsto: bool, real: bool):
        if previsto and real:
            self.vp += 1
        elif previsto:
            self.fp += 1
        elif real:
            self.fn += 1
        else:
            self.vn += 1

    def somar_lote(self, previsto, real):
        """Soma vetores booleanos de uma vez (NumPy)."""
        import numpy as np

        previsto = np.asarray(previsto, dtype=bool)
        real = np.asarray(real, dtype=bool)
        self.vp += int((previsto & real).sum())
        self.fp += int((previsto & ~real).sum())
        self.fn += int((~previsto & real).sum())
        self.vn += int((~previsto & ~real).sum())

    def somar_metricas(self, o: "Metricas"):
        self.vp += o.vp
        self.fp += o.fp
        self.fn += o.fn
        self.vn += o.vn

    def precisao(self) -> float:
        if self.vp + self.fp == 0:
            return 1.0 if self.fn == 0 else 0.0
        return self.vp / (self.vp + self.fp)

    def revocacao(self) -> float:
        if self.vp + self.fn == 0:
            return 1.0 if self.fp == 0 else 0.0
        return self.vp / (self.vp + self.fn)

    def f1(self) -> float:
        d = 2 * self.vp + self.fp + self.fn
        return 1.0 if d == 0 else 2.0 * self.vp / d

    def iou(self) -> float:
        d = self.vp + self.fp + self.fn
        return 1.0 if d == 0 else self.vp / d

    def acuracia(self) -> float:
        t = self.total()
        return 0.0 if t == 0 else (self.vp + self.vn) / t

    def total(self) -> int:
        return self.vp + self.fp + self.fn + self.vn
