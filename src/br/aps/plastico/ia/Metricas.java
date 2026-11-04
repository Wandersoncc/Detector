package br.aps.plastico.ia;

/** Matriz de confusão e as medidas de eficiência derivadas dela. */
public final class Metricas {

    public long vp, fp, fn, vn;

    public void somar(boolean previsto, boolean real) {
        if (previsto && real) {
            vp++;
        } else if (previsto) {
            fp++;
        } else if (real) {
            fn++;
        } else {
            vn++;
        }
    }

    public void somar(Metricas o) {
        vp += o.vp;
        fp += o.fp;
        fn += o.fn;
        vn += o.vn;
    }

    /** Das células que a IA marcou como plástico, quantas eram mesmo. */
    public double precisao() {
        return vp + fp == 0 ? (fn == 0 ? 1 : 0) : vp / (double) (vp + fp);
    }

    /** Do plástico que existia, quanto a IA encontrou. */
    public double revocacao() {
        return vp + fn == 0 ? (fp == 0 ? 1 : 0) : vp / (double) (vp + fn);
    }

    /** F1: média harmônica de precisão e revocação. É a "eficiência" mostrada no sistema. */
    public double f1() {
        long d = 2 * vp + fp + fn;
        return d == 0 ? 1 : 2.0 * vp / d;
    }

    public double iou() {
        long d = vp + fp + fn;
        return d == 0 ? 1 : vp / (double) d;
    }

    public double acuracia() {
        long t = vp + fp + fn + vn;
        return t == 0 ? 0 : (vp + vn) / (double) t;
    }

    public long total() {
        return vp + fp + fn + vn;
    }
}
