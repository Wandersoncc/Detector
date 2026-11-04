package br.aps.plastico.ia;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Random;

/**
 * Rede neural (perceptron multicamadas) escrita do zero: camadas ocultas ReLU,
 * saída sigmoide (probabilidade de ser plástico), treino com retropropagação,
 * otimizador Adam, dropout e decaimento de pesos.
 */
public final class RedeNeural {

    final int[] tam;
    final float[][] w, b;

    public float taxa = 0.0015f;
    public float decaimento = 0.01f;
    public float dropout = 0.15f;

    private float[][] gw, gb, mw, vw, mb, vb, ativ, delta;
    private int passos;

    public RedeNeural(int[] tamanhos, Random rnd) {
        this(tamanhos);
        for (int l = 0; l < w.length; l++) {
            int nIn = tam[l];
            double desvio = l == w.length - 1 ? Math.sqrt(1.0 / nIn) : Math.sqrt(2.0 / nIn);
            for (int i = 0; i < w[l].length; i++) {
                w[l][i] = (float) (rnd.nextGaussian() * desvio);
            }
        }
    }

    private RedeNeural(int[] tamanhos) {
        tam = tamanhos.clone();
        w = new float[tam.length - 1][];
        b = new float[tam.length - 1][];
        for (int l = 0; l < w.length; l++) {
            w[l] = new float[tam[l] * tam[l + 1]];
            b[l] = new float[tam[l + 1]];
        }
    }

    public int entradas() {
        return tam[0];
    }

    /** Probabilidade (0..1) para um vetor já normalizado. Seguro para várias threads. */
    public float prever(float[] x) {
        float[] a = x;
        int L = w.length;
        for (int l = 0; l < L; l++) {
            int nIn = tam[l], nOut = tam[l + 1];
            float[] out = new float[nOut];
            float[] wl = w[l];
            for (int o = 0; o < nOut; o++) {
                float s = b[l][o];
                int base = o * nIn;
                for (int i = 0; i < nIn; i++) {
                    s += wl[base + i] * a[i];
                }
                out[o] = l < L - 1 ? (s > 0 ? s : 0) : sigmoide(s);
            }
            a = out;
        }
        return a[0];
    }

    private static float sigmoide(float s) {
        return (float) (1.0 / (1.0 + Math.exp(-s)));
    }

    private void prepararTreino() {
        if (gw != null) {
            return;
        }
        int L = w.length;
        gw = new float[L][];
        gb = new float[L][];
        mw = new float[L][];
        vw = new float[L][];
        mb = new float[L][];
        vb = new float[L][];
        for (int l = 0; l < L; l++) {
            gw[l] = new float[w[l].length];
            mw[l] = new float[w[l].length];
            vw[l] = new float[w[l].length];
            gb[l] = new float[b[l].length];
            mb[l] = new float[b[l].length];
            vb[l] = new float[b[l].length];
        }
        ativ = new float[L + 1][];
        delta = new float[L + 1][];
        for (int l = 0; l <= L; l++) {
            ativ[l] = new float[tam[l]];
            delta[l] = new float[tam[l]];
        }
    }

    public void zerarGradientes() {
        prepararTreino();
        for (int l = 0; l < w.length; l++) {
            java.util.Arrays.fill(gw[l], 0f);
            java.util.Arrays.fill(gb[l], 0f);
        }
    }

    /**
     * Propaga uma amostra, acumula os gradientes e devolve a perda (entropia cruzada
     * ponderada). O peso permite dar mais importância à classe "plástico", que é rara.
     */
    public double acumular(float[] x, int alvo, float peso, Random rnd) {
        int L = w.length;
        System.arraycopy(x, 0, ativ[0], 0, tam[0]);
        float escalaDrop = 1f / (1f - dropout);
        for (int l = 0; l < L; l++) {
            int nIn = tam[l], nOut = tam[l + 1];
            float[] in = ativ[l], out = ativ[l + 1], wl = w[l];
            for (int o = 0; o < nOut; o++) {
                float s = b[l][o];
                int base = o * nIn;
                for (int i = 0; i < nIn; i++) {
                    s += wl[base + i] * in[i];
                }
                if (l < L - 1) {
                    if (s <= 0 || (dropout > 0 && rnd.nextFloat() < dropout)) {
                        s = 0;
                    } else if (dropout > 0) {
                        s *= escalaDrop;
                    }
                } else {
                    s = sigmoide(s);
                }
                out[o] = s;
            }
        }
        float p = Math.min(1f - 1e-6f, Math.max(1e-6f, ativ[L][0]));
        delta[L][0] = (p - alvo) * peso;
        for (int l = L - 1; l >= 0; l--) {
            int nIn = tam[l], nOut = tam[l + 1];
            float[] in = ativ[l], d = delta[l + 1], wl = w[l], gwl = gw[l];
            for (int o = 0; o < nOut; o++) {
                float dv = d[o];
                if (dv == 0f) {
                    continue;
                }
                gb[l][o] += dv;
                int base = o * nIn;
                for (int i = 0; i < nIn; i++) {
                    gwl[base + i] += dv * in[i];
                }
            }
            if (l > 0) {
                float[] dIn = delta[l];
                float fator = dropout > 0 ? escalaDrop : 1f;
                for (int i = 0; i < nIn; i++) {
                    if (in[i] <= 0) {
                        dIn[i] = 0;
                        continue;
                    }
                    float s = 0;
                    for (int o = 0; o < nOut; o++) {
                        s += wl[o * nIn + i] * d[o];
                    }
                    dIn[i] = s * fator;
                }
            }
        }
        return -peso * (alvo == 1 ? Math.log(p) : Math.log(1 - p));
    }

    /** Atualiza os pesos com Adam usando os gradientes acumulados de {@code n} amostras. */
    public void passoAdam(int n) {
        final float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        passos++;
        float corr = (float) (taxa * Math.sqrt(1 - Math.pow(b2, passos)) / (1 - Math.pow(b1, passos)));
        float inv = 1f / n;
        for (int l = 0; l < w.length; l++) {
            float[] wl = w[l], g = gw[l], m = mw[l], v = vw[l];
            for (int i = 0; i < wl.length; i++) {
                float gi = g[i] * inv;
                m[i] = b1 * m[i] + (1 - b1) * gi;
                v[i] = b2 * v[i] + (1 - b2) * gi * gi;
                wl[i] -= corr * m[i] / ((float) Math.sqrt(v[i]) + eps) + taxa * decaimento * wl[i];
            }
            float[] bl = b[l], gB = gb[l], mB = mb[l], vB = vb[l];
            for (int i = 0; i < bl.length; i++) {
                float gi = gB[i] * inv;
                mB[i] = b1 * mB[i] + (1 - b1) * gi;
                vB[i] = b2 * vB[i] + (1 - b2) * gi * gi;
                bl[i] -= corr * mB[i] / ((float) Math.sqrt(vB[i]) + eps);
            }
        }
    }

    /** Cópia só dos pesos (usada para guardar a melhor época). */
    public RedeNeural copia() {
        RedeNeural c = new RedeNeural(tam);
        for (int l = 0; l < w.length; l++) {
            System.arraycopy(w[l], 0, c.w[l], 0, w[l].length);
            System.arraycopy(b[l], 0, c.b[l], 0, b[l].length);
        }
        return c;
    }

    void salvar(DataOutputStream out) throws IOException {
        out.writeInt(tam.length);
        for (int t : tam) {
            out.writeInt(t);
        }
        for (int l = 0; l < w.length; l++) {
            for (float v : w[l]) {
                out.writeFloat(v);
            }
            for (float v : b[l]) {
                out.writeFloat(v);
            }
        }
    }

    static RedeNeural carregar(DataInputStream in) throws IOException {
        int n = in.readInt();
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = in.readInt();
        }
        RedeNeural r = new RedeNeural(t);
        for (int l = 0; l < r.w.length; l++) {
            for (int i = 0; i < r.w[l].length; i++) {
                r.w[l][i] = in.readFloat();
            }
            for (int i = 0; i < r.b[l].length; i++) {
                r.b[l][i] = in.readFloat();
            }
        }
        return r;
    }
}
