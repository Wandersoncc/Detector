package br.aps.plastico.visao;

/**
 * Transforma cada pedaço da imagem em um vetor de números (descritor) que a rede
 * neural consegue aprender: cor, textura, bordas/forma e o contexto ao redor.
 *
 * A imagem é dividida em sub-células de 8x8 px. Uma "célula" (a unidade que a IA
 * classifica como plástico ou não) tem 2x2 sub-células (16x16 px). Para cada célula
 * olhamos três janelas centradas nela: 16 px (o objeto), 48 px (vizinhança) e
 * 128 px (cena), assim a IA aprende também o contexto e não só a cor do pixel.
 */
public final class Descritores {

    /** Lado da sub-célula, em pixels. */
    public static final int P = 8;
    /** Lado da célula classificada, em sub-células (2 x 8 = 16 px). */
    public static final int CEL = 2;
    /** Janelas de contexto, em sub-células: 16, 48 e 128 px. */
    static final int[] JANELAS = { 2, 6, 16 };

    static final int NC = 20; // faixas de cor (4 neutras + 8 matizes x 2 saturações)
    static final int NO = 8; // orientações do gradiente (forma/contornos)
    static final int NL = 10; // padrões de textura LBP

    // canais acumulados por sub-célula
    static final int C_COR = 0, C_ORI = NC, C_LBP = C_ORI + NO, C_SOMA = C_LBP + NL, C_QUAD = C_SOMA + 5,
            C_MAG = C_QUAD + 5, C_BORDA = C_MAG + 1, C_ESPEC = C_BORDA + 1, C_N = C_ESPEC + 1, NCANAIS = C_N + 1;

    static final int POR_JANELA = NC + 10 + NO + 2 + NL + 1;
    /** Quantidade de números em cada descritor. */
    public static final int TAMANHO = POR_JANELA * JANELAS.length + 10;

    private static final float LIMIAR_BORDA = 0.15f;
    private static final int[] TABELA_LBP = new int[256];

    static {
        for (int b = 0; b < 256; b++) {
            int rot = ((b >> 1) | ((b & 1) << 7)) & 0xFF;
            int transicoes = Integer.bitCount(b ^ rot);
            TABELA_LBP[b] = transicoes <= 2 ? Integer.bitCount(b) : 9;
        }
    }

    private Descritores() {
    }

    /** Somas acumuladas (imagem integral) por canal sobre a grade de sub-células. */
    public static final class Grade {
        public final int gw, gh, w, h;
        final double[] integral;

        Grade(int w, int h, int gw, int gh, double[] integral) {
            this.w = w;
            this.h = h;
            this.gw = gw;
            this.gh = gh;
            this.integral = integral;
        }

        /** Soma do canal no retângulo de sub-células [x0,x1) x [y0,y1). */
        double soma(int c, int x0, int y0, int x1, int y1) {
            int l = gw + 1;
            return integral[(y1 * l + x1) * NCANAIS + c] - integral[(y0 * l + x1) * NCANAIS + c]
                    - integral[(y1 * l + x0) * NCANAIS + c] + integral[(y0 * l + x0) * NCANAIS + c];
        }

        /** Número de posições de célula na horizontal (passo de 1 sub-célula). */
        public int nx() {
            return Math.max(0, gw - CEL + 1);
        }

        public int ny() {
            return Math.max(0, gh - CEL + 1);
        }
    }

    public static Grade calcular(Filtros.Rgb im) {
        final int w = im.w, h = im.h;
        final int gw = (w + P - 1) / P, gh = (h + P - 1) / P;
        double[] acc = new double[gw * gh * NCANAIS];
        float[] cinza = Filtros.cinza(im);
        float[][] sob = Filtros.sobel(cinza, w, h);
        float[] gx = sob[0], gy = sob[1];

        for (int y = 0; y < h; y++) {
            int linhaSub = (y / P) * gw;
            for (int x = 0; x < w; x++) {
                int i = y * w + x;
                float r = im.r[i], g = im.g[i], b = im.b[i];
                float max = Math.max(r, Math.max(g, b)), min = Math.min(r, Math.min(g, b));
                float v = max, s = max > 1e-6f ? (max - min) / max : 0f;
                int base = (linhaSub + x / P) * NCANAIS;

                acc[base + C_COR + faixaCor(r, g, b, max, min, s, v)] += 1;

                float mx = gx[i], my = gy[i];
                float mag = (float) Math.sqrt(mx * mx + my * my);
                if (mag > 1e-4f) {
                    double ang = Math.atan2(my, mx);
                    if (ang < 0) {
                        ang += Math.PI;
                    }
                    int bo = (int) (ang / Math.PI * NO);
                    acc[base + C_ORI + (bo >= NO ? NO - 1 : bo)] += mag;
                }
                acc[base + C_LBP + lbp(cinza, w, h, x, y)] += 1;

                acc[base + C_SOMA] += r;
                acc[base + C_SOMA + 1] += g;
                acc[base + C_SOMA + 2] += b;
                acc[base + C_SOMA + 3] += s;
                acc[base + C_SOMA + 4] += v;
                acc[base + C_QUAD] += r * r;
                acc[base + C_QUAD + 1] += g * g;
                acc[base + C_QUAD + 2] += b * b;
                acc[base + C_QUAD + 3] += s * s;
                acc[base + C_QUAD + 4] += v * v;
                acc[base + C_MAG] += mag;
                if (mag > LIMIAR_BORDA) {
                    acc[base + C_BORDA] += 1;
                }
                if (v > 0.88f && s < 0.15f) {
                    acc[base + C_ESPEC] += 1; // brilho especular, típico de plástico
                }
                acc[base + C_N] += 1;
            }
        }

        int l = gw + 1;
        double[] integ = new double[(gh + 1) * l * NCANAIS];
        double[] linha = new double[NCANAIS];
        for (int y = 1; y <= gh; y++) {
            java.util.Arrays.fill(linha, 0);
            for (int x = 1; x <= gw; x++) {
                int a = ((y - 1) * gw + (x - 1)) * NCANAIS;
                int dst = (y * l + x) * NCANAIS, cima = ((y - 1) * l + x) * NCANAIS;
                for (int c = 0; c < NCANAIS; c++) {
                    linha[c] += acc[a + c];
                    integ[dst + c] = integ[cima + c] + linha[c];
                }
            }
        }
        return new Grade(w, h, gw, gh, integ);
    }

    static int faixaCor(float r, float g, float b, float max, float min, float s, float v) {
        if (v < 0.18f) {
            return 0; // preto
        }
        if (s < 0.18f) {
            return v < 0.45f ? 1 : (v < 0.75f ? 2 : 3); // cinza escuro, cinza claro, branco
        }
        float h = Filtros.matiz(r, g, b, max, min);
        int hb = (int) (((h + 22.5f) % 360f) / 45f);
        return 4 + Math.min(7, hb) * 2 + (s < 0.5f ? 0 : 1);
    }

    private static int lbp(float[] c, int w, int h, int x, int y) {
        float centro = c[y * w + x] + 0.02f;
        int xm = x > 0 ? x - 1 : x, xp = x < w - 1 ? x + 1 : x;
        int ym = (y > 0 ? y - 1 : y) * w, y0 = y * w, yp = (y < h - 1 ? y + 1 : y) * w;
        int bits = (c[ym + xm] >= centro ? 1 : 0)
                | (c[ym + x] >= centro ? 2 : 0)
                | (c[ym + xp] >= centro ? 4 : 0)
                | (c[y0 + xp] >= centro ? 8 : 0)
                | (c[yp + xp] >= centro ? 16 : 0)
                | (c[yp + x] >= centro ? 32 : 0)
                | (c[yp + xm] >= centro ? 64 : 0)
                | (c[y0 + xm] >= centro ? 128 : 0);
        return TABELA_LBP[bits];
    }

    /**
     * Escreve em {@code out[off..off+TAMANHO)} o descritor da célula cujo canto
     * superior esquerdo é a sub-célula (cx, cy).
     */
    public static void extrair(Grade g, int cx, int cy, float[] out, int off) {
        int ccx = cx + CEL / 2, ccy = cy + CEL / 2;
        float[] med = new float[5 * JANELAS.length];
        int o = off;
        for (int k = 0; k < JANELAS.length; k++) {
            int meia = JANELAS[k] / 2;
            int x0 = Math.max(0, ccx - meia), x1 = Math.min(g.gw, ccx + meia);
            int y0 = Math.max(0, ccy - meia), y1 = Math.min(g.gh, ccy + meia);
            double n = Math.max(1, g.soma(C_N, x0, y0, x1, y1));
            for (int c = 0; c < NC; c++) {
                out[o++] = (float) (g.soma(C_COR + c, x0, y0, x1, y1) / n);
            }
            for (int c = 0; c < 5; c++) {
                float m = (float) (g.soma(C_SOMA + c, x0, y0, x1, y1) / n);
                med[k * 5 + c] = m;
                out[o++] = m;
            }
            for (int c = 0; c < 5; c++) {
                double m = med[k * 5 + c];
                double var = g.soma(C_QUAD + c, x0, y0, x1, y1) / n - m * m;
                out[o++] = (float) Math.sqrt(Math.max(0, var));
            }
            double mag = g.soma(C_MAG, x0, y0, x1, y1);
            for (int c = 0; c < NO; c++) {
                out[o++] = (float) (g.soma(C_ORI + c, x0, y0, x1, y1) / (mag + 1e-6));
            }
            out[o++] = (float) (mag / n);
            out[o++] = (float) (g.soma(C_BORDA, x0, y0, x1, y1) / n);
            for (int c = 0; c < NL; c++) {
                out[o++] = (float) (g.soma(C_LBP + c, x0, y0, x1, y1) / n);
            }
            out[o++] = (float) (g.soma(C_ESPEC, x0, y0, x1, y1) / n);
        }
        // contraste entre o objeto e o entorno (plástico costuma "destoar" do rio)
        for (int c = 0; c < 5; c++) {
            out[o++] = med[c] - med[5 + c];
        }
        for (int c = 0; c < 5; c++) {
            out[o++] = med[c] - med[10 + c];
        }
    }
}
