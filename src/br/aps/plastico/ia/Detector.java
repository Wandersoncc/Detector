package br.aps.plastico.ia;

import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import br.aps.plastico.visao.Descritores;
import br.aps.plastico.visao.Filtros;

/** Aplica o modelo treinado em uma imagem e localiza as regiões com plástico. */
public final class Detector {

    /** Menor região reportada: 4 sub-células (equivale a um objeto de ~16x16 px). */
    static final int MIN_SUBCELULAS = 4;

    private Detector() {
    }

    public static final class Regiao {
        /** Retângulo em coordenadas da imagem original. */
        public final Rectangle caixa;
        public final float confianca;
        public final int subcelulas;

        Regiao(Rectangle caixa, float confianca, int subcelulas) {
            this.caixa = caixa;
            this.confianca = confianca;
            this.subcelulas = subcelulas;
        }
    }

    public static final class Resultado {
        public final int largura, altura;
        /** Fator entre a imagem de trabalho e a original. */
        public final double escala;
        public final int gw, gh;
        /** Probabilidade de plástico de cada sub-célula de 8x8 px (na imagem de trabalho). */
        public final float[] prob;
        /** Fração de células em que a IA ficou em dúvida (0,3 &lt; p &lt; 0,7). */
        public final float incerteza;

        Resultado(int largura, int altura, double escala, int gw, int gh, float[] prob, float incerteza) {
            this.largura = largura;
            this.altura = altura;
            this.escala = escala;
            this.gw = gw;
            this.gh = gh;
            this.prob = prob;
            this.incerteza = incerteza;
        }

        public List<Regiao> regioes(float limiar) {
            boolean[] m = new boolean[prob.length];
            for (int i = 0; i < m.length; i++) {
                m[i] = prob[i] >= limiar;
            }
            int[] rot = new int[prob.length];
            List<int[]> comps = Filtros.componentes(m, gw, gh, rot);
            double[] soma = new double[comps.size() + 1];
            for (int i = 0; i < rot.length; i++) {
                if (rot[i] > 0) {
                    soma[rot[i]] += prob[i];
                }
            }
            double px = Descritores.P / escala;
            List<Regiao> out = new ArrayList<Regiao>();
            for (int k = 0; k < comps.size(); k++) {
                int[] c = comps.get(k);
                if (c[4] < MIN_SUBCELULAS) {
                    continue; // manchas de poucos pixels costumam ser ruído
                }
                int x0 = (int) Math.floor(c[0] * px), y0 = (int) Math.floor(c[1] * px);
                int x1 = Math.min(largura, (int) Math.ceil((c[2] + 1) * px));
                int y1 = Math.min(altura, (int) Math.ceil((c[3] + 1) * px));
                out.add(new Regiao(new Rectangle(x0, y0, x1 - x0, y1 - y0), (float) (soma[k + 1] / c[4]), c[4]));
            }
            Collections.sort(out, new Comparator<Regiao>() {
                @Override
                public int compare(Regiao a, Regiao b) {
                    return Integer.compare(b.subcelulas, a.subcelulas);
                }
            });
            return out;
        }

        /** Percentual da imagem coberto por plástico, segundo a IA. */
        public double areaPercentual(float limiar) {
            int P = Descritores.P;
            int w = (int) Math.round(largura * escala), h = (int) Math.round(altura * escala);
            double area = 0;
            for (int y = 0; y < gh; y++) {
                for (int x = 0; x < gw; x++) {
                    if (prob[y * gw + x] >= limiar) {
                        area += Math.min(P, w - x * P) * (double) Math.min(P, h - y * P);
                    }
                }
            }
            return 100.0 * area / (w * (double) h);
        }

        /** Probabilidade interpolada no ponto (x, y) da imagem original. */
        public float probEm(double x, double y) {
            double gx = x * escala / Descritores.P - 0.5, gy = y * escala / Descritores.P - 0.5;
            int x0 = (int) Math.floor(gx), y0 = (int) Math.floor(gy);
            double fx = gx - x0, fy = gy - y0;
            float a = em(x0, y0), b = em(x0 + 1, y0), c = em(x0, y0 + 1), d = em(x0 + 1, y0 + 1);
            return (float) ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy);
        }

        private float em(int x, int y) {
            x = Math.max(0, Math.min(gw - 1, x));
            y = Math.max(0, Math.min(gh - 1, y));
            return prob[y * gw + x];
        }

        /**
         * Mapa de calor (1 pixel por sub-célula), para desenhar ampliado sobre a
         * imagem. Probabilidades abaixo de {@code minimo} ficam transparentes.
         */
        public BufferedImage imagemCalor(float minimo) {
            BufferedImage img = new BufferedImage(gw, gh, BufferedImage.TYPE_INT_ARGB);
            float faixa = Math.max(0.05f, 1f - minimo);
            for (int y = 0; y < gh; y++) {
                for (int x = 0; x < gw; x++) {
                    float p = prob[y * gw + x];
                    int argb = 0;
                    if (p >= minimo) {
                        float t = Math.min(1f, (p - minimo) / faixa);
                        int alfa = (int) (90 + 110 * t);
                        int verde = (int) (220 * (1 - t));
                        argb = (alfa << 24) | (255 << 16) | (verde << 8);
                    }
                    img.setRGB(x, y, argb);
                }
            }
            return img;
        }

        /** Largura e altura (na imagem original) cobertas pela grade do mapa de calor. */
        public double larguraGrade() {
            return gw * Descritores.P / escala;
        }

        public double alturaGrade() {
            return gh * Descritores.P / escala;
        }
    }

    public static Resultado detectar(Modelo modelo, BufferedImage img) {
        int w0 = img.getWidth(), h0 = img.getHeight();
        double esc = Filtros.escalaPara(w0, h0, modelo.ladoBase);
        int w = Math.max(32, (int) Math.round(w0 * esc)), h = Math.max(32, (int) Math.round(h0 * esc));
        Filtros.Rgb im = Filtros.paraRgb(Filtros.redimensionar(img, w, h));
        Descritores.Grade g = Descritores.calcular(im);
        int gw = g.gw, gh = g.gh, cel = Descritores.CEL;
        float[] soma = new float[gw * gh], cont = new float[gw * gh];
        float[] x = new float[Descritores.TAMANHO];
        int duvidosas = 0, total = 0;
        for (int cy = 0; cy < g.ny(); cy++) {
            for (int cx = 0; cx < g.nx(); cx++) {
                Descritores.extrair(g, cx, cy, x, 0);
                float p = modelo.prever(x, 0);
                if (p > 0.3f && p < 0.7f) {
                    duvidosas++;
                }
                total++;
                for (int dy = 0; dy < cel; dy++) {
                    for (int dx = 0; dx < cel; dx++) {
                        int s = (cy + dy) * gw + cx + dx;
                        soma[s] += p;
                        cont[s]++;
                    }
                }
            }
        }
        float[] prob = new float[gw * gh];
        for (int i = 0; i < prob.length; i++) {
            prob[i] = cont[i] > 0 ? soma[i] / cont[i] : 0f;
        }
        return new Resultado(w0, h0, w / (double) w0, gw, gh, prob, total == 0 ? 0 : duvidosas / (float) total);
    }
}
