package br.aps.plastico.visao;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * Filtros clássicos de processamento de imagem (sem aprendizado): HSV, tons de
 * cinza, Sobel, bordas (o filtro "P&B contornos"), morfologia, componentes
 * conexos e a "varinha" que usa os contornos para acelerar a anotação.
 */
public final class Filtros {

    private Filtros() {
    }

    /** Imagem RGB em ponto flutuante, canais no intervalo [0,1]. */
    public static final class Rgb {
        public final int w, h;
        public final float[] r, g, b;

        public Rgb(int w, int h) {
            this.w = w;
            this.h = h;
            r = new float[w * h];
            g = new float[w * h];
            b = new float[w * h];
        }
    }

    // ------------------------------------------------------------------ conversões

    public static Rgb paraRgb(BufferedImage img) {
        int w = img.getWidth(), h = img.getHeight();
        Rgb o = new Rgb(w, h);
        int[] px = img.getRGB(0, 0, w, h, null, 0, w);
        for (int i = 0; i < px.length; i++) {
            int p = px[i];
            o.r[i] = ((p >> 16) & 255) / 255f;
            o.g[i] = ((p >> 8) & 255) / 255f;
            o.b[i] = (p & 255) / 255f;
        }
        return o;
    }

    public static BufferedImage paraImagem(Rgb im) {
        BufferedImage out = new BufferedImage(im.w, im.h, BufferedImage.TYPE_INT_RGB);
        int[] px = new int[im.w * im.h];
        for (int i = 0; i < px.length; i++) {
            px[i] = (nivel(im.r[i]) << 16) | (nivel(im.g[i]) << 8) | nivel(im.b[i]);
        }
        out.setRGB(0, 0, im.w, im.h, px, 0, im.w);
        return out;
    }

    /** Copia para TYPE_INT_RGB (fundo branco onde houver transparência). */
    public static BufferedImage copiaRgb(BufferedImage src) {
        BufferedImage d = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g = d.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, d.getWidth(), d.getHeight());
        g.drawImage(src, 0, 0, null);
        g.dispose();
        return d;
    }

    /** Escala que faz o maior lado da imagem medir {@code lado} pixels. */
    public static double escalaPara(int w, int h, int lado) {
        return lado / (double) Math.max(w, h);
    }

    public static BufferedImage redimensionar(BufferedImage src, int w, int h) {
        BufferedImage atual = src;
        int cw = src.getWidth(), ch = src.getHeight();
        // reduções grandes são feitas pela metade, em etapas, para não perder detalhes
        while (cw / 2 >= w && ch / 2 >= h) {
            cw /= 2;
            ch /= 2;
            atual = desenhar(atual, cw, ch);
        }
        if (cw != w || ch != h || atual == src) {
            atual = desenhar(atual, w, h);
        }
        return atual;
    }

    private static BufferedImage desenhar(BufferedImage s, int w, int h) {
        BufferedImage d = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = d.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g.drawImage(s, 0, 0, w, h, null);
        g.dispose();
        return d;
    }

    public static BufferedImage espelhar(BufferedImage src) {
        int w = src.getWidth(), h = src.getHeight();
        BufferedImage d = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        int[] linha = new int[w];
        for (int y = 0; y < h; y++) {
            src.getRGB(0, y, w, 1, linha, 0, w);
            for (int x = 0; x < w / 2; x++) {
                int t = linha[x];
                linha[x] = linha[w - 1 - x];
                linha[w - 1 - x] = t;
            }
            d.setRGB(0, y, w, 1, linha, 0, w);
        }
        return d;
    }

    /** Variação de brilho/contraste/saturação usada no aumento de dados do treino. */
    public static void ajustarCores(Rgb im, float brilho, float contraste, float saturacao) {
        for (int i = 0; i < im.r.length; i++) {
            float r = im.r[i], g = im.g[i], b = im.b[i];
            float y = 0.299f * r + 0.587f * g + 0.114f * b;
            r = y + (r - y) * saturacao;
            g = y + (g - y) * saturacao;
            b = y + (b - y) * saturacao;
            im.r[i] = limitar((r - 0.5f) * contraste + 0.5f + brilho);
            im.g[i] = limitar((g - 0.5f) * contraste + 0.5f + brilho);
            im.b[i] = limitar((b - 0.5f) * contraste + 0.5f + brilho);
        }
    }

    private static float limitar(float v) {
        return v < 0f ? 0f : (v > 1f ? 1f : v);
    }

    // ------------------------------------------------------------------ HSV e cinza

    /** Matiz (H) em graus [0,360). */
    public static float matiz(float r, float g, float b, float max, float min) {
        float d = max - min;
        if (d < 1e-6f) {
            return 0f;
        }
        float h;
        if (max == r) {
            h = 60f * (((g - b) / d) % 6f);
        } else if (max == g) {
            h = 60f * ((b - r) / d + 2f);
        } else {
            h = 60f * ((r - g) / d + 4f);
        }
        return h < 0 ? h + 360f : h;
    }

    public static float[] cinza(Rgb im) {
        float[] c = new float[im.w * im.h];
        for (int i = 0; i < c.length; i++) {
            c[i] = 0.299f * im.r[i] + 0.587f * im.g[i] + 0.114f * im.b[i];
        }
        return c;
    }

    /** Desfoque gaussiano 5x5 separável (núcleo 1 4 6 4 1). */
    public static float[] desfoque(float[] src, int w, int h) {
        float[] tmp = new float[w * h], dst = new float[w * h];
        for (int y = 0; y < h; y++) {
            int l = y * w;
            for (int x = 0; x < w; x++) {
                float s = 6 * src[l + x]
                        + 4 * (src[l + Math.max(0, x - 1)] + src[l + Math.min(w - 1, x + 1)])
                        + (src[l + Math.max(0, x - 2)] + src[l + Math.min(w - 1, x + 2)]);
                tmp[l + x] = s / 16f;
            }
        }
        for (int y = 0; y < h; y++) {
            int y1 = Math.max(0, y - 1) * w, y2 = Math.max(0, y - 2) * w;
            int y3 = Math.min(h - 1, y + 1) * w, y4 = Math.min(h - 1, y + 2) * w;
            for (int x = 0; x < w; x++) {
                float s = 6 * tmp[y * w + x] + 4 * (tmp[y1 + x] + tmp[y3 + x]) + (tmp[y2 + x] + tmp[y4 + x]);
                dst[y * w + x] = s / 16f;
            }
        }
        return dst;
    }

    /** Gradientes de Sobel {gx, gy}, normalizados para que um degrau de 0 a 1 valha 1. */
    public static float[][] sobel(float[] c, int w, int h) {
        float[] gx = new float[w * h], gy = new float[w * h];
        for (int y = 0; y < h; y++) {
            int ym = (y > 0 ? y - 1 : y) * w, y0 = y * w, yp = (y < h - 1 ? y + 1 : y) * w;
            for (int x = 0; x < w; x++) {
                int xm = x > 0 ? x - 1 : x, xp = x < w - 1 ? x + 1 : x;
                float a = c[ym + xm], b = c[ym + x], d = c[ym + xp];
                float e = c[y0 + xm], f = c[y0 + xp];
                float g = c[yp + xm], hh = c[yp + x], k = c[yp + xp];
                gx[y0 + x] = ((d + 2 * f + k) - (a + 2 * e + g)) / 4f;
                gy[y0 + x] = ((g + 2 * hh + k) - (a + 2 * b + d)) / 4f;
            }
        }
        return new float[][] { gx, gy };
    }

    // ------------------------------------------------------------------ filtros "antigos"

    /**
     * Detector de bordas no estilo Canny: é o filtro preto-e-branco que contorna
     * os objetos. Limiares automáticos pelo percentil do gradiente.
     */
    public static boolean[] bordas(Rgb im) {
        int w = im.w, h = im.h, n = w * h;
        float[] c = desfoque(cinza(im), w, h);
        float[][] s = sobel(c, w, h);
        float[] gx = s[0], gy = s[1];
        float[] mag = new float[n];
        for (int i = 0; i < n; i++) {
            mag[i] = (float) Math.sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
        }
        float alto = Math.max(0.06f, percentil(mag, 0.85f));
        float baixo = alto * 0.45f;

        // supressão de não-máximos: mantém só o pico do gradiente (linhas finas)
        byte[] estado = new byte[n];
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                int i = y * w + x;
                float m = mag[i];
                if (m < baixo) {
                    continue;
                }
                double ang = Math.toDegrees(Math.atan2(gy[i], gx[i]));
                if (ang < 0) {
                    ang += 180;
                }
                float n1, n2;
                if (ang < 22.5 || ang >= 157.5) {
                    n1 = mag[i - 1];
                    n2 = mag[i + 1];
                } else if (ang < 67.5) {
                    n1 = mag[i - w - 1];
                    n2 = mag[i + w + 1];
                } else if (ang < 112.5) {
                    n1 = mag[i - w];
                    n2 = mag[i + w];
                } else {
                    n1 = mag[i - w + 1];
                    n2 = mag[i + w - 1];
                }
                if (m >= n1 && m >= n2) {
                    estado[i] = (byte) (m >= alto ? 2 : 1);
                }
            }
        }
        // histerese: bordas fracas só ficam se encostarem numa forte
        boolean[] out = new boolean[n];
        int[] pilha = new int[n];
        int topo = 0;
        for (int i = 0; i < n; i++) {
            if (estado[i] == 2) {
                out[i] = true;
                pilha[topo++] = i;
            }
        }
        while (topo > 0) {
            int i = pilha[--topo];
            int x = i % w, y = i / w;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int xx = x + dx, yy = y + dy;
                    if (xx < 0 || yy < 0 || xx >= w || yy >= h) {
                        continue;
                    }
                    int j = yy * w + xx;
                    if (estado[j] == 1 && !out[j]) {
                        out[j] = true;
                        pilha[topo++] = j;
                    }
                }
            }
        }
        return out;
    }

    private static float percentil(float[] v, float q) {
        float max = 1e-6f;
        for (float x : v) {
            max = Math.max(max, x);
        }
        int[] hist = new int[1024];
        for (float x : v) {
            hist[Math.min(1023, (int) (x / max * 1023))]++;
        }
        long alvo = (long) (q * v.length), acc = 0;
        for (int i = 0; i < hist.length; i++) {
            acc += hist[i];
            if (acc >= alvo) {
                return i / 1023f * max;
            }
        }
        return max;
    }

    /** Nível 0..255 de um valor 0..1. */
    static int nivel(float v) {
        int q = Math.round(v * 255f);
        return q < 0 ? 0 : (q > 255 ? 255 : q);
    }

    /** Equalização de histograma: espalha os tons por toda a faixa 0..255. */
    public static float[] equalizar(float[] c) {
        int[] hist = new int[256];
        for (float v : c) {
            hist[nivel(v)]++;
        }
        long cdfMin = 0;
        for (int i = 0; i < 256 && cdfMin == 0; i++) {
            cdfMin = hist[i];
        }
        float[] mapa = new float[256];
        long acc = 0, n = c.length;
        for (int i = 0; i < 256; i++) {
            acc += hist[i];
            mapa[i] = n == cdfMin ? i / 255f : Math.max(0f, (acc - cdfMin) / (float) (n - cdfMin));
        }
        float[] out = new float[c.length];
        for (int i = 0; i < c.length; i++) {
            out[i] = mapa[nivel(c[i])];
        }
        return out;
    }

    /** Equaliza só o brilho e mantém as cores (realça objetos em água turva ou escura). */
    public static Rgb equalizarCores(Rgb im) {
        float[] y = cinza(im), ye = equalizar(y);
        Rgb o = new Rgb(im.w, im.h);
        for (int i = 0; i < y.length; i++) {
            if (y[i] < 1e-3f) {
                o.r[i] = o.g[i] = o.b[i] = ye[i];
                continue;
            }
            float f = ye[i] / y[i];
            o.r[i] = limitar(im.r[i] * f);
            o.g[i] = limitar(im.g[i] * f);
            o.b[i] = limitar(im.b[i] * f);
        }
        return o;
    }

    /** Filtro de mediana em janela (2r+1)x(2r+1): remove ruído pontual sem borrar as bordas. */
    public static float[] mediana(float[] c, int w, int h, int raio) {
        int lado = 2 * raio + 1;
        float[] jan = new float[lado * lado], out = new float[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int k = 0;
                for (int dy = -raio; dy <= raio; dy++) {
                    int yy = Math.max(0, Math.min(h - 1, y + dy)) * w;
                    for (int dx = -raio; dx <= raio; dx++) {
                        jan[k++] = c[yy + Math.max(0, Math.min(w - 1, x + dx))];
                    }
                }
                java.util.Arrays.sort(jan);
                out[y * w + x] = jan[k / 2];
            }
        }
        return out;
    }

    /**
     * Limiar de Otsu: o tom (0..1) que melhor separa os pixels em "escuros" e
     * "claros", maximizando a variância entre as duas classes.
     */
    public static float limiarOtsu(float[] c) {
        int[] hist = new int[256];
        for (float v : c) {
            hist[nivel(v)]++;
        }
        double soma = 0;
        for (int i = 0; i < 256; i++) {
            soma += i * (double) hist[i];
        }
        double somaEsc = 0, melhor = -1;
        long nEsc = 0, total = c.length;
        int t = 0;
        for (int i = 0; i < 256; i++) {
            nEsc += hist[i];
            if (nEsc == 0) {
                continue;
            }
            long nCla = total - nEsc;
            if (nCla == 0) {
                break;
            }
            somaEsc += i * (double) hist[i];
            double mEsc = somaEsc / nEsc, mCla = (soma - somaEsc) / nCla;
            double entre = (double) nEsc * nCla * (mEsc - mCla) * (mEsc - mCla);
            if (entre > melhor) {
                melhor = entre;
                t = i;
            }
        }
        return t / 255f;
    }

    /**
     * Filtro clássico por faixas fixas de HSV, sem aprendizado: tons claros e
     * dessaturados (garrafas transparentes, isopor) ou cores vivas quentes (rótulos,
     * tampas), já limpo por abertura morfológica. Serve de comparação com a IA.
     */
    public static boolean[] filtroHsvClassico(Rgb im) {
        return abrir(faixasHsv(im), im.w, im.h);
    }

    /** Só as faixas de HSV, antes da limpeza morfológica. */
    public static boolean[] faixasHsv(Rgb im) {
        int n = im.w * im.h;
        boolean[] m = new boolean[n];
        for (int i = 0; i < n; i++) {
            float r = im.r[i], g = im.g[i], b = im.b[i];
            float max = Math.max(r, Math.max(g, b)), min = Math.min(r, Math.min(g, b));
            float s = max > 1e-6f ? (max - min) / max : 0f;
            boolean claro = s < 0.22f && max > 0.72f;
            boolean corViva = false;
            if (!claro && s > 0.5f && max > 0.45f) {
                float h = matiz(r, g, b, max, min);
                corViva = h < 50f || h > 290f;
            }
            m[i] = claro || corViva;
        }
        return m;
    }

    // ------------------------------------------------------------------ morfologia

    /** Abertura (erosão e depois dilatação): apaga pontos soltos menores que o elemento. */
    public static boolean[] abrir(boolean[] m, int w, int h) {
        return dilatar(erodir(m, w, h), w, h);
    }

    /** Fechamento (dilatação e depois erosão): fecha pequenos buracos e une partes vizinhas. */
    public static boolean[] fechar(boolean[] m, int w, int h) {
        return erodir(dilatar(m, w, h), w, h);
    }

    public static boolean[] dilatar(boolean[] m, int w, int h) {
        boolean[] o = new boolean[m.length];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                boolean v = false;
                for (int dy = -1; dy <= 1 && !v; dy++) {
                    int yy = y + dy;
                    if (yy < 0 || yy >= h) {
                        continue;
                    }
                    for (int dx = -1; dx <= 1; dx++) {
                        int xx = x + dx;
                        if (xx >= 0 && xx < w && m[yy * w + xx]) {
                            v = true;
                            break;
                        }
                    }
                }
                o[y * w + x] = v;
            }
        }
        return o;
    }

    public static boolean[] erodir(boolean[] m, int w, int h) {
        boolean[] inv = new boolean[m.length];
        for (int i = 0; i < m.length; i++) {
            inv[i] = !m[i];
        }
        boolean[] d = dilatar(inv, w, h);
        for (int i = 0; i < m.length; i++) {
            d[i] = !d[i];
        }
        return d;
    }

    // ------------------------------------------------------------------ regiões

    /**
     * Componentes conexos (vizinhança-8). Cada item é {minX, minY, maxX, maxY, área}.
     * Se {@code rotulos} não for nulo, recebe o número do componente (1, 2, ...) por pixel.
     */
    public static List<int[]> componentes(boolean[] m, int w, int h, int[] rotulos) {
        List<int[]> lista = new ArrayList<int[]>();
        boolean[] visto = new boolean[m.length];
        int[] fila = new int[m.length];
        for (int i = 0; i < m.length; i++) {
            if (!m[i] || visto[i]) {
                continue;
            }
            int ini = 0, fim = 0;
            fila[fim++] = i;
            visto[i] = true;
            int[] c = { Integer.MAX_VALUE, Integer.MAX_VALUE, -1, -1, 0 };
            int id = lista.size() + 1;
            while (ini < fim) {
                int p = fila[ini++];
                int x = p % w, y = p / w;
                c[0] = Math.min(c[0], x);
                c[1] = Math.min(c[1], y);
                c[2] = Math.max(c[2], x);
                c[3] = Math.max(c[3], y);
                c[4]++;
                if (rotulos != null) {
                    rotulos[p] = id;
                }
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int xx = x + dx, yy = y + dy;
                        if (xx < 0 || yy < 0 || xx >= w || yy >= h) {
                            continue;
                        }
                        int j = yy * w + xx;
                        if (m[j] && !visto[j]) {
                            visto[j] = true;
                            fila[fim++] = j;
                        }
                    }
                }
            }
            lista.add(c);
        }
        return lista;
    }

    /**
     * Varinha mágica: a partir do clique, preenche a região de cor parecida que
     * fica dentro dos contornos detectados pelo filtro P&B. Retorna {@code null} se
     * a região "vazar" (contorno aberto) e passar de {@code maxPixels}.
     */
    public static boolean[] varinha(Rgb im, boolean[] bordas, int sx, int sy, float tolerancia, int maxPixels) {
        int w = im.w, h = im.h;
        boolean[] barreira = dilatar(bordas, w, h);
        int semente = -1;
        for (int raio = 0; raio <= 4 && semente < 0; raio++) {
            for (int dy = -raio; dy <= raio && semente < 0; dy++) {
                for (int dx = -raio; dx <= raio; dx++) {
                    int x = sx + dx, y = sy + dy;
                    if (x >= 0 && y >= 0 && x < w && y < h && !barreira[y * w + x]) {
                        semente = y * w + x;
                        break;
                    }
                }
            }
        }
        boolean[] reg = new boolean[w * h];
        if (semente < 0) {
            return reg;
        }
        // cor de referência: média 5x5 em volta da semente
        float mr = 0, mg = 0, mb = 0;
        int cont = 0, x0 = semente % w, y0 = semente / w;
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                int x = x0 + dx, y = y0 + dy;
                if (x >= 0 && y >= 0 && x < w && y < h) {
                    int j = y * w + x;
                    mr += im.r[j];
                    mg += im.g[j];
                    mb += im.b[j];
                    cont++;
                }
            }
        }
        mr /= cont;
        mg /= cont;
        mb /= cont;
        float tol2 = tolerancia * tolerancia;
        int[] fila = new int[w * h];
        int ini = 0, fim = 0;
        fila[fim++] = semente;
        reg[semente] = true;
        int[] dxs = { 1, -1, 0, 0 }, dys = { 0, 0, 1, -1 };
        while (ini < fim) {
            int p = fila[ini++];
            if (fim > maxPixels) {
                return null;
            }
            int x = p % w, y = p / w;
            for (int k = 0; k < 4; k++) {
                int xx = x + dxs[k], yy = y + dys[k];
                if (xx < 0 || yy < 0 || xx >= w || yy >= h) {
                    continue;
                }
                int j = yy * w + xx;
                if (reg[j] || barreira[j]) {
                    continue;
                }
                float dr = im.r[j] - mr, dg = im.g[j] - mg, db = im.b[j] - mb;
                if (dr * dr + dg * dg + db * db <= tol2) {
                    reg[j] = true;
                    fila[fim++] = j;
                }
            }
        }
        // inclui o próprio contorno e fecha buracos pequenos (reflexos, rótulos)
        for (int it = 0; it < 2; it++) {
            boolean[] d = dilatar(reg, w, h);
            for (int i = 0; i < reg.length; i++) {
                if (d[i] && barreira[i]) {
                    reg[i] = true;
                }
            }
        }
        reg = erodir(dilatar(dilatar(reg, w, h), w, h), w, h);
        return erodir(reg, w, h);
    }
}
