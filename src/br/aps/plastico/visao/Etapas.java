package br.aps.plastico.visao;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import javax.imageio.ImageIO;

/**
 * O processamento digital de imagens passo a passo (aquisição, realce, filtragem,
 * bordas, limiarização, segmentação e morfologia), com o resultado e uma explicação
 * de cada etapa, para analisar os filtros e ilustrar o trabalho escrito.
 */
public final class Etapas {

    private static final Locale PT = new Locale("pt", "BR");

    private Etapas() {
    }

    public static final class Etapa {
        public final String titulo, explicacao, info;
        public final BufferedImage imagem;

        public Etapa(String titulo, String explicacao, String info, BufferedImage imagem) {
            this.titulo = titulo;
            this.explicacao = explicacao;
            this.info = info;
            this.imagem = imagem;
        }
    }

    public static List<Etapa> calcular(Filtros.Rgb im) {
        int w = im.w, h = im.h, n = w * h;
        List<Etapa> e = new ArrayList<Etapa>();

        e.add(new Etapa("1. Aquisição (original)",
                "A foto como foi capturada, reduzida para no máximo 1024 px para a análise ser rápida.",
                w + " × " + h + " pixels", Filtros.paraImagem(im)));

        float[] cinza = Filtros.cinza(im);
        e.add(new Etapa("2. Tons de cinza",
                "Luminância Y = 0,299·R + 0,587·G + 0,114·B. Os filtros de forma e borda usam um único canal.",
                f("brilho médio: %.0f de 255", media(cinza) * 255), cinzaImg(cinza, w, h)));

        float[] eq = Filtros.equalizar(cinza);
        e.add(new Etapa("3. Equalização de histograma",
                "Redistribui os tons para ocupar toda a faixa 0–255. Realça garrafas e sacolas em água turva ou escura.",
                f("contraste (desvio dos tons): %.0f → %.0f", desvio(cinza) * 255, desvio(eq) * 255),
                Filtros.paraImagem(Filtros.equalizarCores(im))));

        float[] med = Filtros.mediana(cinza, w, h, 2);
        int mudou = 0;
        for (int i = 0; i < n; i++) {
            if (Math.abs(med[i] - cinza[i]) > 5 / 255f) {
                mudou++;
            }
        }
        e.add(new Etapa("4. Filtro de mediana 5×5",
                "Cada pixel vira a mediana dos 25 vizinhos. Remove reflexos e ruído pontual sem borrar as bordas.",
                f("%.1f%% dos pixels mudaram mais de 5 tons", 100.0 * mudou / n), cinzaImg(med, w, h)));

        float[] gau = Filtros.desfoque(cinza, w, h);
        e.add(new Etapa("5. Filtro gaussiano 5×5",
                "Média ponderada pelo núcleo 1-4-6-4-1 (separável). Suaviza a textura da água antes de medir bordas.",
                f("desvio dos tons: %.1f → %.1f", desvio(cinza) * 255, desvio(gau) * 255), cinzaImg(gau, w, h)));

        float[][] s = Filtros.sobel(gau, w, h);
        float[] mag = new float[n];
        double somaMag = 0;
        for (int i = 0; i < n; i++) {
            mag[i] = (float) Math.sqrt(s[0][i] * s[0][i] + s[1][i] * s[1][i]);
            somaMag += mag[i];
        }
        float p99 = percentil(mag, 0.99f);
        float[] magVis = new float[n];
        for (int i = 0; i < n; i++) {
            magVis[i] = Math.min(1f, mag[i] / Math.max(1e-4f, p99));
        }
        e.add(new Etapa("6. Sobel (gradiente)",
                "Derivadas horizontal e vertical com os núcleos de Sobel. Quanto mais claro, mais forte a mudança de brilho.",
                f("gradiente médio: %.3f", somaMag / n), cinzaImg(magVis, w, h)));

        boolean[] bordas = Filtros.bordas(im);
        e.add(new Etapa("7. Canny – contornos P&B",
                "Sobel + supressão de não-máximos + histerese: linhas finas que contornam os objetos (filtro P&B).",
                f("%.1f%% dos pixels são borda", 100.0 * contar(bordas) / n), binImg(bordas, w, h)));

        float t = Filtros.limiarOtsu(med);
        boolean[] otsu = new boolean[n];
        for (int i = 0; i < n; i++) {
            otsu[i] = med[i] >= t;
        }
        e.add(new Etapa("8. Limiarização de Otsu",
                "Escolhe sozinho o tom que melhor separa claros e escuros (máxima variância entre as classes).",
                f("limiar automático: %d de 255 · %.1f%% claros", Math.round(t * 255), 100.0 * contar(otsu) / n),
                binImg(otsu, w, h)));

        float[] sat = new float[n];
        for (int i = 0; i < n; i++) {
            float max = Math.max(im.r[i], Math.max(im.g[i], im.b[i])), min = Math.min(im.r[i], Math.min(im.g[i], im.b[i]));
            sat[i] = max > 1e-6f ? (max - min) / max : 0f;
        }
        e.add(new Etapa("9. Canal S do HSV (saturação)",
                "Plástico transparente ou branco tem pouca saturação (escuro aqui); rótulos e tampas, muita (claro).",
                f("saturação média: %.0f%%", media(sat) * 100), cinzaImg(sat, w, h)));

        boolean[] faixas = Filtros.faixasHsv(im);
        e.add(new Etapa("10. Segmentação por faixas HSV",
                "Regra fixa: claro e pouco saturado (PET, isopor) OU cor viva quente (rótulos, tampas).",
                f("%.1f%% da imagem marcada", 100.0 * contar(faixas) / n), mascaraImg(im, faixas)));

        boolean[] limpo = Filtros.fechar(Filtros.abrir(faixas, w, h), w, h);
        e.add(new Etapa("11. Morfologia: abertura + fechamento",
                "Abertura apaga pontos soltos; fechamento fecha buracos e une partes do mesmo objeto (elemento 3×3).",
                f("%.1f%% → %.1f%% da imagem", 100.0 * contar(faixas) / n, 100.0 * contar(limpo) / n),
                mascaraImg(im, limpo)));

        int[] rot = new int[n];
        List<int[]> comps = Filtros.componentes(limpo, w, h, rot);
        int minArea = Math.max(30, n / 2000);
        e.add(new Etapa("12. Componentes conexos",
                "Cada região ligada (vizinhança-8) vira um objeto candidato com sua caixa. O filtro não sabe o que é.",
                f("%d regiões com pelo menos %d px", contarGrandes(comps, minArea), minArea),
                componentesImg(im, rot, comps, minArea)));
        return e;
    }

    /** Grava cada etapa como PNG e um texto com as explicações, para usar no trabalho escrito. */
    public static void salvar(List<Etapa> etapas, String nomeImagem, File pasta) throws IOException {
        pasta.mkdirs();
        PrintWriter pw = new PrintWriter(new OutputStreamWriter(
                new java.io.FileOutputStream(new File(pasta, "etapas.txt")), "UTF-8"));
        try {
            pw.println("Filtros passo a passo – " + nomeImagem);
            pw.println();
            for (int i = 0; i < etapas.size(); i++) {
                Etapa e = etapas.get(i);
                String arquivo = String.format("%02d_%s.png", i + 1, slug(e.titulo));
                ImageIO.write(e.imagem, "png", new File(pasta, arquivo));
                pw.println(e.titulo + "  (" + arquivo + ")");
                pw.println("  " + e.explicacao);
                pw.println("  Resultado: " + e.info);
                pw.println();
            }
        } finally {
            pw.close();
        }
    }

    static String slug(String titulo) {
        String s = Normalizer.normalize(titulo, Normalizer.Form.NFD).replaceAll("\\p{M}", "");
        s = s.replaceFirst("^\\d+\\.\\s*", "").toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", "_");
        return s.replaceAll("^_+|_+$", "");
    }

    // ------------------------------------------------------------------ imagens das etapas

    private static BufferedImage cinzaImg(float[] v, int w, int h) {
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        int[] px = new int[w * h];
        for (int i = 0; i < px.length; i++) {
            int g = Filtros.nivel(v[i]);
            px[i] = (g << 16) | (g << 8) | g;
        }
        out.setRGB(0, 0, w, h, px, 0, w);
        return out;
    }

    private static BufferedImage binImg(boolean[] m, int w, int h) {
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        int[] px = new int[w * h];
        for (int i = 0; i < px.length; i++) {
            px[i] = m[i] ? 0xFFFFFF : 0;
        }
        out.setRGB(0, 0, w, h, px, 0, w);
        return out;
    }

    /** Pixels marcados em magenta sobre a foto escurecida. */
    private static BufferedImage mascaraImg(Filtros.Rgb im, boolean[] m) {
        BufferedImage out = new BufferedImage(im.w, im.h, BufferedImage.TYPE_INT_RGB);
        int[] px = new int[im.w * im.h];
        for (int i = 0; i < px.length; i++) {
            int r = Filtros.nivel(im.r[i]), g = Filtros.nivel(im.g[i]), b = Filtros.nivel(im.b[i]);
            if (m[i]) {
                r = (r + 255) / 2;
                g = g / 3;
                b = (b + 255) / 2;
            } else {
                r = r * 35 / 100;
                g = g * 35 / 100;
                b = b * 35 / 100;
            }
            px[i] = (r << 16) | (g << 8) | b;
        }
        out.setRGB(0, 0, im.w, im.h, px, 0, im.w);
        return out;
    }

    private static BufferedImage componentesImg(Filtros.Rgb im, int[] rot, List<int[]> comps, int minArea) {
        BufferedImage out = new BufferedImage(im.w, im.h, BufferedImage.TYPE_INT_RGB);
        int[] cores = new int[comps.size() + 1];
        for (int k = 0; k < comps.size(); k++) {
            cores[k + 1] = comps.get(k)[4] >= minArea ? Color.HSBtoRGB((k * 0.618034f) % 1f, 0.85f, 1f) & 0xFFFFFF : 0x606060;
        }
        int[] px = new int[im.w * im.h];
        for (int i = 0; i < px.length; i++) {
            if (rot[i] > 0) {
                px[i] = cores[rot[i]];
            } else {
                int y = Filtros.nivel((0.299f * im.r[i] + 0.587f * im.g[i] + 0.114f * im.b[i]) * 0.3f);
                px[i] = (y << 16) | (y << 8) | y;
            }
        }
        out.setRGB(0, 0, im.w, im.h, px, 0, im.w);
        Graphics2D g = out.createGraphics();
        g.setStroke(new BasicStroke(Math.max(1.5f, im.w / 400f)));
        for (int k = 0; k < comps.size(); k++) {
            int[] c = comps.get(k);
            if (c[4] >= minArea) {
                g.setColor(new Color(cores[k + 1]));
                g.drawRect(c[0], c[1], c[2] - c[0], c[3] - c[1]);
            }
        }
        g.dispose();
        return out;
    }

    // ------------------------------------------------------------------ estatísticas

    private static String f(String fmt, Object... args) {
        return String.format(PT, fmt, args);
    }

    private static double media(float[] v) {
        double s = 0;
        for (float x : v) {
            s += x;
        }
        return s / v.length;
    }

    private static double desvio(float[] v) {
        double m = media(v), s = 0;
        for (float x : v) {
            s += (x - m) * (x - m);
        }
        return Math.sqrt(s / v.length);
    }

    private static int contar(boolean[] m) {
        int n = 0;
        for (boolean b : m) {
            if (b) {
                n++;
            }
        }
        return n;
    }

    private static int contarGrandes(List<int[]> comps, int minArea) {
        int n = 0;
        for (int[] c : comps) {
            if (c[4] >= minArea) {
                n++;
            }
        }
        return n;
    }

    private static float percentil(float[] v, float q) {
        float[] c = v.clone();
        java.util.Arrays.sort(c);
        return c[Math.min(c.length - 1, (int) (q * c.length))];
    }
}
