package br.aps.plastico.ia;

import java.awt.AlphaComposite;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Consumer;

import javax.imageio.ImageIO;

import br.aps.plastico.dados.Projeto;
import br.aps.plastico.visao.Etapas;
import br.aps.plastico.visao.Filtros;

/** Processamento de muitas imagens de uma vez. */
public final class Lote {

    private Lote() {
    }

    /** Imagem não anotada e o quanto a IA ficou em dúvida nela. */
    public static final class Sugestao {
        public final File arquivo;
        public final float incerteza;
        public final double area;

        Sugestao(File arquivo, float incerteza, double area) {
            this.arquivo = arquivo;
            this.incerteza = incerteza;
            this.area = area;
        }
    }

    /**
     * Detecta plástico em todas as imagens e grava em {@code saida} uma cópia
     * marcada de cada uma e o relatorio.csv.
     */
    public static int detectarPasta(final Modelo modelo, List<File> imagens, final File saida, final float limiar,
            Consumer<String> log) throws Exception {
        saida.mkdirs();
        ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, Runtime.getRuntime().availableProcessors()));
        List<Future<String>> fs = new ArrayList<Future<String>>();
        for (final File f : imagens) {
            fs.add(pool.submit(() -> {
                BufferedImage img;
                try {
                    img = Projeto.lerImagem(f);
                } catch (Exception e) {
                    return f.getName() + ";erro;;;" + e.getMessage();
                }
                Detector.Resultado r = Detector.detectar(modelo, img);
                List<Detector.Regiao> regs = r.regioes(limiar);
                String base = f.getName().replaceAll("\\.[^.]+$", "");
                ImageIO.write(desenhar(img, r, limiar), "jpg", new File(saida, base + "_ia.jpg"));
                float conf = 0;
                for (Detector.Regiao g : regs) {
                    conf = Math.max(conf, g.confianca);
                }
                return String.format(Treinamento.PT, "%s;%s;%d;%.1f;%.0f", f.getName(),
                        regs.isEmpty() ? "nao" : "sim", regs.size(), r.areaPercentual(limiar), conf * 100);
            }));
        }
        PrintWriter pw = Projeto.escreverCsv(new File(saida, "relatorio.csv"), false);
        int ok = 0;
        try {
            pw.println("imagem;tem_plastico;regioes;area_plastico_%;maior_confianca_%");
            for (int i = 0; i < fs.size(); i++) {
                String linha = fs.get(i).get();
                pw.println(linha);
                if (!linha.contains(";erro;")) {
                    ok++;
                }
                if ((i + 1) % 10 == 0 || i == fs.size() - 1) {
                    log.accept(String.format("  %d/%d imagens analisadas", i + 1, fs.size()));
                }
            }
        } finally {
            pw.close();
            pool.shutdownNow();
        }
        return ok;
    }

    /** Imagens em que a IA mais ficou em dúvida: são as que mais ensinam se forem anotadas. */
    public static List<Sugestao> rankearIncerteza(final Modelo modelo, List<File> imagens, Consumer<String> log)
            throws Exception {
        ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, Runtime.getRuntime().availableProcessors()));
        List<Future<Sugestao>> fs = new ArrayList<Future<Sugestao>>();
        for (final File f : imagens) {
            fs.add(pool.submit(() -> {
                Detector.Resultado r = Detector.detectar(modelo, Projeto.lerImagem(f));
                return new Sugestao(f, r.incerteza, r.areaPercentual(modelo.limiar));
            }));
        }
        List<Sugestao> out = new ArrayList<Sugestao>();
        try {
            for (int i = 0; i < fs.size(); i++) {
                try {
                    out.add(fs.get(i).get());
                } catch (Exception e) {
                    log.accept("  não foi possível ler " + imagens.get(i).getName());
                }
            }
        } finally {
            pool.shutdownNow();
        }
        Collections.sort(out, new Comparator<Sugestao>() {
            @Override
            public int compare(Sugestao a, Sugestao b) {
                return Float.compare(b.incerteza, a.incerteza);
            }
        });
        return out;
    }

    /** Última etapa da tela "Filtros passo a passo": o que a rede neural vê na imagem. */
    public static Etapas.Etapa etapaIa(int numero, BufferedImage img, Detector.Resultado r, float limiar) {
        return new Etapas.Etapa(numero + ". Visão computacional (IA)",
                "A rede neural treinada com as anotações dá a probabilidade de plástico em cada célula; "
                        + "as regiões acima do limiar viram detecções.",
                String.format(Treinamento.PT, "%d região(ões) · %.1f%% da imagem com plástico (limiar %.0f%%)",
                        r.regioes(limiar).size(), r.areaPercentual(limiar), limiar * 100),
                desenhar(img, r, limiar));
    }

    /** Cópia da imagem com o mapa de calor e as caixas das regiões detectadas. */
    public static BufferedImage desenhar(BufferedImage img, Detector.Resultado r, float limiar) {
        BufferedImage out = Filtros.copiaRgb(img);
        int w = out.getWidth(), h = out.getHeight();
        Graphics2D g = out.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.5f));
        g.drawImage(r.imagemCalor(limiar), 0, 0, (int) Math.round(r.larguraGrade()), (int) Math.round(r.alturaGrade()),
                null);
        g.setComposite(AlphaComposite.SrcOver);

        List<Detector.Regiao> regs = r.regioes(limiar);
        int esp = Math.max(2, Math.round(Math.max(w, h) / 320f));
        Font fonte = new Font(Font.SANS_SERIF, Font.BOLD, Math.max(12, Math.max(w, h) / 55));
        g.setFont(fonte);
        FontMetrics fm = g.getFontMetrics();
        for (Detector.Regiao reg : regs) {
            Rectangle c = reg.caixa;
            g.setColor(new Color(255, 40, 40));
            g.setStroke(new BasicStroke(esp));
            g.drawRect(c.x, c.y, c.width, c.height);
            String txt = String.format("plástico %.0f%%", reg.confianca * 100);
            int tw = fm.stringWidth(txt) + 8, th = fm.getHeight();
            int ty = c.y - th >= 0 ? c.y - th : c.y;
            g.fillRect(c.x, ty, tw, th);
            g.setColor(Color.WHITE);
            g.drawString(txt, c.x + 4, ty + fm.getAscent());
        }
        String resumo = String.format(Treinamento.PT, "IA: %d região(ões) · %.1f%% da imagem com plástico",
                regs.size(), r.areaPercentual(limiar));
        int th = fm.getHeight() + 8;
        g.setColor(new Color(0, 0, 0, 170));
        g.fillRect(0, h - th, w, th);
        g.setColor(Color.WHITE);
        g.drawString(resumo, 8, h - th + 4 + fm.getAscent());
        g.dispose();
        return out;
    }
}
