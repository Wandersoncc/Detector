package br.aps.plastico.dados;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;
import java.util.function.Consumer;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import javax.imageio.ImageIO;

import br.aps.plastico.visao.Filtros;

/**
 * Exporta as imagens anotadas como um dataset pronto para treinar outros modelos de
 * visão computacional (YOLO/Ultralytics para detecção e máscaras PNG para
 * segmentação). É a "preparação dos dados" que o manual da APS pede.
 */
public final class ExportadorYolo {

    private ExportadorYolo() {
    }

    public static final class Opcoes {
        /** Reduz para 640 px no maior lado (tamanho padrão do YOLO). */
        public boolean reduzir640 = true;
        /** Também grava as máscaras PNG (0 = fundo, 1 = plástico, 255 = ignorar). */
        public boolean mascaras = true;
        public boolean incluirImportadas = true;
        /** Gera um .zip ao lado da pasta (prático para enviar ao Google Colab). */
        public boolean gerarZip = true;
        public float fracaoValidacao = 0.2f;
    }

    /** Resultado da exportação, para mostrar ao usuário. */
    public static final class Resumo {
        public int treino, validacao, objetos, semPlastico;
        public File pasta, zip;
    }

    public static Resumo exportar(Projeto projeto, File destino, Opcoes op, Consumer<String> log) throws IOException {
        Set<String> importadas = projeto.imagensDeDatasets();
        List<File> imgs = new ArrayList<File>();
        for (File f : projeto.imagensAnotadas()) {
            if (op.incluirImportadas || !importadas.contains(f.getName())) {
                imgs.add(f);
            }
        }
        if (imgs.isEmpty()) {
            throw new IOException("Nenhuma imagem anotada para exportar.");
        }
        if (destino.exists() && destino.list() != null && destino.list().length > 0) {
            throw new IOException("A pasta de destino já existe e não está vazia: " + destino);
        }
        Collections.shuffle(imgs, new Random(42));
        int nVal = imgs.size() >= 5 ? Math.max(1, Math.round(imgs.size() * op.fracaoValidacao)) : 0;

        Resumo r = new Resumo();
        r.pasta = destino;
        Set<String> usados = new HashSet<String>();
        for (int i = 0; i < imgs.size(); i++) {
            File f = imgs.get(i);
            String parte = i < nVal ? "val" : "train";
            BufferedImage img = Projeto.lerImagem(f);
            int w = img.getWidth(), h = img.getHeight();
            byte[] m = projeto.carregarMascara(f, w, h);
            if (op.reduzir640 && Math.max(w, h) > 640) {
                double esc = 640.0 / Math.max(w, h);
                int nw = Math.max(1, (int) Math.round(w * esc)), nh = Math.max(1, (int) Math.round(h * esc));
                img = Filtros.redimensionar(img, nw, nh);
                m = Projeto.redimensionarMascara(m, w, h, nw, nh);
                w = nw;
                h = nh;
            }
            String base = nomeSeguro(f.getName(), usados);
            gravar(new File(destino, "images/" + parte + "/" + base + ".jpg"), Projeto.jpg(img));

            List<String> linhas = caixas(m, w, h);
            r.objetos += linhas.size();
            if (linhas.isEmpty()) {
                r.semPlastico++;
            }
            File rotulo = new File(destino, "labels/" + parte + "/" + base + ".txt");
            rotulo.getParentFile().mkdirs();
            Files.write(rotulo.toPath(), linhas, Projeto.UTF8);

            if (op.mascaras) {
                File png = new File(destino, "masks/" + parte + "/" + base + ".png");
                png.getParentFile().mkdirs();
                ImageIO.write(mascaraPng(m, w, h), "png", png);
            }
            if (parte.equals("val")) {
                r.validacao++;
            } else {
                r.treino++;
            }
            if ((i + 1) % 50 == 0) {
                log.accept(String.format("  %d/%d imagens exportadas...", i + 1, imgs.size()));
            }
        }
        int deDatasets = contarImportadas(imgs, importadas);
        escreverConfig(destino, r, op, imgs.size() - deDatasets, deDatasets);
        if (op.gerarZip) {
            r.zip = new File(destino.getParentFile(), destino.getName() + ".zip");
            compactar(destino, r.zip);
        }
        log.accept(String.format("Dataset exportado: %d imagens (treino %d, validação %d), %d objetos, %d sem plástico.",
                imgs.size(), r.treino, r.validacao, r.objetos, r.semPlastico));
        return r;
    }

    /** Erosões usadas para separar objetos que se encostam na anotação. */
    private static final int SEPARACAO = 3;

    /**
     * Uma caixa YOLO por objeto de plástico: "classe x_centro y_centro largura altura",
     * tudo de 0 a 1. A máscara é erodida antes de achar as regiões, para que objetos
     * encostados não virem uma caixa só; depois a caixa recupera a margem erodida.
     */
    static List<String> caixas(byte[] m, int w, int h) {
        boolean[] plast = new boolean[w * h];
        for (int i = 0; i < plast.length; i++) {
            plast[i] = m[i] == Projeto.PLASTICO;
        }
        boolean[] miolo = plast;
        for (int k = 0; k < SEPARACAO; k++) {
            miolo = Filtros.erodir(miolo, w, h);
        }
        int[] rotOrig = new int[w * h];
        List<int[]> originais = Filtros.componentes(plast, w, h, rotOrig);
        boolean[] coberto = new boolean[originais.size() + 1];
        for (int i = 0; i < miolo.length; i++) {
            if (miolo[i]) {
                coberto[rotOrig[i]] = true;
            }
        }
        List<int[]> objetos = new ArrayList<int[]>();
        for (int[] c : Filtros.componentes(miolo, w, h, null)) {
            objetos.add(new int[] { Math.max(0, c[0] - SEPARACAO), Math.max(0, c[1] - SEPARACAO),
                    Math.min(w - 1, c[2] + SEPARACAO), Math.min(h - 1, c[3] + SEPARACAO), c[4] });
        }
        // objetos finos demais somem na erosão: ficam com a caixa original
        for (int k = 0; k < originais.size(); k++) {
            if (!coberto[k + 1]) {
                objetos.add(originais.get(k));
            }
        }
        int minLado = 3, minArea = Math.max(16, w * h / 10000);
        List<String> out = new ArrayList<String>();
        for (int[] c : objetos) {
            int bw = c[2] - c[0] + 1, bh = c[3] - c[1] + 1;
            if (bw * bh < minArea || bw < minLado || bh < minLado) {
                continue;
            }
            out.add(String.format(Locale.ROOT, "0 %.6f %.6f %.6f %.6f", (c[0] + bw / 2.0) / w, (c[1] + bh / 2.0) / h,
                    bw / (double) w, bh / (double) h));
        }
        return out;
    }

    private static BufferedImage mascaraPng(byte[] m, int w, int h) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
        byte[] d = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
        for (int i = 0; i < d.length; i++) {
            d[i] = m[i] == Projeto.PLASTICO ? 1 : (m[i] == Projeto.IGNORAR ? (byte) 255 : 0);
        }
        return img;
    }

    private static String nomeSeguro(String nome, Set<String> usados) {
        String base = nome.replaceAll("\\.[^.]+$", "").replaceAll("[^A-Za-z0-9_-]+", "_").replaceAll("^_+|_+$", "");
        if (base.isEmpty()) {
            base = "imagem";
        }
        String n = base;
        for (int i = 2; !usados.add(n.toLowerCase(Locale.ROOT)); i++) {
            n = base + "_" + i;
        }
        return n;
    }

    private static int contarImportadas(List<File> imgs, Set<String> importadas) {
        int n = 0;
        for (File f : imgs) {
            if (importadas.contains(f.getName())) {
                n++;
            }
        }
        return n;
    }

    private static void gravar(File f, byte[] dados) throws IOException {
        f.getParentFile().mkdirs();
        Files.write(f.toPath(), dados);
    }

    private static void escreverConfig(File destino, Resumo r, Opcoes op, int proprias, int importadas)
            throws IOException {
        // sem "path": o Ultralytics usa a pasta onde está o data.yaml
        PrintWriter pw = escritor(new File(destino, "data.yaml"));
        try {
            pw.println("# Dataset exportado pelo Detector de Plastico (APS)");
            pw.println("train: images/train");
            pw.println(r.validacao > 0 ? "val: images/val" : "val: images/train");
            pw.println();
            pw.println("nc: 1");
            pw.println("names: ['plastico']");
        } finally {
            pw.close();
        }
        pw = escritor(new File(destino, "classes.txt"));
        try {
            pw.println("plastico");
        } finally {
            pw.close();
        }
        pw = escritor(new File(destino, "LEIAME.txt"));
        try {
            pw.println("Dataset de plástico em rios – exportado pelo Detector de Plástico (APS)");
            pw.println("Data: " + new SimpleDateFormat("dd/MM/yyyy HH:mm").format(new Date()));
            pw.println();
            pw.println(String.format("Imagens: %d (treino %d, validação %d) – %d anotadas pelo grupo, %d de datasets públicos",
                    r.treino + r.validacao, r.treino, r.validacao, proprias, importadas));
            pw.println(String.format("Objetos (caixas): %d · Imagens sem plástico (só fundo): %d", r.objetos, r.semPlastico));
            pw.println("Classe: 0 = plastico");
            pw.println();
            pw.println("Estrutura (formato YOLO / Ultralytics):");
            pw.println("  images/train, images/val   imagens JPG" + (op.reduzir640 ? " (maior lado com até 640 px)" : ""));
            pw.println("  labels/train, labels/val   uma linha por objeto: classe x_centro y_centro largura altura (0 a 1)");
            if (op.mascaras) {
                pw.println("  masks/train, masks/val     máscaras PNG para segmentação: 0 = fundo, 1 = plástico, 255 = ignorar");
            }
            pw.println("  data.yaml                  configuração para o YOLO");
            pw.println();
            pw.println("As caixas foram geradas a partir das máscaras de anotação: cada região de plástico");
            pw.println("virou um objeto (a máscara é erodida 3 px antes, para separar objetos encostados).");
            pw.println("Objetos sobrepostos na foto podem sair numa caixa só; para segmentação, use as máscaras.");
            pw.println("Áreas \"ignorar\" não geram caixas.");
            pw.println();
            pw.println("Como treinar um YOLO com este dataset (Python; por exemplo, no Google Colab):");
            pw.println("  pip install ultralytics");
            pw.println("  yolo detect train data=CAMINHO/DA/PASTA/data.yaml model=yolov8n.pt imgsz=640 epochs=50");
            pw.println();
            pw.println("Imagens com nome mendeley_* vêm de um dataset público sob licença CC BY 4.0:");
            pw.println("cite a fonte (veja a seção Créditos no LEIAME.md do projeto).");
        } finally {
            pw.close();
        }
    }

    private static PrintWriter escritor(File f) throws IOException {
        f.getParentFile().mkdirs();
        return new PrintWriter(new OutputStreamWriter(new FileOutputStream(f), Projeto.UTF8));
    }

    private static void compactar(File pasta, File zip) throws IOException {
        ZipOutputStream out = new ZipOutputStream(new FileOutputStream(zip), Projeto.UTF8);
        try {
            adicionar(out, pasta, pasta.getName());
        } finally {
            out.close();
        }
    }

    private static void adicionar(ZipOutputStream out, File f, String caminho) throws IOException {
        if (f.isDirectory()) {
            File[] fs = f.listFiles();
            if (fs != null) {
                for (File g : fs) {
                    adicionar(out, g, caminho + "/" + g.getName());
                }
            }
            return;
        }
        out.putNextEntry(new ZipEntry(caminho));
        InputStream in = new FileInputStream(f);
        try {
            byte[] buf = new byte[16384];
            int n;
            while ((n = in.read(buf)) > 0) {
                out.write(buf, 0, n);
            }
        } finally {
            in.close();
        }
        out.closeEntry();
    }
}
