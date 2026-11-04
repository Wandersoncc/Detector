package br.aps.plastico.dados;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.function.Consumer;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;

import br.aps.plastico.visao.Filtros;

/**
 * Pastas do projeto e o conjunto de dados (imagens + anotações).
 *
 * <pre>
 * dados/imagens/     imagens usadas no treino
 * dados/anotacoes/   uma máscara PNG por imagem: preto = fundo, branco = plástico, cinza = ignorar
 * modelos/           modelo treinado, histórico de eficiência e desempenho por imagem
 * resultados/        saídas da detecção em lote
 * </pre>
 */
public final class Projeto {

    public static final byte FUNDO = 0, PLASTICO = (byte) 255, IGNORAR = (byte) 128;
    public static final Charset UTF8 = Charset.forName("UTF-8");
    private static final String[] EXTENSOES = { ".jpg", ".jpeg", ".png", ".bmp", ".gif" };

    public final File raiz, pastaImagens, pastaAnotacoes, pastaModelos, pastaResultados;

    public Projeto(File raiz) {
        this.raiz = raiz.getAbsoluteFile();
        pastaImagens = new File(this.raiz, "dados" + File.separator + "imagens");
        pastaAnotacoes = new File(this.raiz, "dados" + File.separator + "anotacoes");
        pastaModelos = new File(this.raiz, "modelos");
        pastaResultados = new File(this.raiz, "resultados");
        for (File d : new File[] { pastaImagens, pastaAnotacoes, pastaModelos, pastaResultados }) {
            d.mkdirs();
        }
    }

    /** Pasta onde está o .jar (ou a definida com -Dprojeto=...). */
    public static File pastaPadrao() {
        String p = System.getProperty("projeto");
        if (p != null && !p.trim().isEmpty()) {
            return new File(p);
        }
        try {
            File loc = new File(Projeto.class.getProtectionDomain().getCodeSource().getLocation().toURI());
            if (loc.isFile()) {
                return loc.getParentFile();
            }
        } catch (Exception e) {
            // cai no diretório atual
        }
        return new File(System.getProperty("user.dir"));
    }

    // ------------------------------------------------------------------ imagens

    public static boolean ehImagem(String nome) {
        String n = nome.toLowerCase(Locale.ROOT);
        for (String e : EXTENSOES) {
            if (n.endsWith(e)) {
                return true;
            }
        }
        return false;
    }

    public List<File> listarImagens() {
        return listarImagensEm(pastaImagens, false);
    }

    public static List<File> listarImagensEm(File pasta, boolean recursivo) {
        List<File> out = new ArrayList<File>();
        coletar(pasta, recursivo, out);
        Collections.sort(out, ORDEM_NATURAL);
        return out;
    }

    private static void coletar(File pasta, boolean recursivo, List<File> out) {
        File[] fs = pasta.listFiles();
        if (fs == null) {
            return;
        }
        for (File f : fs) {
            if (f.isDirectory()) {
                if (recursivo) {
                    coletar(f, true, out);
                }
            } else if (ehImagem(f.getName())) {
                out.add(f);
            }
        }
    }

    public List<File> imagensAnotadas() {
        List<File> out = new ArrayList<File>();
        for (File f : listarImagens()) {
            if (anotada(f)) {
                out.add(f);
            }
        }
        return out;
    }

    public File arquivoMascara(File imagem) {
        return new File(pastaAnotacoes, imagem.getName() + ".png");
    }

    public boolean anotada(File imagem) {
        return arquivoMascara(imagem).isFile();
    }

    /** Lê qualquer imagem suportada e devolve uma cópia TYPE_INT_RGB. */
    public static BufferedImage lerImagem(File f) throws IOException {
        BufferedImage img = ImageIO.read(f);
        if (img == null) {
            throw new IOException("Formato não suportado: " + f.getName() + " (use JPG, PNG ou BMP)");
        }
        return Filtros.copiaRgb(img);
    }

    // ------------------------------------------------------------------ máscaras

    /** Máscara da imagem no tamanho w x h, ou {@code null} se ainda não foi anotada. */
    public byte[] carregarMascara(File imagem, int w, int h) throws IOException {
        File f = arquivoMascara(imagem);
        if (!f.isFile()) {
            return null;
        }
        BufferedImage m = ImageIO.read(f);
        if (m == null) {
            throw new IOException("Anotação corrompida: " + f.getName());
        }
        int mw = m.getWidth(), mh = m.getHeight();
        int[] px;
        if (m.getRaster().getNumBands() == 1) {
            // lê o valor bruto (getRGB aplicaria correção de gama em imagens cinza)
            px = m.getRaster().getSamples(0, 0, mw, mh, 0, (int[]) null);
            for (int i = 0; i < px.length; i++) {
                px[i] = (px[i] & 255) << 16;
            }
        } else {
            px = m.getRGB(0, 0, mw, mh, null, 0, mw);
        }
        byte[] out = new byte[w * h];
        for (int y = 0; y < h; y++) {
            int sy = Math.min(mh - 1, (int) ((y + 0.5) * mh / h));
            for (int x = 0; x < w; x++) {
                int sx = Math.min(mw - 1, (int) ((x + 0.5) * mw / w));
                int v = (px[sy * mw + sx] >> 16) & 255;
                out[y * w + x] = v >= 192 ? PLASTICO : (v >= 64 ? IGNORAR : FUNDO);
            }
        }
        return out;
    }

    public void salvarMascara(File imagem, byte[] mascara, int w, int h) throws IOException {
        BufferedImage m = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
        byte[] dados = ((DataBufferByte) m.getRaster().getDataBuffer()).getData();
        System.arraycopy(mascara, 0, dados, 0, w * h);
        if (!ImageIO.write(m, "png", arquivoMascara(imagem))) {
            throw new IOException("Não foi possível gravar a anotação.");
        }
    }

    public void removerMascara(File imagem) {
        File f = arquivoMascara(imagem);
        if (f.isFile()) {
            f.delete();
        }
    }

    // ------------------------------------------------------------------ arquivos do modelo

    public File arquivoModelo() {
        return new File(pastaModelos, "modelo_plastico.bin");
    }

    public File arquivoHistorico() {
        return new File(pastaModelos, "historico_eficiencia.csv");
    }

    public File arquivoDesempenho() {
        return new File(pastaModelos, "desempenho_por_imagem.csv");
    }

    /** Nomes das imagens que vieram de datasets externos (uma por linha). */
    public File arquivoImportadas() {
        return new File(pastaImagens.getParentFile(), "importadas_de_datasets.txt");
    }

    public Set<String> imagensDeDatasets() {
        Set<String> s = new HashSet<String>();
        File f = arquivoImportadas();
        if (f.isFile()) {
            try {
                for (String l : Files.readAllLines(f.toPath(), UTF8)) {
                    if (!l.trim().isEmpty()) {
                        s.add(l.trim());
                    }
                }
            } catch (IOException e) {
                // sem a lista, todas contam como imagens próprias
            }
        }
        return s;
    }

    public void registrarImportadas(List<String> nomes) throws IOException {
        if (nomes.isEmpty()) {
            return;
        }
        PrintWriter pw = new PrintWriter(new OutputStreamWriter(new java.io.FileOutputStream(arquivoImportadas(), true), UTF8));
        try {
            for (String n : nomes) {
                pw.println(n);
            }
        } finally {
            pw.close();
        }
    }

    /** Lê um CSV separado por ';' (ignora a primeira linha, o cabeçalho). */
    public static List<String[]> lerCsv(File f) {
        List<String[]> out = new ArrayList<String[]>();
        if (!f.isFile()) {
            return out;
        }
        try {
            List<String> linhas = Files.readAllLines(f.toPath(), UTF8);
            for (int i = 1; i < linhas.size(); i++) {
                String l = linhas.get(i).trim();
                if (!l.isEmpty()) {
                    out.add(l.split(";", -1));
                }
            }
        } catch (IOException e) {
            // arquivo em uso ou corrompido: trata como vazio
        }
        return out;
    }

    /** Abre um CSV em UTF-8 com BOM (o Excel reconhece os acentos). */
    public static PrintWriter escreverCsv(File f, boolean anexar) throws IOException {
        boolean novo = !f.isFile() || !anexar;
        java.io.FileOutputStream fos = new java.io.FileOutputStream(f, anexar);
        if (novo) {
            fos.write(new byte[] { (byte) 0xEF, (byte) 0xBB, (byte) 0xBF });
        }
        return new PrintWriter(new OutputStreamWriter(fos, UTF8));
    }

    public static double numero(String s) {
        try {
            return Double.parseDouble(s.trim().replace("%", "").replace(',', '.'));
        } catch (NumberFormatException e) {
            return Double.NaN;
        }
    }

    // ------------------------------------------------------------------ importação

    /**
     * Copia imagens para dados/imagens. Aceita arquivos, pastas (com subpastas) e
     * arquivos .zip. Imagens repetidas (mesmo conteúdo) são ignoradas.
     */
    public int importarImagens(List<File> fontes, Consumer<String> log) throws IOException {
        Set<String> hashes = hashesExistentes();
        int[] cont = new int[3]; // novas, repetidas, inválidas
        for (File f : fontes) {
            importarFonte(f, hashes, cont, log);
        }
        log.accept(String.format("Importação concluída: %d novas, %d repetidas ignoradas, %d inválidas.",
                cont[0], cont[1], cont[2]));
        return cont[0];
    }

    private void importarFonte(File f, Set<String> hashes, int[] cont, Consumer<String> log) throws IOException {
        String nome = f.getName().toLowerCase(Locale.ROOT);
        if (f.isDirectory()) {
            File[] fs = f.listFiles();
            if (fs != null) {
                for (File g : fs) {
                    importarFonte(g, hashes, cont, log);
                }
            }
        } else if (nome.endsWith(".zip")) {
            ZipInputStream zip = new ZipInputStream(new FileInputStream(f), UTF8);
            try {
                ZipEntry e;
                while ((e = zip.getNextEntry()) != null) {
                    String n = e.getName().replace('\\', '/');
                    n = n.substring(n.lastIndexOf('/') + 1);
                    if (e.isDirectory() || n.startsWith("._") || e.getName().contains("__MACOSX")) {
                        continue;
                    }
                    if (ehImagem(n)) {
                        registrar(n, lerTudo(zip), hashes, cont, log);
                    } else if (n.toLowerCase(Locale.ROOT).endsWith(".webp")) {
                        log.accept("  ignorada (WEBP não suportado, converta para JPG): " + n);
                        cont[2]++;
                    }
                }
            } finally {
                zip.close();
            }
        } else if (ehImagem(nome)) {
            registrar(f.getName(), Files.readAllBytes(f.toPath()), hashes, cont, log);
        } else if (nome.endsWith(".webp")) {
            log.accept("  ignorada (WEBP não suportado, converta para JPG): " + f.getName());
            cont[2]++;
        }
    }

    private void registrar(String nome, byte[] dados, Set<String> hashes, int[] cont, Consumer<String> log)
            throws IOException {
        ImagemImportada im = preparar(nome, dados);
        if (im == null) {
            log.accept("  ignorada (não é uma imagem legível): " + nome);
            cont[2]++;
            return;
        }
        if (gravarImagemNova(im.nome, im.dados, hashes) == null) {
            cont[1]++;
        } else {
            cont[0]++;
        }
    }

    /** Lado máximo guardado no projeto: a IA trabalha em 640 px, então 1280 px sobra para anotar. */
    public static final int LADO_MAXIMO = 1280;

    /** Imagem pronta para entrar no projeto (reduzida se era maior que {@link #LADO_MAXIMO}). */
    public static final class ImagemImportada {
        public final String nome;
        public final byte[] dados;
        public final int largura, altura;
        /** Fator aplicado ao tamanho original (1 = sem redução). */
        public final double escala;

        ImagemImportada(String nome, byte[] dados, int largura, int altura, double escala) {
            this.nome = nome;
            this.dados = dados;
            this.largura = largura;
            this.altura = altura;
            this.escala = escala;
        }
    }

    /** Valida a imagem e reduz fotos grandes; devolve {@code null} se não for uma imagem legível. */
    public static ImagemImportada preparar(String nome, byte[] dados) throws IOException {
        BufferedImage img = ImageIO.read(new ByteArrayInputStream(dados));
        if (img == null) {
            return null;
        }
        int w = img.getWidth(), h = img.getHeight();
        if (Math.max(w, h) <= LADO_MAXIMO) {
            return new ImagemImportada(nome, dados, w, h, 1.0);
        }
        double esc = LADO_MAXIMO / (double) Math.max(w, h);
        int nw = Math.max(1, (int) Math.round(w * esc)), nh = Math.max(1, (int) Math.round(h * esc));
        BufferedImage menor = Filtros.redimensionar(Filtros.copiaRgb(img), nw, nh);
        String base = nome.replaceAll("\\.[^.]+$", "");
        return new ImagemImportada(base + ".jpg", jpg(menor), nw, nh, esc);
    }

    /** Codifica em JPEG com qualidade 92%. */
    public static byte[] jpg(BufferedImage img) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ImageWriter jpg = ImageIO.getImageWritersByFormatName("jpg").next();
        ImageWriteParam p = jpg.getDefaultWriteParam();
        p.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
        p.setCompressionQuality(0.92f);
        ImageOutputStream ios = ImageIO.createImageOutputStream(out);
        try {
            jpg.setOutput(ios);
            jpg.write(null, new IIOImage(img, null, null), p);
        } finally {
            ios.close();
            jpg.dispose();
        }
        return out.toByteArray();
    }

    /** Redimensiona uma máscara pelo vizinho mais próximo (não mistura os rótulos). */
    public static byte[] redimensionarMascara(byte[] m, int w0, int h0, int w, int h) {
        byte[] out = new byte[w * h];
        for (int y = 0; y < h; y++) {
            int sy = Math.min(h0 - 1, (int) ((y + 0.5) * h0 / h));
            for (int x = 0; x < w; x++) {
                int sx = Math.min(w0 - 1, (int) ((x + 0.5) * w0 / w));
                out[y * w + x] = m[sy * w0 + sx];
            }
        }
        return out;
    }

    /** Grava a imagem com nome único; devolve {@code null} se o conteúdo já existe. */
    public File gravarImagemNova(String nome, byte[] dados, Set<String> hashes) throws IOException {
        String h = md5(dados);
        if (!hashes.add(h)) {
            return null;
        }
        String limpo = nome.replaceAll("[\\\\/:*?\"<>|]", "_");
        int p = limpo.lastIndexOf('.');
        String base = p > 0 ? limpo.substring(0, p) : limpo, ext = p > 0 ? limpo.substring(p) : ".jpg";
        File destino = new File(pastaImagens, limpo);
        for (int i = 2; destino.exists(); i++) {
            destino = new File(pastaImagens, base + "_" + i + ext);
        }
        Files.write(destino.toPath(), dados);
        return destino;
    }

    public Set<String> hashesExistentes() throws IOException {
        Set<String> s = new HashSet<String>();
        for (File f : listarImagens()) {
            s.add(md5(Files.readAllBytes(f.toPath())));
        }
        return s;
    }

    public static String md5(byte[] dados) {
        try {
            byte[] d = MessageDigest.getInstance("MD5").digest(dados);
            StringBuilder sb = new StringBuilder();
            for (byte b : d) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    public static byte[] lerTudo(InputStream in) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buf = new byte[16384];
        int n;
        while ((n = in.read(buf)) > 0) {
            out.write(buf, 0, n);
        }
        return out.toByteArray();
    }

    // ------------------------------------------------------------------ ordenação

    /** Ordena "images (2)" antes de "images (10)". */
    public static final Comparator<File> ORDEM_NATURAL = new Comparator<File>() {
        @Override
        public int compare(File a, File b) {
            return compararNatural(a.getName(), b.getName());
        }
    };

    public static int compararNatural(String a, String b) {
        int i = 0, j = 0;
        while (i < a.length() && j < b.length()) {
            char ca = a.charAt(i), cb = b.charAt(j);
            if (Character.isDigit(ca) && Character.isDigit(cb)) {
                int fi = i, fj = j;
                while (fi < a.length() && Character.isDigit(a.charAt(fi))) {
                    fi++;
                }
                while (fj < b.length() && Character.isDigit(b.charAt(fj))) {
                    fj++;
                }
                String na = a.substring(i, fi).replaceFirst("^0+(?=.)", "");
                String nb = b.substring(j, fj).replaceFirst("^0+(?=.)", "");
                int c = na.length() != nb.length() ? na.length() - nb.length() : na.compareTo(nb);
                if (c != 0) {
                    return c;
                }
                i = fi;
                j = fj;
            } else {
                int c = Character.toLowerCase(ca) - Character.toLowerCase(cb);
                if (c != 0) {
                    return c;
                }
                i++;
                j++;
            }
        }
        return (a.length() - i) - (b.length() - j);
    }
}
