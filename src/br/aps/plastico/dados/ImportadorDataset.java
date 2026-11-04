package br.aps.plastico.dados;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Polygon;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;

import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

/**
 * Importa conjuntos de dados já rotulados (baixados de sites como Roboflow,
 * Kaggle etc.) nos formatos mais comuns:
 * <ul>
 * <li>YOLO (caixas ou polígonos de segmentação, arquivos .txt + data.yaml)</li>
 * <li>Pascal VOC (arquivos .xml)</li>
 * </ul>
 * O usuário escolhe quais classes contam como plástico; as demais viram fundo.
 */
public final class ImportadorDataset {

    private ImportadorDataset() {
    }

    static final class Objeto {
        int classe;
        boolean caixa; // caixa (x0,y0,x1,y1) ou polígono (x1,y1,x2,y2,...)
        boolean normalizado; // coordenadas 0..1 (YOLO) ou pixels (VOC)
        float[] pontos;
    }

    static final class Item {
        File imagem;
        final List<Objeto> objetos = new ArrayList<Objeto>();
    }

    public static final class Analise {
        public String formato;
        public final List<String> classes = new ArrayList<String>();
        public int[] contagem = new int[0];
        public int imagensComRotulo;
        final List<Item> itens = new ArrayList<Item>();

        public int totalImagens() {
            return itens.size();
        }
    }

    public static Analise analisar(File pasta) throws IOException {
        List<File> imagens = Projeto.listarImagensEm(pasta, true);
        if (imagens.isEmpty()) {
            throw new IOException("Nenhuma imagem (JPG/PNG/BMP) encontrada em " + pasta);
        }
        List<File> xmls = new ArrayList<File>();
        listarXml(pasta, xmls);
        Analise a = xmls.isEmpty() ? analisarYolo(pasta, imagens) : analisarVoc(xmls, imagens);
        if (a.imagensComRotulo == 0) {
            throw new IOException("Não encontrei rótulos YOLO (.txt) nem Pascal VOC (.xml) para as imagens.\n"
                    + "Para importar só as imagens, use 'Importar imagens'.");
        }
        a.contagem = new int[a.classes.size()];
        for (Item it : a.itens) {
            for (Objeto o : it.objetos) {
                if (o.classe < a.contagem.length) {
                    a.contagem[o.classe]++;
                }
            }
        }
        return a;
    }

    // ------------------------------------------------------------------ YOLO

    private static Analise analisarYolo(File pasta, List<File> imagens) throws IOException {
        Analise a = new Analise();
        a.formato = "YOLO";
        boolean temPastaLabels = false;
        int maiorClasse = -1;
        for (File img : imagens) {
            File rot = rotuloYolo(img);
            Item it = new Item();
            it.imagem = img;
            if (rot != null) {
                a.imagensComRotulo++;
                temPastaLabels = true;
                for (String linha : Files.readAllLines(rot.toPath(), Projeto.UTF8)) {
                    String[] p = linha.trim().split("\\s+");
                    if (p.length < 5) {
                        continue;
                    }
                    Objeto o = new Objeto();
                    o.classe = (int) Double.parseDouble(p[0]);
                    o.normalizado = true;
                    float[] v = new float[p.length - 1];
                    for (int i = 1; i < p.length; i++) {
                        v[i - 1] = Float.parseFloat(p[i]);
                    }
                    if (v.length == 4) {
                        o.caixa = true;
                        o.pontos = new float[] { v[0] - v[2] / 2, v[1] - v[3] / 2, v[0] + v[2] / 2, v[1] + v[3] / 2 };
                    } else {
                        o.pontos = v.length % 2 == 0 ? v : java.util.Arrays.copyOf(v, v.length - 1);
                    }
                    maiorClasse = Math.max(maiorClasse, o.classe);
                    it.objetos.add(o);
                }
            }
            a.itens.add(it);
        }
        // imagem sem .txt num dataset YOLO = imagem sem objetos (só fundo)
        if (!temPastaLabels) {
            a.itens.clear();
        }
        a.classes.addAll(nomesClasses(pasta));
        while (a.classes.size() <= maiorClasse) {
            a.classes.add("classe " + a.classes.size());
        }
        return a;
    }

    private static File rotuloYolo(File img) {
        String nome = img.getName();
        int p = nome.lastIndexOf('.');
        String txt = (p > 0 ? nome.substring(0, p) : nome) + ".txt";
        File mesmaPasta = new File(img.getParentFile(), txt);
        if (mesmaPasta.isFile()) {
            return mesmaPasta;
        }
        // estrutura padrão: .../images/... -> .../labels/...
        List<String> partes = new ArrayList<String>();
        for (File f = img.getParentFile(); f != null; f = f.getParentFile()) {
            partes.add(0, f.getName().isEmpty() ? f.getPath() : f.getName());
        }
        for (int i = partes.size() - 1; i >= 0; i--) {
            if (partes.get(i).equalsIgnoreCase("images")) {
                File dir = null;
                for (int k = 0; k < partes.size(); k++) {
                    String s = k == i ? "labels" : partes.get(k);
                    dir = dir == null ? new File(s) : new File(dir, s);
                }
                File f = new File(dir, txt);
                if (f.isFile()) {
                    return f;
                }
            }
        }
        return null;
    }

    private static List<String> nomesClasses(File pasta) {
        List<File> candidatos = new ArrayList<File>();
        File[] fs = pasta.listFiles();
        if (fs != null) {
            for (File f : fs) {
                String n = f.getName().toLowerCase(Locale.ROOT);
                if (n.endsWith(".yaml") || n.endsWith(".yml") || n.equals("classes.txt") || n.equals("obj.names")
                        || n.equals("_darknet.labels")) {
                    candidatos.add(f);
                }
            }
        }
        for (File f : candidatos) {
            try {
                List<String> linhas = Files.readAllLines(f.toPath(), Projeto.UTF8);
                List<String> nomes = f.getName().toLowerCase(Locale.ROOT).matches(".*\\.ya?ml") ? nomesYaml(linhas)
                        : naoVazias(linhas);
                if (!nomes.isEmpty()) {
                    return nomes;
                }
            } catch (IOException e) {
                // tenta o próximo
            }
        }
        return new ArrayList<String>();
    }

    private static List<String> naoVazias(List<String> linhas) {
        List<String> out = new ArrayList<String>();
        for (String l : linhas) {
            if (!l.trim().isEmpty()) {
                out.add(l.trim());
            }
        }
        return out;
    }

    static List<String> nomesYaml(List<String> linhas) {
        List<String> out = new ArrayList<String>();
        for (int i = 0; i < linhas.size(); i++) {
            String l = linhas.get(i).trim();
            if (!l.startsWith("names:")) {
                continue;
            }
            String resto = l.substring(6).trim();
            if (resto.startsWith("[")) {
                resto = resto.substring(1, resto.lastIndexOf(']') > 0 ? resto.lastIndexOf(']') : resto.length());
                for (String s : resto.split(",")) {
                    if (!s.trim().isEmpty()) {
                        out.add(semAspas(s));
                    }
                }
            } else if (resto.startsWith("{")) {
                resto = resto.replace("{", "").replace("}", "");
                for (String s : resto.split(",")) {
                    int c = s.indexOf(':');
                    if (c > 0) {
                        out.add(semAspas(s.substring(c + 1)));
                    }
                }
            } else {
                for (int j = i + 1; j < linhas.size(); j++) {
                    String s = linhas.get(j);
                    if (s.trim().isEmpty()) {
                        continue;
                    }
                    if (!(s.startsWith(" ") || s.startsWith("\t") || s.startsWith("-"))) {
                        break;
                    }
                    s = s.trim();
                    if (s.startsWith("-")) {
                        out.add(semAspas(s.substring(1)));
                    } else if (s.matches("\\d+\\s*:.*")) {
                        out.add(semAspas(s.substring(s.indexOf(':') + 1)));
                    }
                }
            }
            break;
        }
        return out;
    }

    private static String semAspas(String s) {
        s = s.trim();
        if (s.length() >= 2 && (s.startsWith("'") || s.startsWith("\""))) {
            s = s.substring(1, s.length() - 1);
        }
        return s;
    }

    // ------------------------------------------------------------------ Pascal VOC

    private static void listarXml(File pasta, List<File> out) {
        File[] fs = pasta.listFiles();
        if (fs == null) {
            return;
        }
        for (File f : fs) {
            if (f.isDirectory()) {
                listarXml(f, out);
            } else if (f.getName().toLowerCase(Locale.ROOT).endsWith(".xml")) {
                out.add(f);
            }
        }
    }

    private static Analise analisarVoc(List<File> xmls, List<File> imagens) throws IOException {
        Analise a = new Analise();
        a.formato = "Pascal VOC";
        Map<String, File> porNome = new HashMap<String, File>();
        for (File f : imagens) {
            porNome.put(f.getName().toLowerCase(Locale.ROOT), f);
        }
        Map<String, Integer> classes = new LinkedHashMap<String, Integer>();
        Map<File, Item> itens = new LinkedHashMap<File, Item>();
        for (File xml : xmls) {
            Document d;
            try {
                DocumentBuilderFactory fab = DocumentBuilderFactory.newInstance();
                fab.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
                fab.setExpandEntityReferences(false);
                d = fab.newDocumentBuilder().parse(xml);
            } catch (Exception e) {
                continue;
            }
            if (!"annotation".equals(d.getDocumentElement().getTagName())) {
                continue;
            }
            String arq = texto(d.getDocumentElement(), "filename");
            File img = arq == null ? null : porNome.get(new File(arq).getName().toLowerCase(Locale.ROOT));
            if (img == null) {
                String base = xml.getName().replaceAll("(?i)\\.xml$", "").toLowerCase(Locale.ROOT);
                for (String ext : new String[] { ".jpg", ".jpeg", ".png", ".bmp" }) {
                    if (img == null) {
                        img = porNome.get(base + ext);
                    }
                }
            }
            if (img == null) {
                continue;
            }
            Item it = itens.get(img);
            if (it == null) {
                it = new Item();
                it.imagem = img;
                itens.put(img, it);
                a.imagensComRotulo++;
            }
            NodeList objs = d.getElementsByTagName("object");
            for (int i = 0; i < objs.getLength(); i++) {
                Element ob = (Element) objs.item(i);
                String nome = texto(ob, "name");
                NodeList caixas = ob.getElementsByTagName("bndbox");
                if (nome == null || caixas.getLength() == 0) {
                    continue;
                }
                Element bb = (Element) caixas.item(0);
                Integer id = classes.get(nome);
                if (id == null) {
                    id = classes.size();
                    classes.put(nome, id);
                }
                Objeto o = new Objeto();
                o.classe = id;
                o.caixa = true;
                o.pontos = new float[] { num(bb, "xmin"), num(bb, "ymin"), num(bb, "xmax"), num(bb, "ymax") };
                it.objetos.add(o);
            }
        }
        a.itens.addAll(itens.values());
        a.classes.addAll(classes.keySet());
        return a;
    }

    private static String texto(Element e, String tag) {
        NodeList l = e.getElementsByTagName(tag);
        return l.getLength() == 0 ? null : l.item(0).getTextContent().trim();
    }

    private static float num(Element e, String tag) {
        String t = texto(e, tag);
        return t == null ? 0 : Float.parseFloat(t);
    }

    // ------------------------------------------------------------------ importação

    /** Copia as imagens para o projeto e grava as máscaras de anotação. */
    public static int importar(Analise a, Set<Integer> classesPlastico, Projeto projeto, Consumer<String> log)
            throws IOException {
        Set<String> hashes = projeto.hashesExistentes();
        List<String> gravadas = new ArrayList<String>();
        try {
            importarItens(a, classesPlastico, projeto, log, hashes, gravadas);
        } finally {
            // o sistema separa a eficiência das imagens próprias e das importadas
            projeto.registrarImportadas(gravadas);
        }
        return gravadas.size();
    }

    private static void importarItens(Analise a, Set<Integer> classesPlastico, Projeto projeto, Consumer<String> log,
            Set<String> hashes, List<String> gravadas) throws IOException {
        int repetidas = 0, invalidas = 0, n = 0;
        for (Item it : a.itens) {
            n++;
            Projeto.ImagemImportada im = Projeto.preparar(it.imagem.getName(), Files.readAllBytes(it.imagem.toPath()));
            if (im == null) {
                invalidas++;
                continue;
            }
            File destino = projeto.gravarImagemNova(im.nome, im.dados, hashes);
            if (destino == null) {
                repetidas++;
                continue;
            }
            int w = im.largura, h = im.altura;
            projeto.salvarMascara(destino, desenharMascara(it, w, h, classesPlastico, im.escala), w, h);
            gravadas.add(destino.getName());
            if (n % 25 == 0) {
                log.accept(String.format("  %d/%d imagens processadas...", n, a.itens.size()));
            }
        }
        log.accept(String.format("Dataset importado: %d imagens anotadas novas, %d repetidas, %d inválidas.",
                gravadas.size(), repetidas, invalidas));
    }

    /**
     * Caixas viram: miolo = plástico e borda = "ignorar" (a caixa sempre pega um
     * pouco de fundo nos cantos). Polígonos viram plástico exatamente.
     */
    static byte[] desenharMascara(Item it, int w, int h, Set<Integer> plastico, double escalaPx) {
        BufferedImage m = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = m.createGraphics();
        Color ignorar = new Color(128, 128, 128), plast = Color.WHITE;
        for (int fase = 0; fase < 2; fase++) {
            for (Objeto o : it.objetos) {
                if (!plastico.contains(o.classe)) {
                    continue;
                }
                float sx = o.normalizado ? w : (float) escalaPx, sy = o.normalizado ? h : (float) escalaPx;
                if (o.caixa) {
                    float x0 = o.pontos[0] * sx, y0 = o.pontos[1] * sy, x1 = o.pontos[2] * sx, y1 = o.pontos[3] * sy;
                    if (fase == 0) {
                        g.setColor(ignorar);
                        g.fillRect(Math.round(x0), Math.round(y0), Math.round(x1 - x0), Math.round(y1 - y0));
                    } else {
                        float mx = (x1 - x0) * 0.12f, my = (y1 - y0) * 0.12f;
                        g.setColor(plast);
                        g.fillRect(Math.round(x0 + mx), Math.round(y0 + my), Math.round(x1 - x0 - 2 * mx),
                                Math.round(y1 - y0 - 2 * my));
                    }
                } else if (fase == 1) {
                    Polygon pol = new Polygon();
                    for (int i = 0; i + 1 < o.pontos.length; i += 2) {
                        pol.addPoint(Math.round(o.pontos[i] * sx), Math.round(o.pontos[i + 1] * sy));
                    }
                    g.setColor(plast);
                    g.fillPolygon(pol);
                }
            }
        }
        g.dispose();
        int[] px = m.getRGB(0, 0, w, h, null, 0, w);
        byte[] out = new byte[w * h];
        for (int i = 0; i < px.length; i++) {
            int v = (px[i] >> 16) & 255;
            out[i] = v >= 192 ? Projeto.PLASTICO : (v >= 64 ? Projeto.IGNORAR : Projeto.FUNDO);
        }
        return out;
    }
}
