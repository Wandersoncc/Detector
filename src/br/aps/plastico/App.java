package br.aps.plastico;

import java.awt.image.BufferedImage;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.function.Consumer;

import br.aps.plastico.dados.ExportadorYolo;
import br.aps.plastico.dados.ImportadorDataset;
import br.aps.plastico.dados.Projeto;
import br.aps.plastico.gui.JanelaPrincipal;
import br.aps.plastico.ia.Detector;
import br.aps.plastico.ia.Lote;
import br.aps.plastico.ia.Modelo;
import br.aps.plastico.ia.Treinamento;
import br.aps.plastico.visao.Etapas;
import br.aps.plastico.visao.Filtros;

/**
 * Ponto de entrada. Sem argumentos abre a janela; com argumentos funciona pelo
 * terminal (útil para processar muitas imagens):
 *
 * <pre>
 * java -jar DetectorPlastico.jar importar  &lt;arquivos, pastas ou .zip&gt;
 * java -jar DetectorPlastico.jar importar-dataset &lt;pasta YOLO/VOC&gt; [classes de plástico, ex.: 0,1,2]
 * java -jar DetectorPlastico.jar treinar
 * java -jar DetectorPlastico.jar detectar &lt;pasta ou imagem&gt; [pasta de saída]
 * java -jar DetectorPlastico.jar exportar [pasta de saída]
 * java -jar DetectorPlastico.jar filtros &lt;imagem&gt; [pasta de saída]
 * </pre>
 */
public final class App {

    private App() {
    }

    public static void main(String[] args) throws Exception {
        Projeto projeto = new Projeto(Projeto.pastaPadrao());
        if (args.length == 0) {
            JanelaPrincipal.abrir(projeto);
            return;
        }
        Consumer<String> log = new Consumer<String>() {
            @Override
            public void accept(String s) {
                System.out.println(s);
            }
        };
        String cmd = args[0].toLowerCase(Locale.ROOT);
        if (cmd.equals("importar") && args.length > 1) {
            List<File> fontes = new ArrayList<File>();
            for (int i = 1; i < args.length; i++) {
                fontes.add(new File(args[i]));
            }
            projeto.importarImagens(fontes, log);
        } else if (cmd.equals("importar-dataset") && args.length > 1) {
            ImportadorDataset.Analise a = ImportadorDataset.analisar(new File(args[1]));
            System.out.println("Formato: " + a.formato + " | imagens: " + a.totalImagens());
            Set<Integer> plastico = new HashSet<Integer>();
            for (int i = 0; i < a.classes.size(); i++) {
                System.out.println("  classe " + i + ": " + a.classes.get(i) + " (" + a.contagem[i] + " objetos)");
                plastico.add(i);
            }
            if (args.length > 2) {
                plastico.clear();
                for (String s : args[2].split(",")) {
                    plastico.add(Integer.parseInt(s.trim()));
                }
            }
            ImportadorDataset.importar(a, plastico, projeto, log);
        } else if (cmd.equals("testar")) {
            System.exit(Autoteste.rodar(projeto) ? 0 : 1);
        } else if (cmd.equals("treinar")) {
            Treinamento.treinar(projeto, log);
        } else if (cmd.equals("exportar")) {
            File destino = args.length > 1 ? new File(args[1])
                    : new File(projeto.pastaResultados, "dataset_yolo_" + agora());
            ExportadorYolo.Resumo r = ExportadorYolo.exportar(projeto, destino, new ExportadorYolo.Opcoes(), log);
            System.out.println("Pasta: " + r.pasta.getAbsolutePath() + (r.zip != null ? "\nZip: " + r.zip.getAbsolutePath() : ""));
        } else if (cmd.equals("filtros") && args.length > 1) {
            File arq = new File(args[1]);
            BufferedImage img = Projeto.lerImagem(arq);
            double esc = Math.min(1.0, 1024.0 / Math.max(img.getWidth(), img.getHeight()));
            BufferedImage trab = esc < 1 ? Filtros.redimensionar(img, (int) Math.round(img.getWidth() * esc),
                    (int) Math.round(img.getHeight() * esc)) : img;
            List<Etapas.Etapa> etapas = Etapas.calcular(Filtros.paraRgb(trab));
            if (projeto.arquivoModelo().isFile()) {
                Modelo m = Modelo.carregar(projeto.arquivoModelo());
                etapas.add(Lote.etapaIa(etapas.size() + 1, img, Detector.detectar(m, img), m.limiar));
            }
            File saida = args.length > 2 ? new File(args[2])
                    : new File(projeto.pastaResultados, "filtros_" + arq.getName().replaceAll("\\.[^.]+$", "")
                            .replaceAll("[^A-Za-z0-9_-]+", "_").replaceAll("^_+|_+$", ""));
            Etapas.salvar(etapas, arq.getName(), saida);
            for (Etapas.Etapa e : etapas) {
                System.out.println(e.titulo + " – " + e.info);
            }
            System.out.println("Etapas salvas em " + saida.getAbsolutePath());
        } else if (cmd.equals("detectar") && args.length > 1) {
            File modeloArq = projeto.arquivoModelo();
            if (!modeloArq.isFile()) {
                System.out.println("Nenhum modelo treinado. Rode 'treinar' antes.");
                return;
            }
            Modelo m = Modelo.carregar(modeloArq);
            File alvo = new File(args[1]);
            List<File> imgs = new ArrayList<File>();
            if (alvo.isDirectory()) {
                imgs.addAll(Projeto.listarImagensEm(alvo, false));
            } else {
                imgs.add(alvo);
            }
            File saida = args.length > 2 ? new File(args[2]) : new File(projeto.pastaResultados, "lote_" + agora());
            int n = Lote.detectarPasta(m, imgs, saida, m.limiar, log);
            System.out.println(n + " imagens analisadas. Resultado em " + saida.getAbsolutePath());
        } else {
            System.out.println("Uso:\n"
                    + "  java -jar DetectorPlastico.jar                       (abre a janela)\n"
                    + "  java -jar DetectorPlastico.jar importar <arquivos|pastas|.zip>\n"
                    + "  java -jar DetectorPlastico.jar importar-dataset <pasta YOLO/VOC> [0,1,2]\n"
                    + "  java -jar DetectorPlastico.jar testar                         (confere se tudo funciona)\n"
                    + "  java -jar DetectorPlastico.jar treinar\n"
                    + "  java -jar DetectorPlastico.jar detectar <pasta|imagem> [saida]\n"
                    + "  java -jar DetectorPlastico.jar exportar [pasta de saida]        (dataset YOLO)\n"
                    + "  java -jar DetectorPlastico.jar filtros <imagem> [pasta de saida] (etapas do PDI em PNG)");
        }
    }

    private static String agora() {
        return new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
    }
}
