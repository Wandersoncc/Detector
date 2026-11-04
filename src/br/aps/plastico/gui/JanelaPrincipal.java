package br.aps.plastico.gui;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Cursor;
import java.awt.Desktop;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.swing.AbstractAction;
import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.ButtonGroup;
import javax.swing.DefaultListCellRenderer;
import javax.swing.DefaultListModel;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSlider;
import javax.swing.JTextArea;
import javax.swing.JToggleButton;
import javax.swing.JToolBar;
import javax.swing.KeyStroke;
import javax.swing.ListSelectionModel;
import javax.swing.SwingUtilities;
import javax.swing.SwingWorker;
import javax.swing.UIManager;
import javax.swing.filechooser.FileNameExtensionFilter;

import br.aps.plastico.dados.ExportadorYolo;
import br.aps.plastico.dados.ImportadorDataset;
import br.aps.plastico.dados.Projeto;
import br.aps.plastico.ia.Detector;
import br.aps.plastico.ia.Lote;
import br.aps.plastico.ia.Modelo;
import br.aps.plastico.ia.Treinamento;
import br.aps.plastico.visao.Etapas;
import br.aps.plastico.visao.Filtros;

/** Janela principal: lista de imagens, anotação, filtros e a IA. */
public final class JanelaPrincipal extends JFrame implements PainelImagem.Ouvinte {

    private static final String[] VISOES = { "Original", "Filtro HSV clássico", "P&B contornos",
            "IA – mapa de calor", "IA – detecção" };
    private static final String[] FILTROS = { "Todas as imagens", "Anotadas", "Não anotadas",
            "Piores na validação" };

    private final Projeto projeto;
    private Modelo modelo;
    /** Nome da imagem -> {F1, precisão, revocação} da última validação cruzada. */
    private final Map<String, double[]> desempenho = new HashMap<String, double[]>();

    private final DefaultListModel<File> itensLista = new DefaultListModel<File>();
    private final JList<File> lista = new JList<File>(itensLista);
    private final JComboBox<String> comboFiltro = new JComboBox<String>(FILTROS);
    private final JLabel lblContagem = new JLabel(" ");
    private final PainelImagem painel = new PainelImagem(this);
    private final JComboBox<String> comboVisao = new JComboBox<String>(VISOES);
    private final JCheckBox chkAnotacao = new JCheckBox("Mostrar anotação", true);
    private final JSlider sliderPincel = new JSlider(2, 60, 10);
    private final JSlider sliderLimiar = new JSlider(5, 95, 50);
    private final JLabel lblLimiar = new JLabel();
    private final JLabel lblModelo = new JLabel();
    private final JLabel lblStatus = new JLabel(" ");
    private final JTextArea areaLog = new JTextArea(12, 30);
    private final List<JComponent> acoesLongas = new ArrayList<JComponent>();
    private final Map<PainelImagem.Ferramenta, JToggleButton> botoesFerramenta =
            new EnumMap<PainelImagem.Ferramenta, JToggleButton>(PainelImagem.Ferramenta.class);

    private File atual;
    private BufferedImage imagemAtual;
    private Filtros.Rgb rgbTrabalho;
    private double escalaTrabalho = 1;
    private boolean[] bordasTrabalho;
    private BufferedImage cacheHsv, cacheContornos;
    private Detector.Resultado resultadoAtual;
    private boolean sujo, ignorarSelecao;
    private int geracao;

    public static void abrir(final Projeto projeto) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                try {
                    UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
                } catch (Exception e) {
                    // mantém o visual padrão
                }
                new JanelaPrincipal(projeto).setVisible(true);
            }
        });
    }

    private JanelaPrincipal(Projeto projeto) {
        super("Detector de Plástico em Rios – Visão Computacional");
        this.projeto = projeto;
        setDefaultCloseOperation(DO_NOTHING_ON_CLOSE);
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                salvarSeNecessario();
                dispose();
                System.exit(0);
            }
        });
        setLayout(new BorderLayout());
        add(criarBarras(), BorderLayout.NORTH);
        add(criarPainelLista(), BorderLayout.WEST);
        add(painel, BorderLayout.CENTER);
        add(criarPainelIa(), BorderLayout.EAST);
        lblStatus.setBorder(BorderFactory.createEmptyBorder(4, 8, 4, 8));
        add(lblStatus, BorderLayout.SOUTH);
        configurarAtalhos();
        setMinimumSize(new Dimension(1100, 700));
        setSize(1440, 880);
        setLocationRelativeTo(null);
        setExtendedState(MAXIMIZED_BOTH);

        log("Pasta do projeto: " + projeto.raiz.getAbsolutePath());
        carregarModelo();
        carregarDesempenho();
        recarregarLista(null);
        if (itensLista.isEmpty()) {
            log("Nenhuma imagem ainda. Clique em 'Importar imagens / .zip'.");
        }
    }

    // ------------------------------------------------------------------ montagem da tela

    private JComponent criarBarras() {
        JToolBar t1 = new JToolBar();
        t1.setFloatable(false);
        t1.add(new JLabel(" Visualização: "));
        comboVisao.setMaximumSize(new Dimension(190, 28));
        comboVisao.addActionListener(e -> aplicarVisao());
        t1.add(comboVisao);
        chkAnotacao.addActionListener(e -> painel.setMostrarMascara(chkAnotacao.isSelected()));
        t1.add(chkAnotacao);
        t1.addSeparator();
        t1.add(new JLabel(" Anotar: "));
        ButtonGroup grupo = new ButtonGroup();
        Object[][] fs = { { PainelImagem.Ferramenta.PINCEL, "Plástico (B)", "Pinta o que é plástico (botão direito apaga)" },
                { PainelImagem.Ferramenta.IGNORAR, "Ignorar (I)", "Marca áreas duvidosas que a IA não deve usar" },
                { PainelImagem.Ferramenta.BORRACHA, "Borracha (E)", "Apaga a anotação" },
                { PainelImagem.Ferramenta.RETANGULO, "Retângulo (R)", "Arraste para preencher (botão direito apaga)" },
                { PainelImagem.Ferramenta.VARINHA, "Varinha (W)",
                        "Clique num objeto: preenche a área dentro do contorno P&B (Shift/direito apaga)" },
                { PainelImagem.Ferramenta.MOVER, "Mover (H)", "Arraste para mover; roda do mouse = zoom; duplo clique ajusta" } };
        for (Object[] f : fs) {
            final PainelImagem.Ferramenta fer = (PainelImagem.Ferramenta) f[0];
            JToggleButton b = new JToggleButton((String) f[1]);
            b.setToolTipText((String) f[2]);
            b.setFocusable(false);
            b.addActionListener(e -> painel.setFerramenta(fer));
            grupo.add(b);
            t1.add(b);
            botoesFerramenta.put(fer, b);
        }
        botoesFerramenta.get(PainelImagem.Ferramenta.PINCEL).setSelected(true);
        t1.addSeparator();
        t1.add(new JLabel(" Pincel: "));
        sliderPincel.setMaximumSize(new Dimension(140, 28));
        sliderPincel.setFocusable(false);
        sliderPincel.setToolTipText("Tamanho do pincel ( [ e ] )");
        sliderPincel.addChangeListener(e -> painel.setRaio(sliderPincel.getValue()));
        t1.add(sliderPincel);

        JToolBar t2 = new JToolBar();
        t2.setFloatable(false);
        t2.add(botao("Salvar anotação (Ctrl+S)", "Grava a anotação desta imagem. Salvar sem pintar nada = imagem sem plástico.",
                e -> salvarAnotacao(), false));
        t2.add(botao("Desfazer (Ctrl+Z)", null, e -> {
            if (!painel.desfazer()) {
                status("Nada para desfazer.");
            }
        }, false));
        t2.add(botao("Limpar", "Apaga toda a anotação desta imagem", e -> limparAnotacao(), false));
        t2.add(botao("Remover anotação", "A imagem volta a ser 'não anotada' (não entra no treino)",
                e -> removerAnotacao(), false));
        t2.addSeparator();
        t2.add(botao("Pré-anotar com a IA", "A IA pinta o que acha que é plástico; você só corrige e salva",
                e -> preAnotar(), true));
        t2.addSeparator();
        t2.add(botao("Anterior (PgUp)", null, e -> navegar(-1), false));
        t2.add(botao("Próxima (PgDn)", null, e -> navegar(1), false));
        t2.addSeparator();
        t2.add(botao("Filtros passo a passo (PDI)",
                "Mostra cada etapa do processamento nesta imagem e salva as figuras para o trabalho escrito",
                e -> filtrosPassoAPasso(), true));
        t2.add(Box.createHorizontalGlue());
        t2.add(botao("Como usar", null, e -> ajuda(), false));

        JPanel p = new JPanel(new GridLayout(2, 1));
        p.add(t1);
        p.add(t2);
        return p;
    }

    private JButton botao(String texto, String dica, java.awt.event.ActionListener acao, boolean longa) {
        JButton b = new JButton(texto);
        b.setToolTipText(dica);
        b.setFocusable(false);
        b.addActionListener(acao);
        if (longa) {
            acoesLongas.add(b);
        }
        return b;
    }

    private JComponent criarPainelLista() {
        JPanel p = new JPanel(new BorderLayout(4, 4));
        p.setBorder(BorderFactory.createCompoundBorder(BorderFactory.createEmptyBorder(6, 6, 6, 0),
                BorderFactory.createTitledBorder("Imagens")));
        p.setPreferredSize(new Dimension(270, 100));
        comboFiltro.addActionListener(e -> recarregarLista(null));
        p.add(comboFiltro, BorderLayout.NORTH);

        lista.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        // remove a busca por letra da lista para os atalhos de teclado funcionarem
        for (KeyListener kl : lista.getKeyListeners()) {
            lista.removeKeyListener(kl);
        }
        lista.setCellRenderer(new DefaultListCellRenderer() {
            @Override
            public Component getListCellRendererComponent(JList<?> l, Object v, int i, boolean sel, boolean foco) {
                super.getListCellRendererComponent(l, v, i, sel, foco);
                File f = (File) v;
                boolean anotada = projeto.anotada(f);
                double[] d = desempenho.get(f.getName());
                StringBuilder sb = new StringBuilder("<html>");
                sb.append(anotada ? "<font color='#1a9c3a'>&#9679;</font> " : "<font color='#999999'>&#9675;</font> ");
                sb.append(f.getName().replace("&", "&amp;").replace("<", "&lt;"));
                if (d != null) {
                    String cor = d[0] < 0.5 ? "#c0392b" : (d[0] < 0.7 ? "#d68910" : "#1a9c3a");
                    sb.append(String.format(Treinamento.PT, " <font color='%s'>F1 %.0f%%</font>", cor, d[0] * 100));
                }
                setText(sb.append("</html>").toString());
                return this;
            }
        });
        lista.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting() && !ignorarSelecao) {
                abrirImagem(lista.getSelectedValue());
            }
        });
        p.add(new JScrollPane(lista), BorderLayout.CENTER);

        JPanel sul = new JPanel(new GridLayout(0, 1, 2, 2));
        sul.add(lblContagem);
        sul.add(botao("Importar imagens / .zip...", "Copia imagens (ou um .zip) para o projeto", e -> importarImagens(), true));
        sul.add(botao("Importar dataset rotulado...", "Pasta de dataset no formato YOLO ou Pascal VOC (Roboflow, Kaggle...)",
                e -> importarDataset(), true));
        sul.add(botao("Exportar dataset (YOLO)...",
                "Gera o dataset preparado (imagens + rótulos YOLO + máscaras) para treinar outros modelos",
                e -> exportarDataset(), true));
        p.add(sul, BorderLayout.SOUTH);
        return p;
    }

    private JComponent criarPainelIa() {
        JPanel p = new JPanel(new BorderLayout(4, 6));
        p.setBorder(BorderFactory.createCompoundBorder(BorderFactory.createEmptyBorder(6, 0, 6, 6),
                BorderFactory.createTitledBorder("Inteligência artificial")));
        p.setPreferredSize(new Dimension(330, 100));

        JPanel topo = new JPanel();
        topo.setLayout(new BoxLayout(topo, BoxLayout.Y_AXIS));
        lblModelo.setAlignmentX(0f);
        lblModelo.setBorder(BorderFactory.createEmptyBorder(4, 4, 8, 4));
        topo.add(lblModelo);

        lblLimiar.setAlignmentX(0f);
        topo.add(lblLimiar);
        sliderLimiar.setAlignmentX(0f);
        sliderLimiar.setFocusable(false);
        sliderLimiar.setToolTipText("Abaixe para achar mais plástico (mais alarmes falsos); suba para ser mais rigoroso.");
        sliderLimiar.addChangeListener(e -> {
            lblLimiar.setText(String.format("Limiar de confiança: %d%%", sliderLimiar.getValue()));
            painel.setLimiar(limiar());
            atualizarStatus();
        });
        lblLimiar.setText(String.format("Limiar de confiança: %d%%", sliderLimiar.getValue()));
        topo.add(sliderLimiar);
        topo.add(Box.createVerticalStrut(6));

        JPanel botoes = new JPanel(new GridLayout(0, 1, 4, 4));
        botoes.setAlignmentX(0f);
        JButton treinar = botao("Treinar IA com as anotações", "Treina e mede a eficiência (validação cruzada)",
                e -> treinar(), true);
        treinar.setFont(treinar.getFont().deriveFont(Font.BOLD, treinar.getFont().getSize2D() + 1f));
        botoes.add(treinar);
        botoes.add(botao("Relatório e evolução da eficiência", null, e -> relatorio(), false));
        botoes.add(botao("Sugerir próximas imagens p/ anotar", "Imagens em que a IA mais tem dúvida",
                e -> sugerir(), true));
        botoes.add(botao("Detectar em lote (pasta)...", "Analisa todas as imagens de uma pasta e gera relatório",
                e -> detectarLote(), true));
        botoes.add(botao("Abrir pasta do projeto", null, e -> abrirPasta(projeto.raiz), false));
        topo.add(botoes);
        p.add(topo, BorderLayout.NORTH);

        areaLog.setEditable(false);
        areaLog.setLineWrap(true);
        areaLog.setWrapStyleWord(true);
        areaLog.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));
        JScrollPane sp = new JScrollPane(areaLog);
        sp.setBorder(BorderFactory.createTitledBorder("Registro"));
        p.add(sp, BorderLayout.CENTER);
        return p;
    }

    private void configurarAtalhos() {
        JComponent raiz = getRootPane();
        atalho(raiz, "ctrl S", "salvar", () -> salvarAnotacao());
        atalho(raiz, "ctrl Z", "desfazer", () -> painel.desfazer());
        atalho(raiz, "B", "pincel", () -> escolher(PainelImagem.Ferramenta.PINCEL));
        atalho(raiz, "I", "ignorar", () -> escolher(PainelImagem.Ferramenta.IGNORAR));
        atalho(raiz, "E", "borracha", () -> escolher(PainelImagem.Ferramenta.BORRACHA));
        atalho(raiz, "R", "retangulo", () -> escolher(PainelImagem.Ferramenta.RETANGULO));
        atalho(raiz, "W", "varinha", () -> escolher(PainelImagem.Ferramenta.VARINHA));
        atalho(raiz, "H", "mover", () -> escolher(PainelImagem.Ferramenta.MOVER));
        atalho(raiz, "PAGE_DOWN", "proxima", () -> navegar(1));
        atalho(raiz, "PAGE_UP", "anterior", () -> navegar(-1));
        atalho(raiz, "OPEN_BRACKET", "menor", () -> sliderPincel.setValue(sliderPincel.getValue() - 2));
        atalho(raiz, "CLOSE_BRACKET", "maior", () -> sliderPincel.setValue(sliderPincel.getValue() + 2));
    }

    private void atalho(JComponent c, String tecla, String nome, final Runnable r) {
        c.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW).put(KeyStroke.getKeyStroke(tecla), nome);
        c.getActionMap().put(nome, new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                r.run();
            }
        });
    }

    private void escolher(PainelImagem.Ferramenta f) {
        botoesFerramenta.get(f).setSelected(true);
        painel.setFerramenta(f);
    }

    // ------------------------------------------------------------------ lista de imagens

    private void recarregarLista(File selecionar) {
        List<File> todas = projeto.listarImagens();
        int filtro = comboFiltro.getSelectedIndex();
        List<File> vis = new ArrayList<File>();
        for (File f : todas) {
            boolean an = projeto.anotada(f);
            if ((filtro == 1 && !an) || (filtro == 2 && an) || (filtro == 3 && !desempenho.containsKey(f.getName()))) {
                continue;
            }
            vis.add(f);
        }
        if (filtro == 3) {
            Collections.sort(vis, new Comparator<File>() {
                @Override
                public int compare(File a, File b) {
                    return Double.compare(desempenho.get(a.getName())[0], desempenho.get(b.getName())[0]);
                }
            });
        }
        ignorarSelecao = true;
        itensLista.clear();
        for (File f : vis) {
            itensLista.addElement(f);
        }
        ignorarSelecao = false;
        File alvo = selecionar != null ? selecionar : atual;
        if (alvo != null && vis.contains(alvo)) {
            lista.setSelectedValue(alvo, true);
        } else if (!vis.isEmpty() && atual == null) {
            lista.setSelectedIndex(0);
        }
        atualizarContagem(todas);
    }

    private void atualizarContagem(List<File> todas) {
        int an = 0;
        for (File f : todas) {
            if (projeto.anotada(f)) {
                an++;
            }
        }
        lblContagem.setText(String.format("%d imagens · %d anotadas", todas.size(), an));
    }

    private void navegar(int d) {
        int n = itensLista.size();
        if (n == 0) {
            return;
        }
        int i = lista.getSelectedIndex();
        int j = Math.max(0, Math.min(n - 1, i < 0 ? 0 : i + d));
        lista.setSelectedIndex(j);
        lista.ensureIndexIsVisible(j);
    }

    private void abrirPorNome(String nome) {
        File f = new File(projeto.pastaImagens, nome);
        if (!f.isFile()) {
            return;
        }
        if (!itensLista.contains(f)) {
            comboFiltro.setSelectedIndex(0);
        }
        recarregarLista(f);
        lista.setSelectedValue(f, true);
        toFront();
    }

    // ------------------------------------------------------------------ imagem atual

    private void abrirImagem(File f) {
        if (f == null || f.equals(atual)) {
            return;
        }
        salvarSeNecessario();
        BufferedImage img;
        byte[] m;
        try {
            img = Projeto.lerImagem(f);
            m = projeto.carregarMascara(f, img.getWidth(), img.getHeight());
        } catch (IOException e) {
            erro("Não foi possível abrir " + f.getName(), e);
            return;
        }
        int w = img.getWidth(), h = img.getHeight();
        if (m == null) {
            m = new byte[w * h];
        }
        atual = f;
        imagemAtual = img;
        painel.setImagem(img, m);
        double esc = Math.min(1.0, 1024.0 / Math.max(w, h));
        BufferedImage trab = img;
        if (esc < 1) {
            trab = Filtros.redimensionar(img, Math.max(1, (int) Math.round(w * esc)), Math.max(1, (int) Math.round(h * esc)));
        }
        rgbTrabalho = Filtros.paraRgb(trab);
        escalaTrabalho = trab.getWidth() / (double) w;
        bordasTrabalho = null;
        cacheHsv = null;
        cacheContornos = null;
        resultadoAtual = null;
        sujo = false;
        aplicarVisao();
        detectarAtual();
        atualizarStatus();
    }

    private void aplicarVisao() {
        int v = comboVisao.getSelectedIndex();
        if (imagemAtual != null) {
            if (v == 1) {
                if (cacheHsv == null) {
                    cacheHsv = visualHsv();
                }
                painel.setFundo(cacheHsv);
            } else if (v == 2) {
                if (cacheContornos == null) {
                    cacheContornos = visualContornos();
                }
                painel.setFundo(cacheContornos);
            } else {
                painel.setFundo(null);
            }
        }
        painel.setModoIa(v == 3, v == 4);
        if ((v == 3 || v == 4) && modelo == null) {
            status("Ainda não há modelo treinado: anote algumas imagens e clique em 'Treinar IA'.");
        }
    }

    /** Mostra só o que passa no filtro HSV fixo (o resto fica escuro). */
    private BufferedImage visualHsv() {
        boolean[] m = Filtros.filtroHsvClassico(rgbTrabalho);
        int w = rgbTrabalho.w, h = rgbTrabalho.h;
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < m.length; i++) {
            float f = m[i] ? 1f : 0.18f;
            int r = (int) (rgbTrabalho.r[i] * 255 * f), g = (int) (rgbTrabalho.g[i] * 255 * f),
                    b = (int) (rgbTrabalho.b[i] * 255 * f);
            out.setRGB(i % w, i / w, (r << 16) | (g << 8) | b);
        }
        return out;
    }

    /** Filtro preto e branco: contornos brancos sobre fundo preto. */
    private BufferedImage visualContornos() {
        if (bordasTrabalho == null) {
            bordasTrabalho = Filtros.bordas(rgbTrabalho);
        }
        int w = rgbTrabalho.w, h = rgbTrabalho.h;
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < bordasTrabalho.length; i++) {
            out.setRGB(i % w, i / w, bordasTrabalho[i] ? 0xFFFFFF : 0);
        }
        return out;
    }

    private void detectarAtual() {
        if (modelo == null || imagemAtual == null) {
            painel.setResultado(null);
            return;
        }
        final int ger = ++geracao;
        final BufferedImage img = imagemAtual;
        final Modelo mod = modelo;
        new SwingWorker<Detector.Resultado, Void>() {
            @Override
            protected Detector.Resultado doInBackground() {
                return Detector.detectar(mod, img);
            }

            @Override
            protected void done() {
                if (ger != geracao) {
                    return;
                }
                try {
                    resultadoAtual = get();
                    painel.setResultado(resultadoAtual);
                    atualizarStatus();
                } catch (Exception e) {
                    log("Erro na detecção: " + e.getMessage());
                }
            }
        }.execute();
    }

    @Override
    public void mascaraAlterada() {
        sujo = true;
        atualizarStatus();
    }

    @Override
    public void varinha(int x, int y, boolean apagar) {
        if (rgbTrabalho == null) {
            return;
        }
        if (bordasTrabalho == null) {
            bordasTrabalho = Filtros.bordas(rgbTrabalho);
        }
        int tx = Math.min(rgbTrabalho.w - 1, (int) (x * escalaTrabalho));
        int ty = Math.min(rgbTrabalho.h - 1, (int) (y * escalaTrabalho));
        boolean[] reg = Filtros.varinha(rgbTrabalho, bordasTrabalho, tx, ty, 0.2f,
                (int) (rgbTrabalho.w * (long) rgbTrabalho.h / 4));
        if (reg == null) {
            status("A varinha vazou (o contorno não está fechado nesse ponto). Use o pincel ou o retângulo.");
            return;
        }
        painel.aplicarRegiao(reg, rgbTrabalho.w, rgbTrabalho.h, escalaTrabalho, apagar ? Projeto.FUNDO : Projeto.PLASTICO);
    }

    private void salvarAnotacao() {
        if (atual == null || imagemAtual == null) {
            return;
        }
        try {
            projeto.salvarMascara(atual, painel.getMascara(), imagemAtual.getWidth(), imagemAtual.getHeight());
            sujo = false;
            lista.repaint();
            atualizarContagem(projeto.listarImagens());
            status("Anotação salva: " + atual.getName()
                    + (painel.mascaraVazia() ? " (marcada como imagem SEM plástico)" : ""));
        } catch (IOException e) {
            erro("Não foi possível salvar a anotação", e);
        }
    }

    private void salvarSeNecessario() {
        if (sujo) {
            salvarAnotacao();
        }
    }

    private void limparAnotacao() {
        if (atual == null) {
            return;
        }
        painel.empilharDesfazer();
        Arrays.fill(painel.getMascara(), Projeto.FUNDO);
        painel.mascaraSubstituida();
    }

    private void removerAnotacao() {
        if (atual == null || !projeto.anotada(atual)) {
            return;
        }
        if (JOptionPane.showConfirmDialog(this, "Remover a anotação de " + atual.getName()
                + "?\nEla deixa de ser usada no treino.", "Remover anotação", JOptionPane.YES_NO_OPTION) != 0) {
            return;
        }
        projeto.removerMascara(atual);
        Arrays.fill(painel.getMascara(), Projeto.FUNDO);
        painel.mascaraSubstituida();
        sujo = false;
        lista.repaint();
        atualizarContagem(projeto.listarImagens());
        atualizarStatus();
    }

    private void preAnotar() {
        if (modelo == null) {
            aviso("Treine a IA primeiro (ela precisa de algumas imagens anotadas).");
            return;
        }
        if (resultadoAtual == null) {
            aviso("Aguarde a IA terminar de analisar esta imagem.");
            return;
        }
        if (!painel.mascaraVazia() && JOptionPane.showConfirmDialog(this,
                "Substituir a anotação atual pela sugestão da IA?", "Pré-anotar", JOptionPane.YES_NO_OPTION) != 0) {
            return;
        }
        painel.empilharDesfazer();
        byte[] m = painel.getMascara();
        int w = imagemAtual.getWidth(), h = imagemAtual.getHeight();
        float lim = limiar();
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                m[y * w + x] = resultadoAtual.probEm(x + 0.5, y + 0.5) >= lim ? Projeto.PLASTICO : Projeto.FUNDO;
            }
        }
        chkAnotacao.setSelected(true);
        painel.setMostrarMascara(true);
        painel.mascaraSubstituida();
        status("Pré-anotação aplicada. Corrija com pincel/borracha e salve (Ctrl+S). Ela só vale depois de salva.");
    }

    // ------------------------------------------------------------------ IA

    private float limiar() {
        return sliderLimiar.getValue() / 100f;
    }

    private void carregarModelo() {
        modelo = null;
        File f = projeto.arquivoModelo();
        if (f.isFile()) {
            try {
                modelo = Modelo.carregar(f);
                sliderLimiar.setValue(Math.round(modelo.limiar * 100));
            } catch (IOException e) {
                log("Não foi possível ler o modelo: " + e.getMessage());
            }
        }
        atualizarRotuloModelo();
    }

    private void atualizarRotuloModelo() {
        if (modelo == null) {
            lblModelo.setText("<html><b>Nenhum modelo treinado.</b><br>1. Anote imagens (pinte o plástico).<br>"
                    + "2. Clique em <i>Treinar IA</i>.</html>");
            return;
        }
        StringBuilder sb = new StringBuilder("<html>");
        sb.append("<b>Modelo treinado</b> em ")
                .append(new SimpleDateFormat("dd/MM/yyyy HH:mm").format(new Date(modelo.dataTreino)))
                .append("<br>Imagens anotadas usadas: ").append(modelo.nImagens);
        if (modelo.f1 >= 0) {
            sb.append(String.format(Treinamento.PT, "<br><br><font size='+1'><b>Eficiência (F1): %.1f%%</b></font>"
                    + "<br>Precisão %.1f%% · Revocação %.1f%%"
                    + "<br><font color='gray'>Filtro HSV clássico: %.1f%%</font>",
                    modelo.f1 * 100, modelo.precisao * 100, modelo.revocacao * 100, modelo.f1Classico * 100));
            if (modelo.f1Proprias >= 0 && modelo.f1Importadas >= 0) {
                sb.append(String.format(Treinamento.PT, "<br>Suas imagens: <b>%.1f%%</b> · Datasets importados: <b>%.1f%%</b>",
                        modelo.f1Proprias * 100, modelo.f1Importadas * 100));
            }
        } else {
            sb.append("<br><font color='gray'>Eficiência não medida (anote 4+ imagens).</font>");
        }
        lblModelo.setText(sb.append("</html>").toString());
    }

    private void carregarDesempenho() {
        desempenho.clear();
        for (String[] l : Projeto.lerCsv(projeto.arquivoDesempenho())) {
            if (l.length >= 4) {
                desempenho.put(l[0], new double[] { Projeto.numero(l[1]) / 100, Projeto.numero(l[2]) / 100,
                        Projeto.numero(l[3]) / 100 });
            }
        }
    }

    private void treinar() {
        salvarSeNecessario();
        if (projeto.imagensAnotadas().isEmpty()) {
            aviso("Nenhuma imagem anotada ainda.\n\nEscolha uma imagem, pinte o plástico com o pincel e salve (Ctrl+S).");
            return;
        }
        bloquear(true);
        log("=== Treinamento iniciado ===");
        new SwingWorker<Treinamento.Resultado, String>() {
            @Override
            protected Treinamento.Resultado doInBackground() throws Exception {
                return Treinamento.treinar(projeto, s -> publish(s));
            }

            @Override
            protected void process(List<String> msgs) {
                for (String s : msgs) {
                    log(s);
                }
            }

            @Override
            protected void done() {
                bloquear(false);
                try {
                    Treinamento.Resultado r = get();
                    carregarModelo();
                    carregarDesempenho();
                    lista.repaint();
                    detectarAtual();
                    mostrarResumo(r);
                } catch (Exception e) {
                    erro("Falha no treinamento", e);
                }
            }
        }.execute();
    }

    private void mostrarResumo(Treinamento.Resultado r) {
        StringBuilder sb = new StringBuilder("<html><body style='width:380px'>");
        sb.append(String.format(Treinamento.PT, "<h3>Treino concluído em %.0f s</h3>", r.milissegundos / 1000.0));
        if (r.validado) {
            sb.append(String.format(Treinamento.PT,
                    "<p><b>Eficiência da IA (F1): %.1f%%</b><br>Precisão: %.1f%% · Revocação: %.1f%% · IoU: %.1f%%</p>"
                            + "<p>Filtro HSV clássico (sem aprendizado) nas mesmas imagens: %.1f%%</p>",
                    r.ia.f1() * 100, r.ia.precisao() * 100, r.ia.revocacao() * 100, r.ia.iou() * 100,
                    r.classico.f1() * 100));
            if (r.proprias.total() > 0 && r.importadas.total() > 0) {
                sb.append(String.format(Treinamento.PT, "<p>Por origem: <b>suas imagens %.1f%%</b> · datasets importados "
                        + "%.1f%%</p>", r.proprias.f1() * 100, r.importadas.f1() * 100));
            }
            sb.append("<p>Imagens em que a IA mais errou (vale revisar/anotar parecidas):<ul>");
            for (int i = 0; i < Math.min(5, r.porImagem.size()); i++) {
                Object[] o = r.porImagem.get(i);
                sb.append(String.format(Treinamento.PT, "<li>%s – F1 %.0f%%</li>", o[0],
                        ((br.aps.plastico.ia.Metricas) o[1]).f1() * 100));
            }
            sb.append("</ul></p>");
        } else {
            sb.append("<p>Modelo treinado com ").append(r.imagens)
                    .append(" imagem(ns). Anote pelo menos 4 para o sistema medir a eficiência.</p>");
        }
        JOptionPane.showMessageDialog(this, sb.append("</body></html>").toString(), "Treinamento",
                JOptionPane.INFORMATION_MESSAGE);
    }

    private void relatorio() {
        new DialogoRelatorio(this, projeto, nome -> abrirPorNome(nome)).setVisible(true);
    }

    private void sugerir() {
        if (modelo == null) {
            aviso("Treine a IA primeiro.");
            return;
        }
        final List<File> nao = new ArrayList<File>();
        for (File f : projeto.listarImagens()) {
            if (!projeto.anotada(f)) {
                nao.add(f);
            }
        }
        if (nao.isEmpty()) {
            aviso("Todas as imagens já estão anotadas. Importe mais imagens para continuar melhorando a IA.");
            return;
        }
        bloquear(true);
        log("Analisando " + nao.size() + " imagens não anotadas...");
        new SwingWorker<List<Lote.Sugestao>, String>() {
            @Override
            protected List<Lote.Sugestao> doInBackground() throws Exception {
                return Lote.rankearIncerteza(modelo, nao, s -> publish(s));
            }

            @Override
            protected void process(List<String> msgs) {
                for (String s : msgs) {
                    log(s);
                }
            }

            @Override
            protected void done() {
                bloquear(false);
                try {
                    mostrarSugestoes(get());
                } catch (Exception e) {
                    erro("Falha ao analisar as imagens", e);
                }
            }
        }.execute();
    }

    private void mostrarSugestoes(List<Lote.Sugestao> s) {
        DefaultListModel<String> m = new DefaultListModel<String>();
        for (Lote.Sugestao x : s) {
            m.addElement(String.format(Treinamento.PT, "%s   —   dúvida em %.0f%% · plástico estimado %.1f%%",
                    x.arquivo.getName(), x.incerteza * 100, x.area));
        }
        final JList<String> l = new JList<String>(m);
        l.setVisibleRowCount(14);
        final List<Lote.Sugestao> ss = s;
        l.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (e.getClickCount() == 2 && l.getSelectedIndex() >= 0) {
                    abrirPorNome(ss.get(l.getSelectedIndex()).arquivo.getName());
                }
            }
        });
        JPanel p = new JPanel(new BorderLayout(4, 4));
        p.add(new JLabel("<html>Anotar primeiro as imagens em que a IA tem mais dúvida é o jeito mais rápido de melhorar."
                + "<br>Dê dois cliques para abrir.</html>"), BorderLayout.NORTH);
        p.add(new JScrollPane(l), BorderLayout.CENTER);
        p.setPreferredSize(new Dimension(560, 320));
        JOptionPane.showMessageDialog(this, p, "Sugestões para anotar", JOptionPane.PLAIN_MESSAGE);
    }

    private void detectarLote() {
        if (modelo == null) {
            aviso("Treine a IA primeiro.");
            return;
        }
        JFileChooser ch = new JFileChooser(projeto.raiz);
        ch.setDialogTitle("Escolha a pasta com as imagens para analisar");
        ch.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        if (ch.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) {
            return;
        }
        final List<File> imgs = Projeto.listarImagensEm(ch.getSelectedFile(), false);
        if (imgs.isEmpty()) {
            aviso("Nenhuma imagem JPG/PNG/BMP nessa pasta.");
            return;
        }
        final File saida = new File(projeto.pastaResultados,
                "lote_" + new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date()));
        final float lim = limiar();
        bloquear(true);
        log("Detecção em lote: " + imgs.size() + " imagens...");
        new SwingWorker<Integer, String>() {
            @Override
            protected Integer doInBackground() throws Exception {
                return Lote.detectarPasta(modelo, imgs, saida, lim, s -> publish(s));
            }

            @Override
            protected void process(List<String> msgs) {
                for (String s : msgs) {
                    log(s);
                }
            }

            @Override
            protected void done() {
                bloquear(false);
                try {
                    int n = get();
                    log("Resultado gravado em " + saida.getAbsolutePath());
                    if (JOptionPane.showConfirmDialog(JanelaPrincipal.this, n + " imagens analisadas.\n"
                            + "As imagens marcadas e o relatorio.csv estão em:\n" + saida.getAbsolutePath()
                            + "\n\nAbrir a pasta?", "Detecção em lote", JOptionPane.YES_NO_OPTION) == 0) {
                        abrirPasta(saida);
                    }
                } catch (Exception e) {
                    erro("Falha na detecção em lote", e);
                }
            }
        }.execute();
    }

    private void filtrosPassoAPasso() {
        if (imagemAtual == null) {
            aviso("Escolha uma imagem na lista primeiro.");
            return;
        }
        final Filtros.Rgb rgb = rgbTrabalho;
        final String nome = atual.getName();
        final BufferedImage original = imagemAtual;
        final Detector.Resultado res = resultadoAtual;
        final float lim = limiar();
        bloquear(true);
        new SwingWorker<List<Etapas.Etapa>, Void>() {
            @Override
            protected List<Etapas.Etapa> doInBackground() {
                List<Etapas.Etapa> etapas = Etapas.calcular(rgb);
                if (res != null) {
                    etapas.add(Lote.etapaIa(etapas.size() + 1, original, res, lim));
                }
                return etapas;
            }

            @Override
            protected void done() {
                bloquear(false);
                try {
                    new DialogoFiltros(JanelaPrincipal.this, nome, get(), projeto.pastaResultados).setVisible(true);
                } catch (Exception e) {
                    erro("Falha ao aplicar os filtros", e);
                }
            }
        }.execute();
    }

    private void exportarDataset() {
        salvarSeNecessario();
        int anotadas = projeto.imagensAnotadas().size();
        if (anotadas == 0) {
            aviso("Nenhuma imagem anotada para exportar.");
            return;
        }
        int deDatasets = 0;
        Set<String> importadas = projeto.imagensDeDatasets();
        for (File f : projeto.imagensAnotadas()) {
            if (importadas.contains(f.getName())) {
                deDatasets++;
            }
        }
        JCheckBox c640 = new JCheckBox("Reduzir para 640 px (tamanho padrão do YOLO)", true);
        JCheckBox cMasc = new JCheckBox("Incluir máscaras PNG (para modelos de segmentação)", true);
        JCheckBox cImp = new JCheckBox("Incluir as " + deDatasets + " imagens vindas de datasets públicos", deDatasets > 0);
        cImp.setEnabled(deDatasets > 0);
        JCheckBox cZip = new JCheckBox("Gerar também um .zip (para enviar ao Google Colab)", true);
        JPanel p = new JPanel(new GridLayout(0, 1, 2, 2));
        p.add(new JLabel("<html>Exporta as <b>" + anotadas + " imagens anotadas</b> no formato YOLO (Ultralytics),"
                + "<br>com 80% para treino e 20% para validação. As caixas saem das suas máscaras.</html>"));
        p.add(c640);
        p.add(cMasc);
        p.add(cImp);
        p.add(cZip);
        if (JOptionPane.showConfirmDialog(this, p, "Exportar dataset preparado", JOptionPane.OK_CANCEL_OPTION,
                JOptionPane.PLAIN_MESSAGE) != JOptionPane.OK_OPTION) {
            return;
        }
        final ExportadorYolo.Opcoes op = new ExportadorYolo.Opcoes();
        op.reduzir640 = c640.isSelected();
        op.mascaras = cMasc.isSelected();
        op.incluirImportadas = cImp.isSelected();
        op.gerarZip = cZip.isSelected();
        final File destino = new File(projeto.pastaResultados,
                "dataset_yolo_" + new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date()));
        bloquear(true);
        log("Exportando o dataset...");
        new SwingWorker<ExportadorYolo.Resumo, String>() {
            @Override
            protected ExportadorYolo.Resumo doInBackground() throws Exception {
                return ExportadorYolo.exportar(projeto, destino, op, s -> publish(s));
            }

            @Override
            protected void process(List<String> msgs) {
                for (String s : msgs) {
                    log(s);
                }
            }

            @Override
            protected void done() {
                bloquear(false);
                try {
                    ExportadorYolo.Resumo r = get();
                    String msg = String.format("Dataset exportado: %d imagens (treino %d, validação %d), %d objetos.%n%n"
                            + "Pasta: %s%s%n%nO arquivo LEIAME.txt dentro dela explica como treinar um YOLO.%n%n"
                            + "Abrir a pasta?", r.treino + r.validacao, r.treino, r.validacao, r.objetos,
                            r.pasta.getAbsolutePath(), r.zip != null ? "\nZip: " + r.zip.getName() : "");
                    if (JOptionPane.showConfirmDialog(JanelaPrincipal.this, msg, "Exportar dataset",
                            JOptionPane.YES_NO_OPTION) == 0) {
                        abrirPasta(r.pasta);
                    }
                } catch (Exception e) {
                    erro("Falha ao exportar o dataset", e);
                }
            }
        }.execute();
    }

    // ------------------------------------------------------------------ importação

    private void importarImagens() {
        JFileChooser ch = new JFileChooser();
        ch.setDialogTitle("Escolha imagens, pastas ou arquivos .zip");
        ch.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
        ch.setMultiSelectionEnabled(true);
        ch.setFileFilter(new FileNameExtensionFilter("Imagens e .zip", "jpg", "jpeg", "png", "bmp", "gif", "zip"));
        if (ch.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) {
            return;
        }
        final List<File> fontes = Arrays.asList(ch.getSelectedFiles());
        bloquear(true);
        new SwingWorker<Integer, String>() {
            @Override
            protected Integer doInBackground() throws Exception {
                return projeto.importarImagens(fontes, s -> publish(s));
            }

            @Override
            protected void process(List<String> msgs) {
                for (String s : msgs) {
                    log(s);
                }
            }

            @Override
            protected void done() {
                bloquear(false);
                try {
                    get();
                    recarregarLista(null);
                } catch (Exception e) {
                    erro("Falha na importação", e);
                }
            }
        }.execute();
    }

    private void importarDataset() {
        JFileChooser ch = new JFileChooser();
        ch.setDialogTitle("Escolha a pasta do dataset (YOLO ou Pascal VOC)");
        ch.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        if (ch.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) {
            return;
        }
        final ImportadorDataset.Analise a;
        try {
            setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
            a = ImportadorDataset.analisar(ch.getSelectedFile());
        } catch (Exception e) {
            erro("Não foi possível ler o dataset", e);
            return;
        } finally {
            setCursor(Cursor.getDefaultCursor());
        }
        JPanel p = new JPanel();
        p.setLayout(new BoxLayout(p, BoxLayout.Y_AXIS));
        p.add(new JLabel(String.format("<html>Formato: <b>%s</b> · %d imagens (%d com rótulos).<br><br>"
                + "Marque as classes que são <b>plástico</b>. As outras serão tratadas como fundo:</html>",
                a.formato, a.totalImagens(), a.imagensComRotulo)));
        List<JCheckBox> caixas = new ArrayList<JCheckBox>();
        for (int i = 0; i < a.classes.size(); i++) {
            String nome = a.classes.get(i);
            String n = nome.toLowerCase();
            boolean sugerido = !(n.contains("not") || n.contains("nao") || n.contains("não") || n.contains("metal")
                    || n.contains("glass") || n.contains("vidro") || n.contains("paper") || n.contains("papel")
                    || n.contains("wood") || n.contains("madeira") || n.contains("organic"));
            JCheckBox c = new JCheckBox(String.format("%d: %s  (%d objetos)", i, nome, a.contagem[i]), sugerido);
            caixas.add(c);
            p.add(c);
        }
        JScrollPane sp = new JScrollPane(p);
        sp.setPreferredSize(new Dimension(460, Math.min(420, 110 + 28 * a.classes.size())));
        if (JOptionPane.showConfirmDialog(this, sp, "Importar dataset rotulado", JOptionPane.OK_CANCEL_OPTION,
                JOptionPane.PLAIN_MESSAGE) != JOptionPane.OK_OPTION) {
            return;
        }
        final Set<Integer> plastico = new HashSet<Integer>();
        for (int i = 0; i < caixas.size(); i++) {
            if (caixas.get(i).isSelected()) {
                plastico.add(i);
            }
        }
        bloquear(true);
        log("Importando dataset " + a.formato + "...");
        new SwingWorker<Integer, String>() {
            @Override
            protected Integer doInBackground() throws Exception {
                return ImportadorDataset.importar(a, plastico, projeto, s -> publish(s));
            }

            @Override
            protected void process(List<String> msgs) {
                for (String s : msgs) {
                    log(s);
                }
            }

            @Override
            protected void done() {
                bloquear(false);
                try {
                    get();
                    recarregarLista(null);
                } catch (Exception e) {
                    erro("Falha ao importar o dataset", e);
                }
            }
        }.execute();
    }

    // ------------------------------------------------------------------ utilidades

    private void atualizarStatus() {
        if (atual == null) {
            status(" ");
            return;
        }
        StringBuilder sb = new StringBuilder();
        sb.append(atual.getName()).append("  (").append(imagemAtual.getWidth()).append("×")
                .append(imagemAtual.getHeight()).append(")  ·  ");
        sb.append(projeto.anotada(atual) ? "anotada" : "não anotada");
        if (sujo) {
            sb.append(" (alterações não salvas)");
        }
        if (resultadoAtual != null) {
            List<Detector.Regiao> regs = resultadoAtual.regioes(limiar());
            sb.append(String.format(Treinamento.PT, "   |   IA: %d região(ões) com plástico, %.1f%% da imagem",
                    regs.size(), resultadoAtual.areaPercentual(limiar())));
        }
        double[] d = desempenho.get(atual.getName());
        if (d != null) {
            sb.append(String.format(Treinamento.PT, "   |   F1 na validação: %.0f%%", d[0] * 100));
        }
        status(sb.toString());
    }

    private void status(String s) {
        lblStatus.setText(s);
    }

    private void log(String s) {
        areaLog.append(s + "\n");
        areaLog.setCaretPosition(areaLog.getDocument().getLength());
    }

    private void bloquear(boolean b) {
        for (JComponent c : acoesLongas) {
            c.setEnabled(!b);
        }
        setCursor(b ? Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR) : Cursor.getDefaultCursor());
    }

    private void aviso(String s) {
        JOptionPane.showMessageDialog(this, s, "Detector de Plástico", JOptionPane.INFORMATION_MESSAGE);
    }

    private void erro(String titulo, Exception e) {
        Throwable c = e;
        while (c.getCause() != null && (c instanceof java.util.concurrent.ExecutionException
                || c.getMessage() == null)) {
            c = c.getCause();
        }
        log("ERRO: " + titulo + ": " + c);
        JOptionPane.showMessageDialog(this, titulo + ":\n" + (c.getMessage() != null ? c.getMessage() : c.toString()),
                "Erro", JOptionPane.ERROR_MESSAGE);
    }

    private void abrirPasta(File f) {
        try {
            Desktop.getDesktop().open(f);
        } catch (Exception e) {
            aviso("Pasta: " + f.getAbsolutePath());
        }
    }

    private void ajuda() {
        String html = "<html><body style='width:520px'>"
                + "<h2>Como usar</h2>"
                + "<ol>"
                + "<li><b>Importe imagens</b> (arquivos, pastas ou .zip). Repetidas são ignoradas.</li>"
                + "<li><b>Anote</b>: pinte de verde tudo o que é plástico. Use <i>Varinha</i> para preencher um objeto "
                + "pelo contorno do filtro P&amp;B, <i>Retângulo</i> para áreas grandes e <i>Ignorar</i> (amarelo) "
                + "onde nem você tem certeza. Salve com <b>Ctrl+S</b>. Salvar sem pintar = imagem sem plástico "
                + "(também ensina a IA!).</li>"
                + "<li><b>Treine a IA</b>. O sistema mede a eficiência (F1) testando cada imagem com um modelo que "
                + "não a viu no treino, e compara com o filtro HSV clássico.</li>"
                + "<li><b>Melhore</b>: use <i>Piores na validação</i> e <i>Sugerir próximas</i> para saber onde a IA "
                + "erra; use <i>Pré-anotar com a IA</i> para anotar mais rápido; treine de novo. A curva em "
                + "<i>Relatório e evolução</i> mostra a eficiência subindo com mais dados.</li>"
                + "<li><b>Detectar em lote</b> analisa uma pasta inteira e gera imagens marcadas + relatorio.csv.</li>"
                + "<li><b>Filtros passo a passo (PDI)</b> mostra, na imagem atual, cada etapa do processamento "
                + "(cinza, equalização, mediana, gaussiano, Sobel, Canny, Otsu, HSV, morfologia, componentes e IA) "
                + "e salva as figuras para o trabalho escrito.</li>"
                + "<li><b>Exportar dataset (YOLO)</b> gera os dados preparados (imagens, rótulos e máscaras) para "
                + "treinar outros modelos de visão computacional, como o YOLO.</li>"
                + "</ol>"
                + "<h3>Visualizações</h3>"
                + "<p><b>Filtro HSV clássico</b>: o método antigo, por faixas fixas de cor. "
                + "<b>P&amp;B contornos</b>: bordas (Canny). <b>IA – mapa de calor</b>: probabilidade de plástico "
                + "(amarelo = dúvida, vermelho = certeza). <b>IA – detecção</b>: regiões acima do limiar.</p>"
                + "<h3>Atalhos</h3>"
                + "<p>B pincel · I ignorar · E borracha · R retângulo · W varinha · H mover · [ ] tamanho do pincel · "
                + "Ctrl+S salvar · Ctrl+Z desfazer · PgUp/PgDn imagem anterior/próxima · roda do mouse = zoom · "
                + "botão do meio arrasta · botão direito apaga.</p>"
                + "</body></html>";
        JOptionPane.showMessageDialog(this, html, "Como usar", JOptionPane.PLAIN_MESSAGE);
    }
}
