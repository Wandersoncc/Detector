package br.aps.plastico.gui;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.List;
import java.util.function.Consumer;

import javax.swing.BorderFactory;
import javax.swing.JComponent;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTabbedPane;
import javax.swing.JTable;
import javax.swing.table.DefaultTableModel;

import br.aps.plastico.dados.Projeto;

/** Desempenho por imagem e evolução da eficiência a cada treino. */
final class DialogoRelatorio extends JDialog {

    DialogoRelatorio(JFrame dono, Projeto projeto, final Consumer<String> abrirImagem) {
        super(dono, "Relatório da IA", false);
        JTabbedPane abas = new JTabbedPane();

        // --- piores imagens
        List<String[]> desempenho = Projeto.lerCsv(projeto.arquivoDesempenho());
        DefaultTableModel mt = new DefaultTableModel(
                new Object[] { "Imagem", "F1 (%)", "Precisão (%)", "Revocação (%)", "Células c/ plástico" }, 0) {
            @Override
            public boolean isCellEditable(int r, int c) {
                return false;
            }
        };
        for (String[] l : desempenho) {
            if (l.length >= 5) {
                mt.addRow(new Object[] { l[0], l[1], l[2], l[3], l[4] });
            }
        }
        final JTable tabela = new JTable(mt);
        tabela.getColumnModel().getColumn(0).setPreferredWidth(220);
        tabela.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                int r = tabela.getSelectedRow();
                if (e.getClickCount() == 2 && r >= 0) {
                    abrirImagem.accept((String) tabela.getValueAt(r, 0));
                }
            }
        });
        JPanel p1 = new JPanel(new BorderLayout(4, 4));
        p1.setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));
        p1.add(new JLabel(desempenho.isEmpty()
                ? "<html>Ainda não há validação. Anote 4+ imagens e treine a IA.</html>"
                : "<html>Imagens ordenadas da <b>pior</b> para a melhor na validação cruzada (cada uma foi testada por "
                        + "um modelo que não a viu).<br>Revise a anotação das piores e anote imagens parecidas: é onde o "
                        + "treino rende mais. Dois cliques abrem a imagem.</html>"),
                BorderLayout.NORTH);
        p1.add(new JScrollPane(tabela), BorderLayout.CENTER);
        abas.addTab("Imagens com pior desempenho", p1);

        // --- evolução
        List<String[]> hist = Projeto.lerCsv(projeto.arquivoHistorico());
        DefaultTableModel mh = new DefaultTableModel(new Object[] { "Data", "Imagens anotadas", "F1 IA (%)",
                "Precisão (%)", "Revocação (%)", "F1 filtro HSV (%)", "F1 suas imagens (%)", "F1 datasets (%)" }, 0) {
            @Override
            public boolean isCellEditable(int r, int c) {
                return false;
            }
        };
        for (String[] l : hist) {
            if (l.length >= 7) {
                mh.addRow(new Object[] { l[0], l[1], l[2], l[3], l[4], l[6], l.length > 9 ? l[9] : "",
                        l.length > 10 ? l[10] : "" });
            }
        }
        JSplitPane sp = new JSplitPane(JSplitPane.VERTICAL_SPLIT, new Grafico(hist), new JScrollPane(new JTable(mh)));
        sp.setResizeWeight(0.65);
        sp.setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));
        abas.addTab("Evolução da eficiência", sp);
        if (desempenho.isEmpty() && !hist.isEmpty()) {
            abas.setSelectedIndex(1);
        }

        setContentPane(abas);
        setSize(760, 560);
        setLocationRelativeTo(dono);
    }

    /** Linha da eficiência (F1) da IA e do filtro clássico ao longo dos treinos. */
    private static final class Grafico extends JComponent {
        private final List<String[]> hist;

        Grafico(List<String[]> hist) {
            this.hist = hist;
            setPreferredSize(new Dimension(700, 300));
        }

        @Override
        protected void paintComponent(Graphics g0) {
            Graphics2D g = (Graphics2D) g0.create();
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
            g.setColor(Color.WHITE);
            g.fillRect(0, 0, getWidth(), getHeight());
            int esq = 50, dir = 20, topo = 30, base = 40;
            int w = getWidth() - esq - dir, h = getHeight() - topo - base;
            g.setFont(getFont().deriveFont(11f));
            FontMetrics fm = g.getFontMetrics();
            for (int v = 0; v <= 100; v += 20) {
                int y = topo + h - v * h / 100;
                g.setColor(new Color(230, 230, 230));
                g.drawLine(esq, y, esq + w, y);
                g.setColor(Color.GRAY);
                String s = v + "%";
                g.drawString(s, esq - 6 - fm.stringWidth(s), y + 4);
            }
            g.setColor(Color.DARK_GRAY);
            g.setFont(getFont().deriveFont(Font.BOLD, 12f));
            g.drawString("Eficiência (F1) a cada treino — rótulo = nº de imagens anotadas", esq, 18);
            if (hist.isEmpty()) {
                g.setFont(getFont().deriveFont(13f));
                g.drawString("Nenhum treino com validação ainda.", esq + 20, topo + h / 2);
                g.dispose();
                return;
            }
            int n = hist.size(), folga = 30;
            int[] xs = new int[n], yIa = new int[n], yCl = new int[n], yProp = new int[n];
            boolean misturou = false;
            for (int i = 0; i < n; i++) {
                String[] l = hist.get(i);
                xs[i] = esq + folga + (n == 1 ? (w - 2 * folga) / 2 : i * (w - 2 * folga) / (n - 1));
                yIa[i] = topo + h - (int) Math.round(valor(l, 2) * h / 100);
                yCl[i] = topo + h - (int) Math.round(valor(l, 6) * h / 100);
                // sem datasets importados, "suas imagens" é o próprio F1 geral
                yProp[i] = topo + h - (int) Math.round(valor(l, tem(l, 9) ? 9 : 2) * h / 100);
                misturou |= tem(l, 10);
            }
            Color verde = new Color(26, 156, 58), azul = new Color(41, 98, 255), cinza = new Color(150, 150, 150);
            linha(g, xs, yCl, cinza, new float[] { 5f, 4f });
            if (misturou) {
                linha(g, xs, yProp, azul, null);
            }
            linha(g, xs, yIa, verde, null);
            g.setFont(getFont().deriveFont(11f));
            for (int i = 0; i < n; i++) {
                String[] l = hist.get(i);
                g.setColor(verde);
                String s = l[2] + "%";
                g.drawString(s, xs[i] - fm.stringWidth(s) / 2, yIa[i] - 8);
                if (misturou && tem(l, 10)) {
                    g.setColor(azul);
                    String p = l[9] + "%";
                    g.drawString(p, xs[i] - fm.stringWidth(p) / 2, yProp[i] + (yProp[i] < yIa[i] ? -8 : 16));
                }
                g.setColor(Color.GRAY);
                String r = l[1] + " img";
                g.drawString(r, xs[i] - fm.stringWidth(r) / 2, topo + h + 16);
            }
            int lx = esq;
            lx = legenda(g, lx, topo + h + 30, verde, "IA – todas as imagens");
            if (misturou) {
                lx = legenda(g, lx, topo + h + 30, azul, "IA – só as suas imagens");
            }
            legenda(g, lx, topo + h + 30, cinza, "Filtro HSV clássico");
            g.dispose();
        }

        private static int legenda(Graphics2D g, int x, int y, Color cor, String texto) {
            g.setColor(cor);
            g.fillRect(x, y - 6, 14, 4);
            g.setColor(Color.DARK_GRAY);
            g.drawString(texto, x + 20, y);
            return x + 40 + g.getFontMetrics().stringWidth(texto);
        }

        private static boolean tem(String[] l, int i) {
            return i < l.length && !l[i].trim().isEmpty() && !Double.isNaN(Projeto.numero(l[i]));
        }

        private static double valor(String[] l, int i) {
            double v = i < l.length ? Projeto.numero(l[i]) : 0;
            return Double.isNaN(v) ? 0 : Math.max(0, Math.min(100, v));
        }

        private static void linha(Graphics2D g, int[] xs, int[] ys, Color cor, float[] traco) {
            g.setColor(cor);
            g.setStroke(traco == null ? new BasicStroke(2.5f)
                    : new BasicStroke(1.8f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_ROUND, 10f, traco, 0));
            for (int i = 1; i < xs.length; i++) {
                g.drawLine(xs[i - 1], ys[i - 1], xs[i], ys[i]);
            }
            for (int i = 0; i < xs.length; i++) {
                g.fillOval(xs[i] - 4, ys[i] - 4, 8, 8);
            }
        }
    }
}
