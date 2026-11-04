package br.aps.plastico.gui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Desktop;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.Image;
import java.awt.Toolkit;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;

import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.SwingConstants;

import br.aps.plastico.visao.Etapas;

/** Mostra o processamento da imagem atual etapa por etapa, com explicação e resultado. */
final class DialogoFiltros extends JDialog {

    private static final int LARGURA_MINIATURA = 340;

    DialogoFiltros(final JFrame dono, final String nomeImagem, final List<Etapas.Etapa> etapas, final File pastaSaida) {
        super(dono, "Filtros passo a passo – " + nomeImagem, false);
        JPanel grade = new JPanel(new GridLayout(0, 3, 12, 12));
        grade.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        for (final Etapas.Etapa e : etapas) {
            grade.add(cartao(e));
        }
        JScrollPane sp = new JScrollPane(grade);
        sp.getVerticalScrollBar().setUnitIncrement(24);

        JPanel sul = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        sul.add(new JLabel("Clique numa etapa para ampliar.   "));
        JButton salvar = new JButton("Salvar as etapas em PNG (para o trabalho escrito)");
        salvar.addActionListener(ev -> {
            File pasta = new File(pastaSaida, "filtros_" + nomeImagem.replaceAll("\\.[^.]+$", "")
                    .replaceAll("[^A-Za-z0-9_-]+", "_").replaceAll("^_+|_+$", ""));
            try {
                Etapas.salvar(etapas, nomeImagem, pasta);
                if (JOptionPane.showConfirmDialog(this, etapas.size() + " imagens e o arquivo etapas.txt foram salvos em:\n"
                        + pasta.getAbsolutePath() + "\n\nAbrir a pasta?", "Etapas salvas", JOptionPane.YES_NO_OPTION) == 0) {
                    Desktop.getDesktop().open(pasta);
                }
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this, "Não foi possível salvar: " + ex.getMessage(), "Erro",
                        JOptionPane.ERROR_MESSAGE);
            }
        });
        JButton fechar = new JButton("Fechar");
        fechar.addActionListener(ev -> dispose());
        sul.add(salvar);
        sul.add(fechar);

        getContentPane().setLayout(new BorderLayout());
        getContentPane().add(sp, BorderLayout.CENTER);
        getContentPane().add(sul, BorderLayout.SOUTH);
        Dimension tela = Toolkit.getDefaultToolkit().getScreenSize();
        setSize(Math.min(tela.width - 60, 3 * LARGURA_MINIATURA + 150), Math.min(tela.height - 80, 900));
        setLocationRelativeTo(dono);
    }

    private JPanel cartao(final Etapas.Etapa e) {
        JPanel p = new JPanel(new BorderLayout(4, 4));
        p.setBorder(BorderFactory.createCompoundBorder(BorderFactory.createLineBorder(new Color(210, 210, 210)),
                BorderFactory.createEmptyBorder(6, 6, 6, 6)));
        JLabel titulo = new JLabel(e.titulo);
        titulo.setFont(titulo.getFont().deriveFont(Font.BOLD, titulo.getFont().getSize2D() + 1f));
        p.add(titulo, BorderLayout.NORTH);

        BufferedImage img = e.imagem;
        int h = Math.max(1, img.getHeight() * LARGURA_MINIATURA / img.getWidth());
        JLabel figura = new JLabel(new ImageIcon(img.getScaledInstance(LARGURA_MINIATURA, h, Image.SCALE_SMOOTH)));
        figura.setHorizontalAlignment(SwingConstants.CENTER);
        figura.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        figura.setToolTipText("Clique para ampliar");
        figura.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent ev) {
                ampliar(e);
            }
        });
        p.add(figura, BorderLayout.CENTER);

        // o HTML do Swing mede "px" como ponto (x4/3), por isso a largura vai reduzida
        JLabel texto = new JLabel("<html><body style='width:" + (LARGURA_MINIATURA - 10) * 3 / 4 + "px'>" + html(e.explicacao)
                + "<br><b>" + html(e.info) + "</b></body></html>");
        texto.setFont(texto.getFont().deriveFont(Font.PLAIN));
        p.add(texto, BorderLayout.SOUTH);
        return p;
    }

    private void ampliar(Etapas.Etapa e) {
        Dimension tela = Toolkit.getDefaultToolkit().getScreenSize();
        double esc = Math.min(1.5, Math.min((tela.width - 120) / (double) e.imagem.getWidth(),
                (tela.height - 200) / (double) e.imagem.getHeight()));
        int w = (int) (e.imagem.getWidth() * esc), h = (int) (e.imagem.getHeight() * esc);
        JDialog d = new JDialog(this, e.titulo, false);
        JLabel figura = new JLabel(new ImageIcon(e.imagem.getScaledInstance(w, h, Image.SCALE_SMOOTH)));
        JLabel texto = new JLabel("<html>" + html(e.explicacao) + "<br><b>" + html(e.info) + "</b></html>");
        texto.setBorder(BorderFactory.createEmptyBorder(6, 8, 8, 8));
        d.getContentPane().add(figura, BorderLayout.CENTER);
        d.getContentPane().add(texto, BorderLayout.SOUTH);
        d.pack();
        d.setLocationRelativeTo(this);
        d.setVisible(true);
    }

    private static String html(String s) {
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;");
    }
}
