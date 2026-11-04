package br.aps.plastico.gui;

import java.awt.AlphaComposite;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Deque;
import java.util.List;

import javax.swing.JComponent;
import javax.swing.SwingUtilities;

import br.aps.plastico.dados.Projeto;
import br.aps.plastico.ia.Detector;

/**
 * Área central: mostra a imagem (ou o filtro escolhido), o resultado da IA e
 * permite pintar a anotação (o "gabarito" que ensina a IA).
 */
public final class PainelImagem extends JComponent {

    public enum Ferramenta {
        MOVER, PINCEL, IGNORAR, BORRACHA, RETANGULO, VARINHA
    }

    public interface Ouvinte {
        void mascaraAlterada();

        void varinha(int x, int y, boolean apagar);
    }

    private static final int COR_PLASTICO = 0x7800E060, COR_IGNORAR = 0x70FFC000;
    private static final int MAX_DESFAZER = 25;

    private final Ouvinte ouvinte;
    private BufferedImage imagem, fundo, calor, calorDeteccao, camada;
    private byte[] mascara;
    private int mw, mh;
    private boolean mostrarMascara = true, mostrarCalor, mostrarCaixas;
    private Detector.Resultado resultado;
    private List<Detector.Regiao> regioes = Collections.emptyList();
    private float limiar = 0.5f;
    private Ferramenta ferramenta = Ferramenta.PINCEL;
    private int raio = 10;
    private double zoom = 1, ox, oy;
    private Point mouse, inicioArrasto;
    private Point2D.Double inicioRet, fimRet, ultimo;
    private byte valorTraco;
    private boolean pintando, movendo, retApagar;
    private final Deque<byte[]> desfazer = new ArrayDeque<byte[]>();

    public PainelImagem(Ouvinte ouvinte) {
        this.ouvinte = ouvinte;
        setOpaque(true);
        setFocusable(true);
        Mouse m = new Mouse();
        addMouseListener(m);
        addMouseMotionListener(m);
        addMouseWheelListener(m);
        atualizarCursor();
    }

    // ------------------------------------------------------------------ estado

    public void setImagem(BufferedImage img, byte[] mascara) {
        this.imagem = img;
        this.mascara = mascara;
        this.mw = img.getWidth();
        this.mh = img.getHeight();
        this.fundo = null;
        camada = new BufferedImage(mw, mh, BufferedImage.TYPE_INT_ARGB);
        atualizarCamada(0, 0, mw - 1, mh - 1);
        desfazer.clear();
        setResultado(null);
        ajustarNaTela();
    }

    public byte[] getMascara() {
        return mascara;
    }

    public boolean temImagem() {
        return imagem != null;
    }

    public void setFundo(BufferedImage f) {
        fundo = f;
        repaint();
    }

    public void setResultado(Detector.Resultado r) {
        resultado = r;
        calor = r == null ? null : r.imagemCalor(0.3f);
        setLimiar(limiar);
    }

    public void setLimiar(float l) {
        limiar = l;
        calorDeteccao = resultado == null ? null : resultado.imagemCalor(limiar);
        regioes = resultado == null ? Collections.<Detector.Regiao>emptyList() : resultado.regioes(limiar);
        repaint();
    }

    public void setModoIa(boolean calor, boolean caixas) {
        mostrarCalor = calor || caixas;
        mostrarCaixas = caixas;
        repaint();
    }

    public void setMostrarMascara(boolean v) {
        mostrarMascara = v;
        repaint();
    }

    public void setFerramenta(Ferramenta f) {
        ferramenta = f;
        atualizarCursor();
        repaint();
    }

    public int getRaio() {
        return raio;
    }

    public void setRaio(int r) {
        raio = Math.max(1, r);
        repaint();
    }

    // ------------------------------------------------------------------ edição da máscara

    public void empilharDesfazer() {
        if (mascara == null) {
            return;
        }
        desfazer.push(mascara.clone());
        while (desfazer.size() > MAX_DESFAZER) {
            desfazer.removeLast();
        }
    }

    public boolean desfazer() {
        if (desfazer.isEmpty()) {
            return false;
        }
        byte[] ant = desfazer.pop();
        System.arraycopy(ant, 0, mascara, 0, mascara.length);
        mascaraSubstituida();
        return true;
    }

    /** Chamar depois de alterar o vetor da máscara diretamente. */
    public void mascaraSubstituida() {
        atualizarCamada(0, 0, mw - 1, mh - 1);
        repaint();
        ouvinte.mascaraAlterada();
    }

    public boolean mascaraVazia() {
        if (mascara == null) {
            return true;
        }
        for (byte b : mascara) {
            if (b != Projeto.FUNDO) {
                return false;
            }
        }
        return true;
    }

    /** Aplica uma região calculada numa versão reduzida da imagem (fator {@code esc}). */
    public void aplicarRegiao(boolean[] reg, int rw, int rh, double esc, byte valor) {
        int minX = rw, minY = rh, maxX = -1, maxY = -1;
        for (int y = 0; y < rh; y++) {
            for (int x = 0; x < rw; x++) {
                if (reg[y * rw + x]) {
                    minX = Math.min(minX, x);
                    maxX = Math.max(maxX, x);
                    minY = Math.min(minY, y);
                    maxY = Math.max(maxY, y);
                }
            }
        }
        if (maxX < 0) {
            return;
        }
        int x0 = (int) Math.floor(minX / esc), y0 = (int) Math.floor(minY / esc);
        int x1 = Math.min(mw - 1, (int) Math.ceil((maxX + 1) / esc)), y1 = Math.min(mh - 1, (int) Math.ceil((maxY + 1) / esc));
        for (int y = y0; y <= y1; y++) {
            int ry = Math.min(rh - 1, (int) (y * esc));
            for (int x = x0; x <= x1; x++) {
                int rx = Math.min(rw - 1, (int) (x * esc));
                if (reg[ry * rw + rx]) {
                    mascara[y * mw + x] = valor;
                }
            }
        }
        atualizarCamada(x0, y0, x1, y1);
        repaint();
        ouvinte.mascaraAlterada();
    }

    private void atualizarCamada(int x0, int y0, int x1, int y1) {
        x0 = Math.max(0, x0);
        y0 = Math.max(0, y0);
        x1 = Math.min(mw - 1, x1);
        y1 = Math.min(mh - 1, y1);
        if (x1 < x0 || y1 < y0) {
            return;
        }
        int larg = x1 - x0 + 1;
        int[] linha = new int[larg];
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                byte v = mascara[y * mw + x];
                linha[x - x0] = v == Projeto.PLASTICO ? COR_PLASTICO : (v == Projeto.IGNORAR ? COR_IGNORAR : 0);
            }
            camada.setRGB(x0, y, larg, 1, linha, 0, larg);
        }
    }

    private void pincelar(double px, double py) {
        int r = raio, cx = (int) Math.round(px), cy = (int) Math.round(py);
        for (int dy = -r; dy <= r; dy++) {
            int yy = cy + dy;
            if (yy < 0 || yy >= mh) {
                continue;
            }
            int dx = (int) Math.sqrt(r * r - dy * dy);
            int xa = Math.max(0, cx - dx), xb = Math.min(mw - 1, cx + dx);
            for (int xx = xa; xx <= xb; xx++) {
                mascara[yy * mw + xx] = valorTraco;
            }
        }
        atualizarCamada(cx - r, cy - r, cx + r, cy + r);
    }

    private void tracar(Point2D.Double a, Point2D.Double b) {
        double d = a.distance(b);
        int passos = Math.max(1, (int) Math.ceil(d / Math.max(1, raio * 0.4)));
        for (int i = 1; i <= passos; i++) {
            double t = i / (double) passos;
            pincelar(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
        }
    }

    private void preencherRet(Point2D.Double a, Point2D.Double b, byte valor) {
        int x0 = (int) Math.max(0, Math.floor(Math.min(a.x, b.x))), x1 = (int) Math.min(mw - 1, Math.ceil(Math.max(a.x, b.x)));
        int y0 = (int) Math.max(0, Math.floor(Math.min(a.y, b.y))), y1 = (int) Math.min(mh - 1, Math.ceil(Math.max(a.y, b.y)));
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                mascara[y * mw + x] = valor;
            }
        }
        atualizarCamada(x0, y0, x1, y1);
    }

    // ------------------------------------------------------------------ zoom e coordenadas

    public void ajustarNaTela() {
        if (imagem == null || getWidth() <= 0) {
            return;
        }
        double z = Math.min((getWidth() - 20) / (double) mw, (getHeight() - 20) / (double) mh);
        zoom = Math.max(0.05, z);
        ox = (getWidth() - mw * zoom) / 2;
        oy = (getHeight() - mh * zoom) / 2;
        repaint();
    }

    private Point2D.Double paraImagem(Point p) {
        return new Point2D.Double((p.x - ox) / zoom, (p.y - oy) / zoom);
    }

    @Override
    public void setBounds(int x, int y, int w, int h) {
        boolean mudou = w != getWidth() || h != getHeight();
        super.setBounds(x, y, w, h);
        if (mudou) {
            ajustarNaTela();
        }
    }

    private void atualizarCursor() {
        setCursor(Cursor.getPredefinedCursor(
                ferramenta == Ferramenta.MOVER ? Cursor.HAND_CURSOR : Cursor.CROSSHAIR_CURSOR));
    }

    // ------------------------------------------------------------------ desenho

    @Override
    protected void paintComponent(Graphics g0) {
        Graphics2D g = (Graphics2D) g0.create();
        g.setColor(new Color(45, 45, 48));
        g.fillRect(0, 0, getWidth(), getHeight());
        if (imagem == null) {
            g.setColor(new Color(200, 200, 200));
            g.setFont(getFont().deriveFont(Font.PLAIN, 16f));
            String s = "Importe imagens (painel à esquerda) para começar.";
            FontMetrics fm = g.getFontMetrics();
            g.drawString(s, (getWidth() - fm.stringWidth(s)) / 2, getHeight() / 2);
            g.dispose();
            return;
        }
        Graphics2D gi = (Graphics2D) g.create();
        gi.translate(ox, oy);
        gi.scale(zoom, zoom);
        gi.setRenderingHint(RenderingHints.KEY_INTERPOLATION, zoom >= 2 ? RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR
                : RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        gi.drawImage(fundo != null ? fundo : imagem, 0, 0, mw, mh, null);
        BufferedImage mapa = mostrarCaixas ? calorDeteccao : calor;
        if (mostrarCalor && mapa != null) {
            gi.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            gi.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, mostrarCaixas ? 0.5f : 0.85f));
            gi.drawImage(mapa, 0, 0, (int) Math.round(resultado.larguraGrade()), (int) Math.round(resultado.alturaGrade()),
                    null);
            gi.setComposite(AlphaComposite.SrcOver);
        }
        if (mostrarMascara && camada != null) {
            gi.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);
            gi.drawImage(camada, 0, 0, null);
        }
        gi.dispose();

        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        if (mostrarCaixas) {
            g.setFont(getFont().deriveFont(Font.BOLD, 12f));
            FontMetrics fm = g.getFontMetrics();
            for (Detector.Regiao r : regioes) {
                Rectangle c = r.caixa;
                int x = (int) Math.round(ox + c.x * zoom), y = (int) Math.round(oy + c.y * zoom);
                int w = (int) Math.round(c.width * zoom), h = (int) Math.round(c.height * zoom);
                g.setColor(new Color(255, 50, 50));
                g.setStroke(new BasicStroke(2f));
                g.drawRect(x, y, w, h);
                String txt = String.format("plástico %.0f%%", r.confianca * 100);
                int tw = fm.stringWidth(txt) + 6, th = fm.getHeight();
                int ty = y - th >= 0 ? y - th : y;
                g.fillRect(x, ty, tw, th);
                g.setColor(Color.WHITE);
                g.drawString(txt, x + 3, ty + fm.getAscent());
            }
        }
        if (inicioRet != null && fimRet != null) {
            double x0 = Math.min(inicioRet.x, fimRet.x), y0 = Math.min(inicioRet.y, fimRet.y);
            double x1 = Math.max(inicioRet.x, fimRet.x), y1 = Math.max(inicioRet.y, fimRet.y);
            g.setColor(retApagar ? Color.RED : Color.GREEN);
            g.setStroke(new BasicStroke(1.5f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10f, new float[] { 6f, 4f }, 0));
            g.drawRect((int) (ox + x0 * zoom), (int) (oy + y0 * zoom), (int) ((x1 - x0) * zoom), (int) ((y1 - y0) * zoom));
        }
        if (mouse != null && (ferramenta == Ferramenta.PINCEL || ferramenta == Ferramenta.IGNORAR
                || ferramenta == Ferramenta.BORRACHA)) {
            int r = (int) Math.round(raio * zoom);
            g.setStroke(new BasicStroke(1.2f));
            g.setColor(Color.BLACK);
            g.drawOval(mouse.x - r - 1, mouse.y - r - 1, 2 * r + 2, 2 * r + 2);
            g.setColor(ferramenta == Ferramenta.PINCEL ? new Color(0, 230, 100)
                    : ferramenta == Ferramenta.IGNORAR ? new Color(255, 200, 0) : Color.WHITE);
            g.drawOval(mouse.x - r, mouse.y - r, 2 * r, 2 * r);
        }
        g.dispose();
    }

    // ------------------------------------------------------------------ mouse

    private final class Mouse extends MouseAdapter {

        @Override
        public void mousePressed(MouseEvent e) {
            requestFocusInWindow();
            if (imagem == null) {
                return;
            }
            boolean esq = SwingUtilities.isLeftMouseButton(e), dir = SwingUtilities.isRightMouseButton(e);
            if (SwingUtilities.isMiddleMouseButton(e) || (esq && ferramenta == Ferramenta.MOVER)) {
                movendo = true;
                inicioArrasto = e.getPoint();
                return;
            }
            Point2D.Double p = paraImagem(e.getPoint());
            switch (ferramenta) {
                case PINCEL:
                case IGNORAR:
                case BORRACHA:
                    empilharDesfazer();
                    pintando = true;
                    valorTraco = dir || ferramenta == Ferramenta.BORRACHA ? Projeto.FUNDO
                            : ferramenta == Ferramenta.IGNORAR ? Projeto.IGNORAR : Projeto.PLASTICO;
                    ultimo = p;
                    pincelar(p.x, p.y);
                    repaint();
                    break;
                case RETANGULO:
                    inicioRet = p;
                    fimRet = p;
                    retApagar = dir;
                    break;
                case VARINHA:
                    if (p.x >= 0 && p.y >= 0 && p.x < mw && p.y < mh) {
                        empilharDesfazer();
                        ouvinte.varinha((int) p.x, (int) p.y, dir || e.isShiftDown());
                    }
                    break;
                default:
                    break;
            }
        }

        @Override
        public void mouseDragged(MouseEvent e) {
            mouse = e.getPoint();
            if (movendo) {
                ox += e.getX() - inicioArrasto.x;
                oy += e.getY() - inicioArrasto.y;
                inicioArrasto = e.getPoint();
                repaint();
                return;
            }
            Point2D.Double p = paraImagem(e.getPoint());
            if (pintando) {
                tracar(ultimo, p);
                ultimo = p;
                repaint();
            } else if (inicioRet != null) {
                fimRet = p;
                repaint();
            }
        }

        @Override
        public void mouseReleased(MouseEvent e) {
            if (movendo) {
                movendo = false;
                return;
            }
            if (pintando) {
                pintando = false;
                ouvinte.mascaraAlterada();
            } else if (inicioRet != null) {
                Point2D.Double p = paraImagem(e.getPoint());
                if (Math.abs(p.x - inicioRet.x) >= 2 && Math.abs(p.y - inicioRet.y) >= 2) {
                    empilharDesfazer();
                    preencherRet(inicioRet, p, retApagar ? Projeto.FUNDO : Projeto.PLASTICO);
                    ouvinte.mascaraAlterada();
                }
                inicioRet = null;
                fimRet = null;
                repaint();
            }
        }

        @Override
        public void mouseMoved(MouseEvent e) {
            mouse = e.getPoint();
            repaint();
        }

        @Override
        public void mouseExited(MouseEvent e) {
            mouse = null;
            repaint();
        }

        @Override
        public void mouseClicked(MouseEvent e) {
            if (e.getClickCount() == 2 && ferramenta == Ferramenta.MOVER) {
                ajustarNaTela();
            }
        }

        @Override
        public void mouseWheelMoved(MouseWheelEvent e) {
            if (imagem == null) {
                return;
            }
            double f = Math.pow(1.15, -e.getPreciseWheelRotation());
            double nz = Math.max(0.05, Math.min(40, zoom * f));
            Point2D.Double p = paraImagem(e.getPoint());
            zoom = nz;
            ox = e.getX() - p.x * zoom;
            oy = e.getY() - p.y * zoom;
            repaint();
        }
    }
}
