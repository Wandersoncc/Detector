package br.aps.plastico.ia;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Consumer;

import br.aps.plastico.dados.Projeto;
import br.aps.plastico.visao.Descritores;
import br.aps.plastico.visao.Filtros;

/**
 * Treina a rede neural a partir das imagens anotadas e mede a eficiência com
 * validação cruzada: cada imagem é avaliada por um modelo que não a viu no treino.
 */
public final class Treinamento {

    public static final Locale PT = new Locale("pt", "BR");

    public static final class Resultado {
        public int imagens;
        public boolean validado;
        public final Metricas ia = new Metricas(), classico = new Metricas();
        /** A eficiência da IA separada por origem: imagens anotadas por vocês e as vindas de datasets. */
        public final Metricas proprias = new Metricas(), importadas = new Metricas();
        public float limiar = 0.5f;
        public long milissegundos;
        /** Imagens ordenadas da pior para a melhor: {nome, F1}. */
        public final List<Object[]> porImagem = new ArrayList<Object[]>();
    }

    static final class DadosImagem {
        final String nome;
        float[] xTreino;
        byte[] yTreino;
        int nTreino;
        float[] xAval, classico, probOof;
        byte[] yAval;
        int nAval;

        DadosImagem(String nome) {
            this.nome = nome;
        }
    }

    // aumento de dados: {escala, espelhar, brilho, contraste, saturação}
    private static final float[][] VARIANTES = {
            { 1.00f, 0, 0.00f, 1.00f, 1.00f },
            { 1.00f, 1, 0.06f, 1.10f, 0.85f },
            { 0.75f, 0, -0.06f, 0.90f, 1.15f },
            { 1.30f, 1, 0.00f, 1.05f, 1.05f },
    };
    /** Célula é plástico se >= 40% dela foi pintada; fundo se <= 10%; o meio-termo é ignorado. */
    static final float FRAC_POS = 0.4f, FRAC_NEG = 0.1f, FRAC_CLASSICO = 0.3f;
    private static final int LOTE = 64, MAX_AMOSTRAS_EPOCA = 30000, MAX_EPOCAS = 40, MIN_EPOCAS = 6, PACIENCIA = 6;
    private static final int MIN_NEGATIVOS = 100;
    private static final int[] CAMADAS = { Descritores.TAMANHO, 64, 32, 1 };
    private static final float LIMITE_Z = 8f;

    private Treinamento() {
    }

    public static Resultado treinar(final Projeto projeto, final Consumer<String> log) throws Exception {
        long t0 = System.currentTimeMillis();
        final List<File> anotadas = projeto.imagensAnotadas();
        final int n = anotadas.size();
        if (n == 0) {
            throw new IllegalStateException(
                    "Nenhuma imagem anotada ainda. Anote pelo menos 1 imagem (o ideal é 10 ou mais) e salve.");
        }
        int nucleos = Math.max(1, Runtime.getRuntime().availableProcessors());
        ExecutorService pool = Executors.newFixedThreadPool(nucleos);
        try {
            log.accept(String.format("Preparando %d imagens anotadas (%d variações de cada uma)...", n,
                    VARIANTES.length));
            List<Future<DadosImagem>> futuros = new ArrayList<Future<DadosImagem>>();
            for (int i = 0; i < n; i++) {
                final File arq = anotadas.get(i);
                final long semente = 1000 + i;
                futuros.add(pool.submit(() -> preparar(projeto, arq, n, new Random(semente))));
            }
            final List<DadosImagem> dados = new ArrayList<DadosImagem>();
            long amostras = 0, positivas = 0, celulas = 0;
            for (int i = 0; i < n; i++) {
                DadosImagem d = futuros.get(i).get();
                dados.add(d);
                amostras += d.nTreino;
                celulas += d.nAval;
                for (int k = 0; k < d.nTreino; k++) {
                    positivas += d.yTreino[k];
                }
                if ((i + 1) % Math.max(1, n / 4) == 0 || i == n - 1) {
                    log.accept(String.format("  %d/%d imagens preparadas", i + 1, n));
                }
            }
            if (positivas == 0) {
                throw new IllegalStateException("As anotações não têm nenhuma área marcada como plástico.");
            }
            log.accept(String.format("Amostras de treino: %d (%d de plástico). Células de avaliação: %d.", amostras,
                    positivas, celulas));

            Resultado res = new Resultado();
            res.imagens = n;
            int epocasFinal = 20;
            if (n >= 4) {
                final int k = Math.min(5, n);
                final int[] dobra = new int[n];
                List<Integer> ordem = new ArrayList<Integer>();
                for (int i = 0; i < n; i++) {
                    ordem.add(i);
                }
                Collections.shuffle(ordem, new Random(7));
                for (int i = 0; i < n; i++) {
                    dobra[ordem.get(i)] = i % k;
                }
                log.accept(String.format("Validação cruzada em %d partes (cada imagem é testada por um modelo que "
                        + "não a viu no treino)...", k));
                List<Future<Integer>> fs = new ArrayList<Future<Integer>>();
                for (int f = 0; f < k; f++) {
                    final int ff = f;
                    fs.add(pool.submit(() -> rodarDobra(dados, dobra, ff, k, log)));
                }
                int[] melhores = new int[k];
                for (int f = 0; f < k; f++) {
                    melhores[f] = fs.get(f).get();
                }
                Arrays.sort(melhores);
                epocasFinal = melhores[k / 2];
                avaliar(dados, res, projeto.imagensDeDatasets());
                res.validado = true;
                log.accept(String.format(PT, "Eficiência na validação: F1 = %.1f%% (precisão %.1f%%, revocação %.1f%%)."
                        + " Filtro HSV clássico: F1 = %.1f%%.", res.ia.f1() * 100, res.ia.precisao() * 100,
                        res.ia.revocacao() * 100, res.classico.f1() * 100));
                if (res.proprias.total() > 0 && res.importadas.total() > 0) {
                    log.accept(String.format(PT, "  por origem: imagens próprias F1 = %.1f%% · datasets importados F1 = %.1f%%",
                            res.proprias.f1() * 100, res.importadas.f1() * 100));
                }
            } else {
                log.accept("Com menos de 4 imagens anotadas o modelo é treinado, mas a eficiência não é medida.");
            }

            epocasFinal = Math.max(MIN_EPOCAS, Math.min(MAX_EPOCAS, epocasFinal));
            log.accept(String.format("Treinando o modelo final com todas as %d imagens (%d épocas)...", n,
                    epocasFinal));
            float[][] nz = normalizacao(dados);
            RedeNeural rede = treinarRede(dados, Collections.<DadosImagem>emptyList(), epocasFinal, nz[0], nz[1],
                    new Random(99), new int[1], log, "modelo final");
            Modelo m = new Modelo();
            m.media = nz[0];
            m.desvio = nz[1];
            m.rede = rede;
            m.limiar = res.limiar;
            m.dataTreino = System.currentTimeMillis();
            m.nImagens = n;
            if (res.validado) {
                m.f1 = (float) res.ia.f1();
                m.precisao = (float) res.ia.precisao();
                m.revocacao = (float) res.ia.revocacao();
                m.iou = (float) res.ia.iou();
                m.f1Classico = (float) res.classico.f1();
                m.f1Proprias = res.proprias.total() > 0 ? (float) res.proprias.f1() : -1;
                m.f1Importadas = res.importadas.total() > 0 ? (float) res.importadas.f1() : -1;
            }
            m.salvar(projeto.arquivoModelo());
            res.milissegundos = System.currentTimeMillis() - t0;
            gravarRelatorios(projeto, dados, res);
            log.accept(String.format(PT, "Modelo salvo em %s (%.0f s).", projeto.arquivoModelo().getName(),
                    res.milissegundos / 1000.0));
            return res;
        } finally {
            pool.shutdownNow();
        }
    }

    // ------------------------------------------------------------------ preparação dos dados

    static DadosImagem preparar(Projeto projeto, File arq, int nImagens, Random rnd) throws IOException {
        BufferedImage img = Projeto.lerImagem(arq);
        int w0 = img.getWidth(), h0 = img.getHeight();
        byte[] mascara = projeto.carregarMascara(arq, w0, h0);
        DadosImagem d = new DadosImagem(arq.getName());
        final int F = Descritores.TAMANHO, CEL = Descritores.CEL;
        // no pior caso ~2 milhões de amostras (cerca de 1,3 GB), mesmo com milhares de imagens
        int limitePos = Math.max(60, Math.min(600, 2000000 / Math.max(1, nImagens * VARIANTES.length * 3)));
        List<float[]> blocosX = new ArrayList<float[]>();
        List<byte[]> blocosY = new ArrayList<byte[]>();

        for (int v = 0; v < VARIANTES.length; v++) {
            float[] var = VARIANTES[v];
            double esc = Filtros.escalaPara(w0, h0, Modelo.LADO_BASE) * var[0];
            int w = Math.max(32, (int) Math.round(w0 * esc)), h = Math.max(32, (int) Math.round(h0 * esc));
            BufferedImage iv = Filtros.redimensionar(img, w, h);
            byte[] mv = Projeto.redimensionarMascara(mascara, w0, h0, w, h);
            if (var[1] > 0) {
                iv = Filtros.espelhar(iv);
                espelharMascara(mv, w, h);
            }
            Filtros.Rgb rgb = Filtros.paraRgb(iv);
            if (v > 0) {
                Filtros.ajustarCores(rgb, var[2], var[3], var[4]);
            }
            Descritores.Grade g = Descritores.calcular(rgb);
            int[][] cont = contarPorSubcelula(mv, w, h, g.gw, g.gh);

            if (v == 0) {
                // avaliação: todas as células sem sobreposição que têm rótulo definido
                boolean[] cl = Filtros.filtroHsvClassico(rgb);
                byte[] clB = new byte[cl.length];
                for (int i = 0; i < cl.length; i++) {
                    clB[i] = cl[i] ? Projeto.PLASTICO : Projeto.FUNDO;
                }
                int[][] contCl = contarPorSubcelula(clB, w, h, g.gw, g.gh);
                List<int[]> aval = new ArrayList<int[]>();
                for (int cy = 0; cy + CEL <= g.gh; cy += CEL) {
                    for (int cx = 0; cx + CEL <= g.gw; cx += CEL) {
                        int r = rotulo(cont, g.gw, cx, cy);
                        if (r >= 0) {
                            aval.add(new int[] { cx, cy, r });
                        }
                    }
                }
                // com muitas imagens, uma amostra das células basta e economiza memória
                int limiteAval = nImagens <= 300 ? Integer.MAX_VALUE : Math.max(200, 300000 / nImagens);
                if (aval.size() > limiteAval) {
                    Collections.shuffle(aval, rnd);
                    aval = aval.subList(0, limiteAval);
                }
                d.nAval = aval.size();
                d.xAval = new float[d.nAval * F];
                d.yAval = new byte[d.nAval];
                d.classico = new float[d.nAval];
                for (int k = 0; k < d.nAval; k++) {
                    int[] a = aval.get(k);
                    Descritores.extrair(g, a[0], a[1], d.xAval, k * F);
                    d.yAval[k] = (byte) a[2];
                    d.classico[k] = fracao(contCl, g.gw, a[0], a[1]);
                }
            }

            // treino: plástico em todas as posições (passo de 8 px), fundo em passo de 16 px
            List<int[]> pos = new ArrayList<int[]>(), neg = new ArrayList<int[]>();
            for (int cy = 0; cy + CEL <= g.gh; cy++) {
                for (int cx = 0; cx + CEL <= g.gw; cx++) {
                    int r = rotulo(cont, g.gw, cx, cy);
                    if (r == 1) {
                        pos.add(new int[] { cx, cy });
                    } else if (r == 0 && cx % CEL == 0 && cy % CEL == 0) {
                        neg.add(new int[] { cx, cy });
                    }
                }
            }
            // amostras proporcionais ao plástico anotado: imagens ricas em exemplos pesam mais
            // que fotos com um único objeto pequeno, e o fundo fica em ~2 para 1
            Collections.shuffle(pos, rnd);
            Collections.shuffle(neg, rnd);
            int np = Math.min(pos.size(), limitePos);
            int nn = Math.min(neg.size(), Math.max(MIN_NEGATIVOS, 2 * np));
            float[] bx = new float[(np + nn) * F];
            byte[] by = new byte[np + nn];
            for (int k = 0; k < np + nn; k++) {
                int[] c = k < np ? pos.get(k) : neg.get(k - np);
                Descritores.extrair(g, c[0], c[1], bx, k * F);
                by[k] = (byte) (k < np ? 1 : 0);
            }
            blocosX.add(bx);
            blocosY.add(by);
        }
        int total = 0;
        for (byte[] b : blocosY) {
            total += b.length;
        }
        d.nTreino = total;
        d.xTreino = new float[total * F];
        d.yTreino = new byte[total];
        int o = 0;
        for (int i = 0; i < blocosY.size(); i++) {
            System.arraycopy(blocosX.get(i), 0, d.xTreino, o * F, blocosX.get(i).length);
            System.arraycopy(blocosY.get(i), 0, d.yTreino, o, blocosY.get(i).length);
            o += blocosY.get(i).length;
        }
        return d;
    }

    private static void espelharMascara(byte[] m, int w, int h) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w / 2; x++) {
                byte t = m[y * w + x];
                m[y * w + x] = m[y * w + w - 1 - x];
                m[y * w + w - 1 - x] = t;
            }
        }
    }

    /** Por sub-célula: {pixels de plástico, pixels "ignorar", total de pixels}. */
    private static int[][] contarPorSubcelula(byte[] m, int w, int h, int gw, int gh) {
        int P = Descritores.P;
        int[] pl = new int[gw * gh], ig = new int[gw * gh], tot = new int[gw * gh];
        for (int y = 0; y < h; y++) {
            int linha = (y / P) * gw;
            for (int x = 0; x < w; x++) {
                int s = linha + x / P;
                byte v = m[y * w + x];
                tot[s]++;
                if (v == Projeto.PLASTICO) {
                    pl[s]++;
                } else if (v == Projeto.IGNORAR) {
                    ig[s]++;
                }
            }
        }
        return new int[][] { pl, ig, tot };
    }

    /** 1 = plástico, 0 = fundo, -1 = incerto/ignorado. */
    private static int rotulo(int[][] c, int gw, int cx, int cy) {
        int pl = 0, ig = 0, tot = 0;
        for (int dy = 0; dy < Descritores.CEL; dy++) {
            for (int dx = 0; dx < Descritores.CEL; dx++) {
                int s = (cy + dy) * gw + cx + dx;
                pl += c[0][s];
                ig += c[1][s];
                tot += c[2][s];
            }
        }
        if (tot == 0 || ig * 2 > tot) {
            return -1;
        }
        float f = pl / (float) (tot - ig);
        return f >= FRAC_POS ? 1 : (f <= FRAC_NEG ? 0 : -1);
    }

    private static float fracao(int[][] c, int gw, int cx, int cy) {
        int pl = 0, tot = 0;
        for (int dy = 0; dy < Descritores.CEL; dy++) {
            for (int dx = 0; dx < Descritores.CEL; dx++) {
                int s = (cy + dy) * gw + cx + dx;
                pl += c[0][s];
                tot += c[2][s];
            }
        }
        return tot == 0 ? 0 : pl / (float) tot;
    }

    // ------------------------------------------------------------------ treino da rede

    private static int rodarDobra(List<DadosImagem> dados, int[] dobra, int f, int k, Consumer<String> log) {
        Random rnd = new Random(77 + f);
        List<DadosImagem> treino = new ArrayList<DadosImagem>(), teste = new ArrayList<DadosImagem>();
        for (int i = 0; i < dados.size(); i++) {
            (dobra[i] == f ? teste : treino).add(dados.get(i));
        }
        // ~15% das imagens de treino decidem quando parar (evita decorar as imagens)
        List<DadosImagem> embaralhadas = new ArrayList<DadosImagem>(treino);
        Collections.shuffle(embaralhadas, rnd);
        int nParada = treino.size() >= 3 ? Math.max(1, Math.round(treino.size() * 0.15f)) : 0;
        List<DadosImagem> parada = embaralhadas.subList(0, nParada);
        List<DadosImagem> efetivo = embaralhadas.subList(nParada, embaralhadas.size());
        float[][] nz = normalizacao(efetivo);
        int[] melhor = new int[1];
        RedeNeural rede = treinarRede(efetivo, parada, 0, nz[0], nz[1], rnd, melhor, null, null);
        int F = Descritores.TAMANHO;
        float[] z = new float[F];
        for (DadosImagem d : teste) {
            d.probOof = new float[d.nAval];
            for (int i = 0; i < d.nAval; i++) {
                normalizar(d.xAval, i * F, nz[0], nz[1], z);
                d.probOof[i] = rede.prever(z);
            }
        }
        log.accept(String.format("  parte %d/%d concluída (melhor época: %d)", f + 1, k, melhor[0]));
        return melhor[0];
    }

    private static RedeNeural treinarRede(List<DadosImagem> treino, List<DadosImagem> parada, int epocasFixas,
            float[] media, float[] desvio, Random rnd, int[] melhorEpoca, Consumer<String> log, String nome) {
        final int F = Descritores.TAMANHO;
        int total = 0, positivas = 0;
        for (DadosImagem d : treino) {
            total += d.nTreino;
            for (int i = 0; i < d.nTreino; i++) {
                positivas += d.yTreino[i];
            }
        }
        int[] refImg = new int[total], refLin = new int[total], ordem = new int[total];
        for (int i = 0, a = 0; i < treino.size(); i++) {
            for (int l = 0; l < treino.get(i).nTreino; l++, a++) {
                refImg[a] = i;
                refLin[a] = l;
                ordem[a] = a;
            }
        }
        float pesoPos = positivas == 0 ? 1f : Math.max(1f, Math.min(4f, (total - positivas) / (float) positivas));
        RedeNeural rede = new RedeNeural(CAMADAS, rnd);
        RedeNeural melhorRede = null;
        double melhorPerda = Double.MAX_VALUE;
        int semMelhora = 0;
        int maxEpocas = epocasFixas > 0 ? epocasFixas : MAX_EPOCAS;
        melhorEpoca[0] = maxEpocas;
        float[] x = new float[F];

        for (int ep = 1; ep <= maxEpocas; ep++) {
            for (int i = total - 1; i > 0; i--) {
                int j = rnd.nextInt(i + 1), t = ordem[i];
                ordem[i] = ordem[j];
                ordem[j] = t;
            }
            int usar = Math.min(total, MAX_AMOSTRAS_EPOCA);
            double perda = 0;
            for (int ini = 0; ini < usar; ini += LOTE) {
                int fim = Math.min(usar, ini + LOTE);
                rede.zerarGradientes();
                for (int q = ini; q < fim; q++) {
                    int a = ordem[q];
                    DadosImagem d = treino.get(refImg[a]);
                    normalizar(d.xTreino, refLin[a] * F, media, desvio, x);
                    int y = d.yTreino[refLin[a]];
                    perda += rede.acumular(x, y, y == 1 ? pesoPos : 1f, rnd);
                }
                rede.passoAdam(fim - ini);
            }
            if (log != null && (ep == 1 || ep % 5 == 0 || ep == maxEpocas)) {
                log.accept(String.format(PT, "  %s – época %d/%d, perda %.4f", nome, ep, maxEpocas, perda / usar));
            }
            if (epocasFixas <= 0 && !parada.isEmpty()) {
                double pv = perdaBalanceada(rede, parada, media, desvio);
                if (pv < melhorPerda - 1e-4) {
                    melhorPerda = pv;
                    melhorRede = rede.copia();
                    melhorEpoca[0] = ep;
                    semMelhora = 0;
                } else if (++semMelhora >= PACIENCIA && ep >= MIN_EPOCAS) {
                    break;
                }
            }
        }
        return melhorRede != null ? melhorRede : rede;
    }

    /** Média da perda das células de plástico e das de fundo (não favorece a classe maior). */
    private static double perdaBalanceada(RedeNeural rede, List<DadosImagem> dados, float[] media, float[] desvio) {
        int F = Descritores.TAMANHO;
        float[] z = new float[F];
        double[] soma = new double[2];
        long[] n = new long[2];
        for (DadosImagem d : dados) {
            for (int i = 0; i < d.nAval; i++) {
                normalizar(d.xAval, i * F, media, desvio, z);
                float p = Math.min(1f - 1e-6f, Math.max(1e-6f, rede.prever(z)));
                int y = d.yAval[i];
                soma[y] += -(y == 1 ? Math.log(p) : Math.log(1 - p));
                n[y]++;
            }
        }
        double a = n[0] == 0 ? 0 : soma[0] / n[0], b = n[1] == 0 ? 0 : soma[1] / n[1];
        return n[0] == 0 || n[1] == 0 ? a + b : (a + b) / 2;
    }

    static float[][] normalizacao(List<DadosImagem> dados) {
        int F = Descritores.TAMANHO;
        double[] s = new double[F], s2 = new double[F];
        long n = 0;
        for (DadosImagem d : dados) {
            for (int i = 0; i < d.nTreino; i++) {
                int o = i * F;
                for (int j = 0; j < F; j++) {
                    double v = d.xTreino[o + j];
                    s[j] += v;
                    s2[j] += v * v;
                }
            }
            n += d.nTreino;
        }
        float[] media = new float[F], desvio = new float[F];
        for (int j = 0; j < F; j++) {
            double m = n == 0 ? 0 : s[j] / n;
            double var = n == 0 ? 1 : s2[j] / n - m * m;
            media[j] = (float) m;
            desvio[j] = (float) Math.max(0.01, Math.sqrt(Math.max(0, var)));
        }
        return new float[][] { media, desvio };
    }

    static void normalizar(float[] x, int off, float[] media, float[] desvio, float[] out) {
        for (int j = 0; j < out.length; j++) {
            float z = (x[off + j] - media[j]) / desvio[j];
            out[j] = z > LIMITE_Z ? LIMITE_Z : (z < -LIMITE_Z ? -LIMITE_Z : z);
        }
    }

    // ------------------------------------------------------------------ avaliação

    private static void avaliar(List<DadosImagem> dados, Resultado res, java.util.Set<String> importadas) {
        float melhorT = 0.5f;
        double melhorF1 = -1;
        for (int i = 0; i <= 28; i++) {
            float t = 0.15f + i * 0.025f;
            Metricas m = new Metricas();
            for (DadosImagem d : dados) {
                for (int k = 0; k < d.nAval; k++) {
                    m.somar(d.probOof[k] >= t, d.yAval[k] == 1);
                }
            }
            if (m.f1() > melhorF1 + 1e-9) {
                melhorF1 = m.f1();
                melhorT = t;
            }
        }
        res.limiar = melhorT;
        for (DadosImagem d : dados) {
            Metricas m = new Metricas();
            for (int k = 0; k < d.nAval; k++) {
                boolean real = d.yAval[k] == 1;
                m.somar(d.probOof[k] >= melhorT, real);
                res.classico.somar(d.classico[k] >= FRAC_CLASSICO, real);
            }
            res.ia.somar(m);
            (importadas.contains(d.nome) ? res.importadas : res.proprias).somar(m);
            res.porImagem.add(new Object[] { d.nome, m });
        }
        Collections.sort(res.porImagem, new Comparator<Object[]>() {
            @Override
            public int compare(Object[] a, Object[] b) {
                return Double.compare(((Metricas) a[1]).f1(), ((Metricas) b[1]).f1());
            }
        });
    }

    /** F1 em % com uma casa, ou vazio se o grupo não tem imagens. */
    private static String pct(Metricas m) {
        return m.total() == 0 ? "" : String.format(PT, "%.1f", m.f1() * 100);
    }

    private static void gravarRelatorios(Projeto projeto, List<DadosImagem> dados, Resultado res) throws IOException {
        File desempenho = projeto.arquivoDesempenho();
        if (!res.validado) {
            desempenho.delete();
            return;
        }
        PrintWriter pw = Projeto.escreverCsv(desempenho, false);
        try {
            pw.println("imagem;f1_%;precisao_%;revocacao_%;celulas_plastico;celulas_avaliadas;vp;fp;fn;vn");
            for (Object[] o : res.porImagem) {
                Metricas m = (Metricas) o[1];
                pw.println(String.format(PT, "%s;%.1f;%.1f;%.1f;%d;%d;%d;%d;%d;%d", o[0], m.f1() * 100,
                        m.precisao() * 100, m.revocacao() * 100, m.vp + m.fn, m.total(), m.vp, m.fp, m.fn, m.vn));
            }
        } finally {
            pw.close();
        }
        File hist = projeto.arquivoHistorico();
        boolean novo = !hist.isFile();
        pw = Projeto.escreverCsv(hist, true);
        try {
            if (novo) {
                pw.println("data;imagens_anotadas;f1_ia_%;precisao_%;revocacao_%;iou_%;f1_filtro_hsv_%;limiar;segundos;"
                        + "f1_imagens_proprias_%;f1_datasets_importados_%");
            }
            pw.println(String.format(PT, "%s;%d;%.1f;%.1f;%.1f;%.1f;%.1f;%.3f;%.0f;%s;%s",
                    new SimpleDateFormat("dd/MM/yyyy HH:mm").format(new Date()), res.imagens, res.ia.f1() * 100,
                    res.ia.precisao() * 100, res.ia.revocacao() * 100, res.ia.iou() * 100,
                    res.classico.f1() * 100, res.limiar, res.milissegundos / 1000.0, pct(res.proprias),
                    pct(res.importadas)));
        } finally {
            pw.close();
        }
    }
}
