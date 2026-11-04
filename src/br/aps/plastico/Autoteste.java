package br.aps.plastico;

import java.awt.GraphicsEnvironment;
import java.awt.image.BufferedImage;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

import br.aps.plastico.dados.Projeto;
import br.aps.plastico.ia.Detector;
import br.aps.plastico.ia.Modelo;
import br.aps.plastico.ia.Treinamento;
import br.aps.plastico.visao.Etapas;
import br.aps.plastico.visao.Filtros;

/** Confere, em poucos segundos, se tudo o que o sistema precisa está funcionando nesta máquina. */
final class Autoteste {

    private int falhas;

    private Autoteste() {
    }

    static boolean rodar(Projeto projeto) {
        Autoteste t = new Autoteste();
        System.out.println("=== Teste do Detector de Plástico ===");
        System.out.println();
        t.java();
        t.pastas(projeto);
        Modelo m = t.modelo(projeto);
        t.detectarEFiltrar(projeto, m);
        t.janela();
        System.out.println();
        if (t.falhas == 0) {
            System.out.println("RESULTADO: TUDO OK – pode abrir o programa.");
        } else {
            System.out.println("RESULTADO: " + t.falhas + " problema(s). Veja as linhas com [FALHOU] acima.");
        }
        return t.falhas == 0;
    }

    private void ok(String s) {
        System.out.println("  [OK]      " + s);
    }

    private void falhou(String s, String dica) {
        falhas++;
        System.out.println("  [FALHOU]  " + s);
        System.out.println("            -> " + dica);
    }

    private void java() {
        String versao = System.getProperty("java.version");
        String bits = System.getProperty("sun.arch.data.model", "?");
        long mb = Runtime.getRuntime().maxMemory() / (1024 * 1024);
        ok("Java " + versao + " (" + bits + " bits), memória disponível para o programa: " + mb + " MB");
        if (mb < 700) {
            falhou("Pouca memória para treinar com muitas imagens (" + mb + " MB)",
                    "use o INICIAR.bat (ele pede mais memória ao Java) ou instale o Java 64 bits");
        }
    }

    private void pastas(Projeto p) {
        int imagens = p.listarImagens().size(), anotadas = p.imagensAnotadas().size();
        if (imagens == 0) {
            falhou("Nenhuma imagem em " + p.pastaImagens, "extraia o .zip inteiro antes de rodar (botão direito > Extrair tudo)");
        } else {
            ok(String.format("Pasta do projeto: %s (%d imagens, %d anotadas)", p.raiz, imagens, anotadas));
        }
        File teste = new File(p.pastaResultados, ".teste_escrita");
        try {
            if (!teste.createNewFile() && !teste.exists()) {
                throw new java.io.IOException("sem permissão");
            }
            teste.delete();
            ok("Permissão para gravar resultados");
        } catch (Exception e) {
            falhou("Não consigo gravar em " + p.pastaResultados,
                    "copie a pasta para Documentos ou Área de Trabalho (fora de Arquivos de Programas)");
        }
    }

    private Modelo modelo(Projeto p) {
        File f = p.arquivoModelo();
        if (!f.isFile()) {
            falhou("Modelo treinado não encontrado", "abra o programa e clique em 'Treinar IA com as anotações'");
            return null;
        }
        try {
            Modelo m = Modelo.carregar(f);
            String f1 = m.f1 >= 0 ? String.format(Treinamento.PT, ", eficiência F1 %.1f%%", m.f1 * 100) : "";
            ok("Modelo da IA carregado (treinado em "
                    + new SimpleDateFormat("dd/MM/yyyy HH:mm").format(new Date(m.dataTreino)) + " com " + m.nImagens
                    + " imagens" + f1 + ")");
            return m;
        } catch (Exception e) {
            falhou("Modelo corrompido: " + e.getMessage(), "treine a IA de novo (opção 3 do INICIAR.bat)");
            return null;
        }
    }

    private void detectarEFiltrar(Projeto p, Modelo m) {
        List<File> imgs = p.imagensAnotadas().isEmpty() ? p.listarImagens() : p.imagensAnotadas();
        if (imgs.isEmpty()) {
            return;
        }
        File arq = imgs.get(0);
        try {
            BufferedImage img = Projeto.lerImagem(arq);
            if (m != null) {
                long t0 = System.currentTimeMillis();
                Detector.Resultado r = Detector.detectar(m, img);
                ok(String.format(Treinamento.PT, "Detecção funcionando (%s: %d região(ões) em %d ms)", arq.getName(),
                        r.regioes(m.limiar).size(), System.currentTimeMillis() - t0));
            }
            List<Etapas.Etapa> etapas = Etapas.calcular(Filtros.paraRgb(img));
            ok("Filtros de processamento de imagem funcionando (" + etapas.size() + " etapas)");
        } catch (Exception e) {
            falhou("Erro ao processar " + arq.getName() + ": " + e, "envie esta tela para o grupo");
        }
    }

    private void janela() {
        if (GraphicsEnvironment.isHeadless()) {
            falhou("Sem interface gráfica disponível", "rode em um computador com tela (não por acesso remoto em modo texto)");
        } else {
            ok("Interface gráfica disponível");
        }
    }
}
