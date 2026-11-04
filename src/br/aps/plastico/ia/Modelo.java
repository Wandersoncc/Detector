package br.aps.plastico.ia;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import br.aps.plastico.visao.Descritores;

/** O "cérebro" treinado: normalização dos descritores + rede neural + limiar escolhido. */
public final class Modelo {

    /** As imagens são reescaladas para que o maior lado tenha este tamanho. */
    public static final int LADO_BASE = 640;
    private static final int MAGICO = 0x504C4153; // "PLAS"
    private static final int VERSAO = 2;

    public int ladoBase = LADO_BASE;
    public float[] media, desvio;
    public RedeNeural rede;
    public float limiar = 0.5f;
    public long dataTreino;
    public int nImagens;
    /** Medidas da validação cruzada (-1 quando não houve validação). */
    public float f1 = -1, precisao = -1, revocacao = -1, iou = -1, f1Classico = -1;
    /** F1 só nas imagens anotadas pelo grupo e só nas vindas de datasets (-1 quando não há). */
    public float f1Proprias = -1, f1Importadas = -1;

    /** Probabilidade de plástico para o descritor em {@code x[off..]}. */
    public float prever(float[] x, int off) {
        float[] z = new float[media.length];
        Treinamento.normalizar(x, off, media, desvio, z);
        return rede.prever(z);
    }

    public void salvar(File arquivo) throws IOException {
        File tmp = new File(arquivo.getPath() + ".tmp");
        DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(tmp)));
        try {
            out.writeInt(MAGICO);
            out.writeInt(VERSAO);
            out.writeInt(ladoBase);
            out.writeInt(media.length);
            for (int i = 0; i < media.length; i++) {
                out.writeFloat(media[i]);
                out.writeFloat(desvio[i]);
            }
            out.writeFloat(limiar);
            out.writeLong(dataTreino);
            out.writeInt(nImagens);
            out.writeFloat(f1);
            out.writeFloat(precisao);
            out.writeFloat(revocacao);
            out.writeFloat(iou);
            out.writeFloat(f1Classico);
            out.writeFloat(f1Proprias);
            out.writeFloat(f1Importadas);
            rede.salvar(out);
        } finally {
            out.close();
        }
        if (arquivo.exists() && !arquivo.delete()) {
            throw new IOException("Não foi possível substituir " + arquivo);
        }
        if (!tmp.renameTo(arquivo)) {
            throw new IOException("Não foi possível gravar " + arquivo);
        }
    }

    public static Modelo carregar(File arquivo) throws IOException {
        DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(arquivo)));
        try {
            if (in.readInt() != MAGICO) {
                throw new IOException("Arquivo de modelo inválido: " + arquivo.getName());
            }
            int versao = in.readInt();
            if (versao < 1 || versao > VERSAO) {
                throw new IOException("Versão de modelo não suportada: " + versao);
            }
            Modelo m = new Modelo();
            m.ladoBase = in.readInt();
            int n = in.readInt();
            if (n != Descritores.TAMANHO) {
                throw new IOException("Modelo feito com outra versão dos descritores; treine novamente.");
            }
            m.media = new float[n];
            m.desvio = new float[n];
            for (int i = 0; i < n; i++) {
                m.media[i] = in.readFloat();
                m.desvio[i] = in.readFloat();
            }
            m.limiar = in.readFloat();
            m.dataTreino = in.readLong();
            m.nImagens = in.readInt();
            m.f1 = in.readFloat();
            m.precisao = in.readFloat();
            m.revocacao = in.readFloat();
            m.iou = in.readFloat();
            m.f1Classico = in.readFloat();
            if (versao >= 2) {
                m.f1Proprias = in.readFloat();
                m.f1Importadas = in.readFloat();
            }
            m.rede = RedeNeural.carregar(in);
            return m;
        } finally {
            in.close();
        }
    }
}
