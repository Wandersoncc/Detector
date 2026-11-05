"""Detector de plástico em corpos d'água — 100% Python (Pillow + NumPy).

Pipeline de Processamento Digital de Imagens (PDI) e Visão Computacional:

    PDI (filtros clássicos) -> descritores -> rede neural -> detecção de regiões

Módulos principais:
    filtros       filtros clássicos de PDI (cinza, HSV, Sobel, Canny, Otsu, morfologia)
    descritores   vetor de características por célula (cor, forma/HOG, textura/LBP)
    rede_neural   MLP escrita do zero com NumPy (ReLU + sigmoide, Adam, dropout)
    modelo        normalização + rede + limiar, com salvar/carregar
    treino        treino com aumento de dados e validação cruzada
    detector      aplica o modelo e localiza as regiões com plástico
    lote          detecção em lote (imagens marcadas + relatorio.csv)
    etapas        filtros passo a passo, salvos em PNG para o trabalho escrito
    projeto       pastas do projeto e dataset (imagens + máscaras de anotação)
"""

__version__ = "1.0.0"
