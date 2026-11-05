# Detector de Plástico em Rios — versão Python (PDI + Visão Computacional)

Porte para **Python** do detector de plástico em corpos d'água, mantendo o mesmo
pipeline do projeto Java original, mas com **Pillow + NumPy** no Processamento
Digital de Imagens (PDI):

```
PDI (filtros clássicos)  ->  descritores  ->  rede neural  ->  detecção de regiões
```

O sistema **aprende com exemplos**: você pinta de verde o que é plástico, a rede
neural treina com essas anotações e passa a marcar plástico em imagens novas.
Ele junta os filtros clássicos (HSV e contornos P&B) com uma rede neural treinada
pelas imagens que você anota — quanto mais imagens, melhor ela fica.

As imagens, anotações e resultados ficam na **raiz do projeto** (`dados/`,
`modelos/`, `resultados/`), **compartilhados com a versão Java**.

---

## Como rodar (Linux Mint / Ubuntu)

Abra um terminal nesta pasta (`python/`) e execute:

```sh
sh INICIAR.sh
```

Na primeira vez o script cria um ambiente Python local (`.venv`) e instala as
dependências (Pillow + NumPy + SciPy). Depois abre a janela.

Se faltar algum pacote do sistema, instale:

```sh
sudo apt update && sudo apt install python3 python3-venv python3-pip python3-tk
```

> A interface gráfica usa **tkinter** (`python3-tk`). Sem ele, os comandos de
> terminal (abaixo) continuam funcionando normalmente.

### Comandos de terminal

```sh
./INICIAR.sh                      # abre a janela
./INICIAR.sh testar               # confere se tudo funciona
./INICIAR.sh treinar              # treina a IA com as imagens anotadas
./INICIAR.sh detectar <pasta>     # detecta plástico numa pasta (gera imagens + relatorio.csv)
./INICIAR.sh filtros <imagem>     # salva as 12/13 etapas do PDI em PNG
./INICIAR.sh importar <arquivos>  # importa imagens soltas, pastas ou .zip
```

Também funciona sem o script, com o ambiente ativado:

```sh
python -m detector_plastico testar
```

---

## Como usar (o ciclo de treino)

1. **Importar imagens**: arquivos soltos, pastas ou `.zip`. Imagens repetidas
   (mesmo conteúdo) são ignoradas automaticamente.
2. **Anotar** (ensinar a IA): pinte de verde o que é plástico.
   - **Plástico (B)**: pincel. O botão direito apaga.
   - **Varinha (W)**: clique num objeto para preencher a área dentro do contorno
     detectado pelo filtro P&B.
   - **Retângulo (R)**: preenche áreas grandes.
   - **Ignorar (I)**, em amarelo: onde nem você tem certeza; a IA não usa.
   - **Ctrl+S** salva. Salvar sem pintar marca a imagem como "sem plástico", o
     que também ensina a IA (o que NÃO é plástico).
3. **Treinar IA**: mede a **eficiência (F1)** com validação cruzada — cada imagem
   é testada por um modelo que **não a viu** no treino — e compara com o filtro
   HSV clássico nas mesmas imagens.
4. **Pré-anotar com a IA**: a IA pinta sozinha; você só corrige e salva. Acelera
   muito a anotação.
5. **Detectar em lote**: analisa uma pasta inteira e gera, em
   `resultados/lote_<data>/`, as imagens marcadas (`*_ia.jpg`) e um
   `relatorio.csv` (tem plástico?, quantas regiões, % da imagem, confiança).
6. **Filtros passo a passo (PDI)**: mostra na imagem atual as etapas do
   processamento, cada uma com explicação e um número medido. O botão
   **"Salvar as etapas em PNG"** grava as figuras e um `etapas.txt` em
   `resultados/filtros_<imagem>/`, prontos para o capítulo de filtragem do
   trabalho escrito.

---

## Como funciona

**Processamento digital de imagens** (todos os filtros escritos com Pillow +
NumPy, sem OpenCV):

- *Realce*: equalização de histograma.
- *Suavização*: filtro de mediana 5×5 e gaussiano 5×5 (separável, núcleo 1-4-6-4-1).
- *Bordas*: Sobel e Canny (supressão de não-máximos + histerese) — o "P&B contornos".
- *Limiarização*: Otsu automático.
- *Cor*: canais HSV e segmentação por faixas fixas.
- *Morfologia*: erosão, dilatação, abertura e fechamento.
- *Regiões*: componentes conexos (vizinhança-8).

**Visão computacional com aprendizado:**

1. A imagem é dividida em células de 16×16 px (sobre a imagem reescalada para 640 px).
2. Para cada célula é calculado um **descritor de 163 números**, em três janelas
   (16 px = o objeto, 48 px = a vizinhança, 128 px = a cena):
   - **cor**: histograma de 20 faixas HSV, médias e desvios de R, G, B, S e V;
   - **forma e contorno**: histograma de orientação dos gradientes (tipo HOG) e densidade de bordas;
   - **textura**: LBP (padrões binários locais);
   - **brilho especular**: o reflexo típico do plástico;
   - **contraste** do objeto com o entorno.
3. Uma **rede neural** (163 → 64 → 32 → 1, ReLU + sigmoide), escrita do zero com
   NumPy — retropropagação, otimizador Adam, dropout e decaimento de pesos —
   aprende a dar a probabilidade de "plástico".
4. **Aumento de dados**: cada imagem anotada gera 4 versões (espelhada, em outra
   escala e com brilho/contraste/saturação diferentes), com cerca de 2 amostras
   de fundo para 1 de plástico.
5. A detecção junta as células vizinhas acima do limiar em **regiões** com % de
   confiança.

**Medida de eficiência**: precisão, revocação e **F1** (média harmônica). O
**limiar** é escolhido automaticamente para maximizar o F1 e pode ser ajustado
no controle deslizante da janela.

---

## Pastas

```
python/
  INICIAR.sh                iniciador para Linux Mint (cria o .venv e roda)
  requirements.txt          dependências (Pillow, NumPy, SciPy)
  detector_plastico/
    filtros.py              filtros de PDI (cinza, HSV, Sobel, Canny, Otsu, morfologia, varinha)
    descritores.py          descritor de 163 números (cor, forma/HOG, textura/LBP)
    rede_neural.py          a rede neural escrita do zero (NumPy)
    modelo.py               normalização + rede + limiar (salvar/carregar .npz)
    treino.py               treino, aumento de dados, validação cruzada
    detector.py             aplica o modelo e localiza as regiões
    lote.py                 detecção em lote (imagens marcadas + relatorio.csv)
    etapas.py               os filtros passo a passo, com explicação e números
    projeto.py              imagens, máscaras de anotação, importação
    metricas.py             precisão, revocação, F1, IoU
    autoteste.py            o "está tudo OK?"
    gui.py                  a interface (Tkinter)
    cli.py                  a linha de comando

(na raiz do projeto, compartilhado com a versão Java)
  dados/imagens/      as imagens
  dados/anotacoes/    máscaras PNG (branco = plástico, cinza = ignorar, preto = fundo)
  modelos/            modelo treinado (modelo_plastico.npz) e CSVs de eficiência
  resultados/         detecção em lote e filtros passo a passo
```

## Diferenças em relação à versão Java

- PDI com **Pillow + NumPy** (vetorizado) em vez dos laços manuais em Java.
- A rede neural é a mesma arquitetura (163→64→32→1), treinada por **mini-lote**
  vetorizado — mais rápida, mesma matemática (Adam, dropout, weight decay).
- Modelo salvo em **`.npz`** (NumPy), não no binário próprio do Java.
- Os componentes conexos usam SciPy quando disponível, com **fallback** em Python
  puro se o SciPy não estiver instalado.
- As anotações (`dados/anotacoes/*.png`) são **as mesmas** das duas versões.

## Créditos (citar na bibliografia)

As fotos `mendeley_*` são do dataset abaixo, sob licença **CC BY 4.0**:

> RADAM, I. F. et al. *Floating Waste and Aquatic Vegetation Image Dataset from
> Turbid Rivers*. Mendeley Data, v. 2, 2026. DOI: 10.17632/j26w4m645z.2.
> Licença CC BY 4.0.
