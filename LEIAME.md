# Detector de Plástico em Rios – Visão Computacional em Java

Sistema que **aprende com exemplos** a reconhecer plástico (garrafas PET, sacolas, isopor, embalagens)
em rios poluídos, mangues, praias e nascentes. Junta os filtros clássicos (HSV e P&B de contornos) com
uma **rede neural treinada pelas imagens que vocês anotam**: quanto mais imagens, melhor ela fica.

100% Java, sem bibliotecas externas. Roda em Windows, Linux e Mac com Java 8 ou mais novo.

## Como abrir

### Linux Mint / Ubuntu

O arquivo `INICIAR.bat` é exclusivo do Windows. No Linux, abra um terminal nesta pasta
(botão direito na pasta > **Abrir no terminal**) e execute:

```sh
sh INICIAR.sh
```

O iniciador usa o Java do sistema, o definido em `JAVA_HOME` ou a cópia local em
`~/.local/share/detector-plastico/java`. Esta máquina foi preparada com essa cópia local,
sem precisar de senha de administrador. Se nenhum deles estiver disponível, instale o Java:

```sh
sudo apt update && sudo apt install default-jre
```

Para conferir a instalação e os dados do projeto, execute `sh INICIAR.sh testar`.
Veja outros comandos com `sh INICIAR.sh --help`.

### Windows

Dê dois cliques em **`INICIAR.bat`** e escolha no menu:

| Opção | O que faz |
|---|---|
| 1 | Abre o programa |
| 2 | Testa se está tudo funcionando |
| 3 | Treina a IA |
| 4 | Detecta plástico numa pasta |
| 5 | Exporta o dataset YOLO |
| 6 | Mostra os filtros passo a passo |
| 7 | Abre o guia |
| 8 | Abre o vídeo |

Passo a passo com figuras: **`COMO_RODAR.pdf`**. Vídeo explicativo narrado: pasta **`video/`**.

- Precisa de Java 8 ou mais novo. Se faltar, o `INICIAR.bat` avisa e abre o site do Java gratuito.
- Extraia o .zip antes de usar; se você rodar de dentro do .zip, o `.bat` avisa.
- Se você mudar o código, rode **`compilar.bat`** para gerar o `DetectorPlastico.jar` de novo.
- Também funciona direto pelo terminal: `java -jar DetectorPlastico.jar`.

## Como usar (o ciclo de treino)

1. **Importar imagens**: arquivos soltos, pastas ou um `.zip`. Imagens repetidas são ignoradas
   automaticamente (o zip enviado tinha 2 duplicadas).
2. **Anotar** (ensinar a IA): pinte de verde o que é plástico.
   - **Plástico (B)**: pincel. O botão direito apaga.
   - **Varinha (W)**: clique num objeto para preencher a área dentro do contorno do filtro P&B.
     Funciona melhor em objetos opacos (tampas, potes, sacolas coloridas).
   - **Retângulo (R)**: preenche áreas grandes, como montes de garrafas.
   - **Ignorar (I)**, em amarelo: onde nem você tem certeza. A IA não usa essas áreas.
   - **Ctrl+S** salva. Salvar sem pintar nada marca a imagem como "sem plástico", o que também
     ensina a IA (o que NÃO é plástico).
3. **Treinar IA**: leva segundos. O sistema mede a **eficiência (F1)** com validação cruzada: cada
   imagem é testada por um modelo que **não a viu** no treino. Ele também compara com o filtro HSV
   clássico nas mesmas imagens.
4. **Melhorar** (o que o Otávio sugeriu: treinar onde ele não reconhece direito):
   - Filtro **"Piores na validação"**: mostra as imagens em que a IA mais errou.
   - **"Sugerir próximas imagens p/ anotar"**: lista as imagens novas em que a IA tem mais dúvida.
   - **"Pré-anotar com a IA"**: a IA pinta sozinha e você só corrige e salva. Acelera muito.
   - Treine de novo e veja a curva subir em **"Relatório e evolução da eficiência"**.
5. **Detectar em lote**: analisa uma pasta inteira e gera, em `resultados/lote_<data>/`, as imagens
   marcadas e um `relatorio.csv` (abre no Excel): tem plástico?, quantas regiões, % da imagem.
6. **Filtros passo a passo (PDI)**, na barra de cima: mostra na imagem atual as 13 etapas do
   processamento, cada uma com explicação e um número de resultado. Clique numa etapa para ampliar.
   O botão **"Salvar as etapas em PNG"** grava as figuras e um `etapas.txt` em
   `resultados/filtros_<imagem>/`, prontos para o capítulo de filtragem do trabalho escrito.
7. **Exportar dataset (YOLO)**, no painel da esquerda: gera em `resultados/dataset_yolo_<data>/` o
   dataset preparado para treinar outros modelos de visão computacional:
   - imagens com até 640 px;
   - rótulos YOLO (uma caixa por objeto);
   - máscaras PNG para segmentação;
   - `data.yaml`, separação 80% treino / 20% validação e um `.zip` para enviar ao Google Colab.

   O `LEIAME.txt` de dentro explica como treinar um YOLO com ele.

## Situação atual (já vem treinado)

- **Imagens próprias:** 35, vindas do zip do grupo (37 menos 2 repetidas).
  - **23 já anotadas**, com polígonos aproximados. **Revisem** principalmente as que aparecem em
    "Piores na validação".
  - **12 sem anotação**, para praticar com "Pré-anotar com a IA": 9, 11, 18, 20, 22, 23, 24, 25, 26,
    29, 31 e 32.
- **400 fotos de um dataset público** de rios turvos (veja *Créditos*), com nomes `mendeley_...`. O
  lixo flutuante virou plástico e a vegetação virou fundo. A lista delas fica em
  `dados/importadas_de_datasets.txt`, e o sistema mostra a eficiência separada por origem.

### Resultados medidos

Medido em células de 16 px. O "teste externo" usa 158 fotos de vídeos que **nunca** entram no treino.

| Treino | F1 nas imagens próprias (validação cruzada) | F1 no teste externo | Filtro HSV no teste externo |
|---|---|---|---|
| 6 imagens próprias  | 17,2% | – | – |
| 10 imagens próprias | 42,0% | – | – |
| 15 imagens próprias | 52,7% | – | – |
| 23 imagens próprias | **53,8%** | 13,5% | 4,4% |
| 23 próprias + 400 do dataset (modelo atual) | 46,9% | **29,8%** | 4,4% |

O que isso mostra:

- **Mais imagens do mesmo tipo melhoram a IA:** foi de 17% para 54% indo de 6 para 23 imagens.
- **Imagens de outro cenário generalizam melhor, mas com um custo:**
  - as 400 fotos de rio turvo (objetos pequenos e espalhados, anotados com caixas) **dobraram** o
    acerto em vídeos nunca vistos;
  - ao mesmo tempo, **custaram ~7 pontos** nas fotos de lixo acumulado do grupo.

  Uma rede pequena com descritores feitos à mão não consegue dominar os dois cenários ao mesmo tempo
  com poucos exemplos. Para "qualquer cenário" são precisos mais dados anotados de forma consistente,
  e depois um modelo mais forte (CNN/YOLO).
- O filtro HSV de faixas fixas fica entre 2% e 27%, conforme o cenário. Isso mostra por que o
  aprendizado é necessário.

Se quiserem voltar ao modelo só com as imagens próprias, use o filtro "Anotadas" e "Remover anotação"
nas fotos `mendeley_...`, ou apague as anotações `mendeley_*.png` em `dados/anotacoes/`. Depois,
treinem de novo.

**Já prontos em `resultados/`:**
- `dataset_yolo_20261101_211953/` e o `.zip` dela: as 423 imagens anotadas e 824 objetos
  (treino 338 / validação 85);
- `filtros_images_18/`: as 13 etapas aplicadas na imagem 18.

## Relação com o manual da APS

| Capítulo do manual | Onde está no sistema |
|---|---|
| 2.e.i Aquisição | Importar imagens / .zip, importar dataset rotulado, redução automática de fotos grandes |
| 2.e.ii Manipulação | Redimensionamento, espelhamento, tons de cinza, HSV, brilho/contraste (aumento de dados) |
| 2.e.iii Filtragem | Filtros passo a passo: equalização, mediana, gaussiano, Sobel, Canny |
| 2.e.iv Segmentação | Otsu, faixas HSV, morfologia, componentes conexos, varinha por contorno, segmentação pela IA |
| 2.e.v / 3.b Modelo de visão computacional | Descritores (cor, HOG, LBP) + rede neural; dataset exportado para YOLO |
| 3.a Filtros aplicados e analisados | Figuras e números de cada etapa; comparação filtro HSV × IA pelo F1 |
| 4.c / 4.d Testes e resultados | Validação cruzada, teste externo, relatório por imagem, gráfico de evolução |

## Usar datasets da internet (imagens em lote, já rotuladas)

**"Importar dataset rotulado..."** aceita pastas nos formatos:

- **YOLO**: caixas ou polígonos (`images/` + `labels/` + `data.yaml`). É o formato "YOLOv5/v8" do
  Roboflow e de muitos datasets do Kaggle.
- **Pascal VOC**: arquivos `.xml`.

O sistema lista as classes do dataset e você marca quais são plástico. As outras viram fundo (exemplos
de "não é plástico"). Caixas viram uma máscara com o miolo como plástico e a borda como "ignorar",
porque a caixa sempre pega um pouco de fundo nos cantos.

Fotos com mais de 1280 px são reduzidas na importação, já que a IA trabalha em 640 px. Isso mantém a
pasta do projeto leve.

Dica: imagens `.webp` (comuns em sites) não são lidas pelo Java. Converta para JPG antes.

## Como funciona (para a apresentação da APS)

**Processamento digital de imagens** (todos os filtros foram escritos do zero, sem OpenCV):
- *Realce*: equalização de histograma.
- *Suavização*: filtro de mediana 5×5 e gaussiano 5×5.
- *Bordas*: Sobel e Canny (supressão de não-máximos + histerese), que é o "P&B contornos".
- *Limiarização*: Otsu automático.
- *Cor*: canais HSV e segmentação por faixas fixas. Esse filtro só sabe que "isso tem H=107, S=66".
- *Morfologia*: erosão, dilatação, abertura e fechamento.
- *Regiões*: componentes conexos.

**Visão computacional com aprendizado (o que faltava):**
1. A imagem é dividida em células de 16x16 px (a 640 px de largura).
2. Para cada célula é calculado um **descritor de 163 números**, em três janelas
   (16 px = o objeto, 48 px = a vizinhança, 128 px = a cena):
   - **cor**: histograma de 20 faixas HSV, médias e desvios de R, G, B, S e V;
   - **forma e contorno**: histograma de orientação dos gradientes (tipo HOG) e densidade de bordas;
   - **textura**: LBP (padrões binários locais);
   - **brilho especular**: o reflexo típico do plástico;
   - **contraste** do objeto com o entorno.
3. Uma **rede neural** (163 → 64 → 32 → 1, ReLU + sigmoide), escrita do zero com retropropagação,
   otimizador Adam, dropout e decaimento de pesos, aprende a dar a probabilidade de "plástico".
4. **Aumento de dados**: cada imagem anotada gera 4 versões (espelhada, em outra escala e com
   brilho/contraste/saturação diferentes). As amostras de cada imagem são proporcionais ao plástico
   anotado nela, numa proporção de cerca de 2 de fundo para 1 de plástico.
5. A detecção junta as células vizinhas acima do limiar em **regiões** com % de confiança.

**Medida de eficiência**:
- *Precisão*: do que a IA marcou, quanto era plástico.
- *Revocação*: do plástico que existia, quanto ela achou.
- *F1*: média harmônica das duas.
- O **limiar** é escolhido automaticamente para maximizar o F1 e pode ser ajustado no controle
  deslizante.

## Linha de comando (opcional)

```
java -jar DetectorPlastico.jar testar                             (confere se tudo funciona)
java -jar DetectorPlastico.jar importar  <arquivos | pastas | .zip>
java -jar DetectorPlastico.jar importar-dataset <pasta YOLO/VOC> [classes de plástico, ex.: 0,1,2]
java -jar DetectorPlastico.jar treinar
java -jar DetectorPlastico.jar detectar <pasta ou imagem> [pasta de saída]
java -jar DetectorPlastico.jar exportar [pasta de saída]           (dataset YOLO + máscaras + .zip)
java -jar DetectorPlastico.jar filtros <imagem> [pasta de saída]   (as 13 etapas do PDI em PNG)
```

## Pastas

```
INICIAR.bat              menu com tudo (abrir, testar, treinar, detectar, exportar...)
COMO_RODAR.pdf           guia passo a passo com figuras (fonte em doc/)
video/                   vídeo explicativo narrado
compilar.bat / DetectorPlastico.jar
src/br/aps/plastico/
    App.java                 ponto de entrada (janela ou linha de comando)
    Autoteste.java           o teste "está tudo OK?" (opção 2 do menu)
    visao/Filtros.java       filtros: equalização, mediana, gaussiano, Sobel, Canny, Otsu, HSV, morfologia, varinha
    visao/Etapas.java        os filtros passo a passo, com explicação e números
    visao/Descritores.java   descritores de cor, forma e textura
    ia/RedeNeural.java       a rede neural (do zero)
    ia/Treinamento.java      treino, aumento de dados, validação cruzada
    ia/Detector.java         aplica o modelo e encontra as regiões
    ia/Lote.java             detecção em lote e sugestões de imagens
    dados/Projeto.java       imagens, anotações, importação
    dados/ImportadorDataset.java   importa YOLO e Pascal VOC
    dados/ExportadorYolo.java      exporta o dataset preparado (YOLO + máscaras)
    gui/                     a interface (Swing)
dados/imagens/      as imagens
dados/anotacoes/    máscaras PNG (branco = plástico, cinza = ignorar, preto = fundo)
modelos/            modelo treinado, histórico de eficiência, desempenho por imagem
resultados/         detecção em lote, filtros passo a passo e datasets exportados
```

Para treinar em outra máquina, copie a pasta inteira. As anotações e o modelo vão junto.

## Créditos (citar na bibliografia)

As fotos `mendeley_*` são do dataset abaixo, sob licença **CC BY 4.0**, que exige citar a fonte. As
imagens foram reduzidas para 1280 px e as caixas foram convertidas em máscaras.

> RADAM, I. F. et al. *Floating Waste and Aquatic Vegetation Image Dataset from Turbid Rivers*.
> Mendeley Data, v. 2, 2026. DOI: 10.17632/j26w4m645z.2. Licença CC BY 4.0.
