@echo off
rem ============================================================
rem  Detector de Plastico em Rios - APS
rem  Duplo clique aqui e escolha uma opcao no menu.
rem  Tambem aceita: INICIAR.bat abrir | testar | treinar
rem ============================================================
chcp 65001 >nul
title Detector de Plastico em Rios - APS
cd /d "%~dp0" 2>nul

rem ---- caminho longo demais: o Windows nao abre programas numa pasta com mais de ~240 letras
if not "%CD:~240,1%"=="" goto caminholongo

rem ---- rodando de dentro do .zip? (o Windows so extrai este arquivo)
if not exist "dados\imagens" goto dentrodozip

rem ---- precisa do Java
where java >nul 2>nul
if errorlevel 1 goto semjava

rem ---- memoria: 3 GB; se o Java for 32 bits, 1 GB
set "MEM=-Xmx3g"
java -Xmx3g -version >nul 2>nul
if errorlevel 1 set "MEM=-Xmx1g"
set "OPC=%MEM% -Dfile.encoding=UTF-8 -Dstdout.encoding=UTF-8 -Dsun.stdout.encoding=UTF-8"

rem ---- o programa ja vem compilado; compila so se o .jar sumir
if exist DetectorPlastico.jar goto pronto
call compilar.bat
if not exist DetectorPlastico.jar goto erro

:pronto
if /i "%~1"=="abrir" goto abrir
if /i "%~1"=="testar" goto testar_direto
if /i "%~1"=="treinar" goto treinar_direto

:menu
cls
echo.
echo  ==========================================================
echo         DETECTOR DE PLASTICO EM RIOS  -  APS
echo  ==========================================================
echo.
echo     [1]  Abrir o programa
echo     [2]  Testar se esta tudo funcionando
echo     [3]  Treinar a IA de novo (com as anotacoes atuais)
echo     [4]  Detectar plastico em uma pasta de imagens
echo     [5]  Exportar o dataset (YOLO) para o Google Colab
echo     [6]  Ver os filtros passo a passo de uma imagem
echo     [7]  Abrir o guia "Como rodar"
echo     [8]  Assistir ao video explicativo
echo     [0]  Sair
echo.
set "OP="
set /p "OP=  Digite o numero e aperte ENTER: "
rem ---- 3 respostas vazias seguidas (ou fim da entrada) fecham o menu
if not defined VAZIO set "VAZIO=0"
if defined OP (set "VAZIO=0") else (set /a VAZIO+=1)
if %VAZIO% GEQ 3 exit /b 0
if "%OP%"=="1" goto abrir
if "%OP%"=="2" goto testar
if "%OP%"=="3" goto treinar
if "%OP%"=="4" goto detectar
if "%OP%"=="5" goto exportar
if "%OP%"=="6" goto filtros
if "%OP%"=="7" goto guia
if "%OP%"=="8" goto video
if "%OP%"=="0" exit /b 0
goto menu

:abrir
echo.
echo  Abrindo o programa... (a janela aparece em alguns segundos)
where javaw >nul 2>nul
if errorlevel 1 (start "Detector de Plastico" java %OPC% -jar DetectorPlastico.jar) else (start "" javaw %MEM% -jar DetectorPlastico.jar)
if /i "%~1"=="abrir" exit /b 0
timeout /t 3 >nul
goto menu

:testar
cls
java %OPC% -jar DetectorPlastico.jar testar
echo.
pause
goto menu

:testar_direto
java %OPC% -jar DetectorPlastico.jar testar
exit /b %errorlevel%

:treinar
cls
echo  Treinando a IA com todas as imagens anotadas (leva de 10 segundos a 2 minutos)...
echo.
java %OPC% -jar DetectorPlastico.jar treinar
echo.
pause
goto menu

:treinar_direto
java %OPC% -jar DetectorPlastico.jar treinar
exit /b %errorlevel%

:detectar
cls
echo  ARRASTE a pasta com as imagens para esta janela e aperte ENTER.
echo  (ou digite o caminho da pasta)
echo.
set "PASTA="
set /p "PASTA=  Pasta: "
if not defined PASTA goto menu
set "PASTA=%PASTA:"=%"
if not exist "%PASTA%" (
    echo.
    echo  Pasta nao encontrada: %PASTA%
    pause
    goto menu
)
java %OPC% -jar DetectorPlastico.jar detectar "%PASTA%"
echo.
echo  As imagens marcadas e o relatorio.csv estao na pasta "resultados".
start "" "resultados"
pause
goto menu

:exportar
cls
java %OPC% -jar DetectorPlastico.jar exportar
echo.
start "" "resultados"
pause
goto menu

:filtros
cls
echo  ARRASTE uma imagem para esta janela e aperte ENTER.
echo  (ENTER sem nada = usa a imagem de exemplo "images (18).jpg")
echo.
set "IMG="
set /p "IMG=  Imagem: "
if not defined IMG set "IMG=dados\imagens\images (18).jpg"
set "IMG=%IMG:"=%"
java %OPC% -jar DetectorPlastico.jar filtros "%IMG%"
echo.
start "" "resultados"
pause
goto menu

:guia
if exist "COMO_RODAR.pdf" (start "" "COMO_RODAR.pdf") else (start "" "LEIAME.md")
goto menu

:video
if exist "video\Como_usar_Detector_de_Plastico.mp4" (start "" "video\Como_usar_Detector_de_Plastico.mp4") else (echo  Video nao encontrado. & pause)
goto menu

:dentrodozip
echo.
echo  ATENCAO: parece que voce abriu este arquivo DE DENTRO do .zip.
echo.
echo  1. Feche esta janela.
echo  2. Clique com o botao direito no .zip e escolha "Extrair tudo...".
echo  3. Entre na pasta extraida e de dois cliques em INICIAR.bat de novo.
echo.
pause
exit /b 1

:caminholongo
echo.
echo  ATENCAO: o caminho desta pasta e longo demais para o Windows.
echo.
echo  Mova a pasta DetectorPlastico_APS para um lugar mais curto,
echo  por exemplo C:\APS ou a Area de Trabalho, e abra o INICIAR.bat de novo.
echo.
pause
exit /b 1

:semjava
echo.
echo  O Java nao esta instalado neste computador.
echo.
echo  Instale o Java gratuito (Eclipse Temurin, versao 21, Windows x64, .msi),
echo  marcando a opcao "Set JAVA_HOME / Add to PATH" durante a instalacao.
echo  Depois feche esta janela e abra o INICIAR.bat de novo.
echo.
set "SN="
set /p "SN=  Abrir o site de download agora? (S/N): "
if /i "%SN%"=="S" start "" "https://adoptium.net/temurin/releases/?version=21&os=windows&arch=x64&package=jdk"
exit /b 1

:erro
echo.
echo  Nao foi possivel preparar o programa. Envie uma foto desta tela para o grupo.
pause
exit /b 1
