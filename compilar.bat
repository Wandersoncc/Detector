@echo off
rem Compila o projeto e gera DetectorPlastico.jar (precisa do JDK 8 ou mais novo).
cd /d "%~dp0"
where javac >nul 2>nul
if errorlevel 1 (
    echo [ERRO] javac nao encontrado. Instale o JDK ^(ex.: Eclipse Temurin 21^) e tente de novo.
    pause
    exit /b 1
)
if exist build\classes rmdir /s /q build\classes
mkdir build\classes
echo Compilando...
javac --release 8 -Xlint:-options -encoding UTF-8 -d build\classes -sourcepath src src\br\aps\plastico\*.java src\br\aps\plastico\dados\*.java src\br\aps\plastico\visao\*.java src\br\aps\plastico\ia\*.java src\br\aps\plastico\gui\*.java
if errorlevel 1 (
    echo [ERRO] Falha na compilacao.
    pause
    exit /b 1
)
jar cfe DetectorPlastico.jar br.aps.plastico.App -C build\classes .
if errorlevel 1 (
    echo [ERRO] Falha ao gerar o .jar
    pause
    exit /b 1
)
echo OK: DetectorPlastico.jar gerado.
