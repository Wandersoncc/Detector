#!/bin/sh
set -eu

cd -- "$(dirname -- "$0")"

case "${1:-}" in
    -h|--help)
        printf '%s\n' \
            'Uso: ./INICIAR.sh [comando] [argumentos]' \
            '  (sem comando) ou abrir: abre a janela' \
            '  testar: verifica imagens, modelo, filtros e interface grafica' \
            '  treinar: treina novamente com as anotacoes atuais' \
            '  detectar <pasta|imagem> [saida]: detecta plastico' \
            '  filtros <imagem> [saida]: salva as etapas do PDI' \
            '  exportar [saida]: exporta o dataset YOLO'
        exit 0
        ;;
    abrir)
        shift
        ;;
esac

if command -v java >/dev/null 2>&1; then
    java_cmd=java
elif [ -n "${JAVA_HOME:-}" ] && [ -x "$JAVA_HOME/bin/java" ]; then
    java_cmd="$JAVA_HOME/bin/java"
elif [ -x "$HOME/.local/share/detector-plastico/java/bin/java" ]; then
    java_cmd="$HOME/.local/share/detector-plastico/java/bin/java"
else
    printf '%s\n' \
        'Java nao encontrado. No Linux Mint/Ubuntu, instale pelo terminal:' \
        '  sudo apt update && sudo apt install default-jre' \
        'Depois execute ./INICIAR.sh novamente.' >&2
    exit 1
fi

if [ ! -f DetectorPlastico.jar ]; then
    printf '%s\n' 'DetectorPlastico.jar nao encontrado. Extraia o projeto completo antes de executar.' >&2
    exit 1
fi

exec "$java_cmd" -Xmx3g -Dfile.encoding=UTF-8 -jar DetectorPlastico.jar "$@"
