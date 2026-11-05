#!/bin/sh
# ============================================================================
#  Detector de Plástico em Rios — 100% Python (PDI + Visão Computacional)
#  Iniciador para Linux Mint / Ubuntu. A aplicação roda no seu PC (desktop).
#
#  Uso:
#     ./INICIAR.sh                       abre a janela (interface gráfica)
#     ./INICIAR.sh testar                confere se tudo funciona
#     ./INICIAR.sh treinar               treina a IA com as imagens anotadas
#     ./INICIAR.sh detectar <pasta>      detecta plástico numa pasta
#     ./INICIAR.sh filtros <imagem>      salva as etapas do PDI em PNG
#     ./INICIAR.sh importar <arquivos>   importa imagens / .zip
#     ./INICIAR.sh --help
#
#  Na primeira execução o script cria um ambiente Python local (.venv) e
#  instala Pillow + NumPy (+ SciPy). Depois disso, abre direto.
# ============================================================================
set -eu

cd -- "$(dirname -- "$0")"
RAIZ="$(pwd)"
VENV="$RAIZ/.venv"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

# ---- 1. encontrar um Python 3 -------------------------------------------------
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    printf '%s\n' \
        'Python 3 nao encontrado. No Linux Mint/Ubuntu instale pelo terminal:' \
        '  sudo apt update && sudo apt install python3 python3-venv python3-pip python3-tk' \
        'Depois execute ./INICIAR.sh novamente.' >&2
    exit 1
fi

# ---- 2. criar o ambiente virtual na primeira vez ------------------------------
if [ ! -x "$VENV/bin/python" ]; then
    printf '%s\n' 'Primeira execucao: preparando o ambiente Python (Pillow + NumPy)...'
    if ! "$PY" -m venv "$VENV" 2>/dev/null; then
        printf '%s\n' \
            'Falha ao criar o ambiente virtual. Instale o pacote de venv:' \
            '  sudo apt install python3-venv python3-tk' >&2
        exit 1
    fi
    "$VENV/bin/python" -m pip install --upgrade pip >/dev/null 2>&1 || true
    if ! "$VENV/bin/python" -m pip install -r "$RAIZ/requirements.txt"; then
        printf '%s\n' 'Falha ao instalar as dependencias (verifique a internet).' >&2
        exit 1
    fi
    printf '%s\n' 'Ambiente pronto.'
fi

# ---- 3. avisar se faltar o tkinter para a interface grafica -------------------
if [ "$#" -eq 0 ]; then
    if ! "$VENV/bin/python" -c "import tkinter" >/dev/null 2>&1; then
        printf '%s\n' \
            'A interface grafica precisa do tkinter. Instale com:' \
            '  sudo apt install python3-tk' \
            'Enquanto isso, voce pode usar a linha de comando, por exemplo:' \
            '  ./INICIAR.sh testar' >&2
        exit 1
    fi
fi

# ---- 4. rodar -----------------------------------------------------------------
exec "$VENV/bin/python" -m detector_plastico "$@"
