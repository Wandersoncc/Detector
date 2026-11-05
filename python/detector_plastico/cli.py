"""Ponto de entrada por linha de comando.

Sem argumentos abre a janela (GUI). Com argumentos, roda pelo terminal:

    python -m detector_plastico testar
    python -m detector_plastico importar  <arquivos | pastas | .zip>
    python -m detector_plastico treinar
    python -m detector_plastico detectar  <pasta | imagem> [saida]
    python -m detector_plastico filtros   <imagem> [saida]
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from . import autoteste, etapas as etapas_mod, lote, treino
from .detector import detectar
from .modelo import Modelo
from .projeto import Projeto


def _agora():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _uso():
    print(
        "Uso:\n"
        "  python -m detector_plastico                       (abre a janela)\n"
        "  python -m detector_plastico importar <arquivos|pastas|.zip>\n"
        "  python -m detector_plastico testar                (confere se tudo funciona)\n"
        "  python -m detector_plastico treinar\n"
        "  python -m detector_plastico detectar <pasta|imagem> [saida]\n"
        "  python -m detector_plastico filtros <imagem> [saida]   (etapas do PDI em PNG)"
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    projeto = Projeto(Projeto.pasta_padrao())

    if not argv:
        try:
            from .gui import abrir
        except Exception as e:  # noqa: BLE001
            print(f"Não foi possível abrir a interface gráfica: {e}")
            print("Use a linha de comando. Exemplo: python -m detector_plastico testar")
            return 1
        abrir(projeto)
        return 0

    log = print
    cmd = argv[0].lower()

    if cmd == "importar" and len(argv) > 1:
        projeto.importar_imagens([Path(p) for p in argv[1:]], log)

    elif cmd == "testar":
        return 0 if autoteste.rodar(projeto, log) else 1

    elif cmd == "treinar":
        treino.treinar(projeto, log)

    elif cmd == "detectar" and len(argv) > 1:
        modelo_arq = projeto.arquivo_modelo()
        if not modelo_arq.is_file():
            print("Nenhum modelo treinado. Rode 'treinar' antes.")
            return 1
        m = Modelo.carregar(modelo_arq)
        alvo = Path(argv[1])
        imgs = Projeto.listar_imagens_em(alvo, recursivo=False) if alvo.is_dir() else [alvo]
        saida = Path(argv[2]) if len(argv) > 2 else projeto.pasta_resultados / f"lote_{_agora()}"
        n = lote.detectar_pasta(m, imgs, saida, m.limiar, log)
        print(f"{n} imagens analisadas. Resultado em {saida}")

    elif cmd == "filtros" and len(argv) > 1:
        arq = Path(argv[1])
        img = Projeto.ler_imagem(arq)
        h, w = img.shape[:2]
        from . import filtros as fil
        esc = min(1.0, 1024.0 / max(w, h))
        trab = fil.redimensionar(img, round(w * esc), round(h * esc)) if esc < 1 else img
        et = etapas_mod.calcular(trab)
        if projeto.arquivo_modelo().is_file():
            m = Modelo.carregar(projeto.arquivo_modelo())
            et.append(lote.etapa_ia(len(et) + 1, img, detectar(m, img), m.limiar))
        base = arq.stem
        import re
        base = re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_")
        saida = Path(argv[2]) if len(argv) > 2 else projeto.pasta_resultados / f"filtros_{base}"
        etapas_mod.salvar(et, arq.name, saida)
        for e in et:
            print(f"{e.titulo} – {e.info}")
        print(f"Etapas salvas em {saida}")

    else:
        _uso()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
