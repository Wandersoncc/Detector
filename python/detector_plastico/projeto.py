"""Pastas do projeto e o conjunto de dados (imagens + anotações).

    dados/imagens/     imagens usadas no treino
    dados/anotacoes/   uma máscara PNG por imagem: preto = fundo, branco = plástico,
                       cinza = ignorar
    modelos/           modelo treinado, histórico de eficiência, desempenho por imagem
    resultados/        saídas da detecção em lote, filtros e datasets exportados

As máscaras são compatíveis com o projeto Java: PNG em tons de cinza onde
255 = plástico, 128 = ignorar, 0 = fundo.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from . import filtros

FUNDO = 0
PLASTICO = 255
IGNORAR = 128

EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
LADO_MAXIMO = 1280  # maior lado guardado no projeto (a IA trabalha em 640 px)


class Projeto:
    def __init__(self, raiz):
        self.raiz = Path(raiz).resolve()
        self.pasta_imagens = self.raiz / "dados" / "imagens"
        self.pasta_anotacoes = self.raiz / "dados" / "anotacoes"
        self.pasta_modelos = self.raiz / "modelos"
        self.pasta_resultados = self.raiz / "resultados"
        for d in (self.pasta_imagens, self.pasta_anotacoes, self.pasta_modelos, self.pasta_resultados):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- descoberta

    @staticmethod
    def pasta_padrao() -> Path:
        p = os.environ.get("DETECTOR_PROJETO")
        if p:
            return Path(p)
        # por padrão, a raiz do repositório (pai da pasta python/)
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def eh_imagem(nome: str) -> bool:
        return nome.lower().endswith(EXTENSOES)

    def listar_imagens(self):
        return self.listar_imagens_em(self.pasta_imagens, recursivo=False)

    @staticmethod
    def listar_imagens_em(pasta, recursivo=False):
        pasta = Path(pasta)
        out = []
        it = pasta.rglob("*") if recursivo else pasta.glob("*")
        for f in it:
            if f.is_file() and Projeto.eh_imagem(f.name):
                out.append(f)
        out.sort(key=lambda f: _chave_natural(f.name))
        return out

    def imagens_anotadas(self):
        return [f for f in self.listar_imagens() if self.anotada(f)]

    def arquivo_mascara(self, imagem) -> Path:
        return self.pasta_anotacoes / (Path(imagem).name + ".png")

    def anotada(self, imagem) -> bool:
        return self.arquivo_mascara(imagem).is_file()

    # --------------------------------------------------------------- imagens

    @staticmethod
    def ler_imagem(f) -> np.ndarray:
        """Lê qualquer imagem suportada e devolve RGB float32 (H, W, 3)."""
        try:
            return filtros.abrir_rgb(f)
        except Exception as e:  # noqa: BLE001
            raise IOError(f"Formato não suportado: {Path(f).name} (use JPG, PNG ou BMP)") from e

    # --------------------------------------------------------------- máscaras

    def carregar_mascara(self, imagem, w: int, h: int):
        """Máscara (h, w) uint8 com {0,128,255}, ou None se ainda não anotada."""
        f = self.arquivo_mascara(imagem)
        if not f.is_file():
            return None
        m = Image.open(f).convert("L")
        arr = np.asarray(m, dtype=np.uint8)
        if (arr.shape[1], arr.shape[0]) != (w, h):
            # vizinho mais próximo, para não misturar rótulos
            arr = np.asarray(Image.fromarray(arr).resize((w, h), Image.NEAREST), dtype=np.uint8)
        out = np.where(arr >= 192, PLASTICO, np.where(arr >= 64, IGNORAR, FUNDO)).astype(np.uint8)
        return out

    def salvar_mascara(self, imagem, mascara: np.ndarray):
        Image.fromarray(mascara.astype(np.uint8), mode="L").save(self.arquivo_mascara(imagem))

    def remover_mascara(self, imagem):
        f = self.arquivo_mascara(imagem)
        if f.is_file():
            f.unlink()

    @staticmethod
    def redimensionar_mascara(m: np.ndarray, w: int, h: int) -> np.ndarray:
        return np.asarray(Image.fromarray(m.astype(np.uint8)).resize((w, h), Image.NEAREST), dtype=np.uint8)

    @staticmethod
    def espelhar_mascara(m: np.ndarray) -> np.ndarray:
        return m[:, ::-1].copy()

    # ----------------------------------------------------------- arquivos modelo

    def arquivo_modelo(self) -> Path:
        return self.pasta_modelos / "modelo_plastico.npz"

    def arquivo_historico(self) -> Path:
        # nome próprio da versão Python, para não sobrescrever o CSV da versão Java
        return self.pasta_modelos / "historico_eficiencia_py.csv"

    def arquivo_desempenho(self) -> Path:
        return self.pasta_modelos / "desempenho_por_imagem_py.csv"

    def arquivo_importadas(self) -> Path:
        return self.pasta_imagens.parent / "importadas_de_datasets.txt"

    def imagens_de_datasets(self):
        f = self.arquivo_importadas()
        s = set()
        if f.is_file():
            for l in f.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    s.add(l.strip())
        return s

    def registrar_importadas(self, nomes):
        if not nomes:
            return
        with open(self.arquivo_importadas(), "a", encoding="utf-8") as fh:
            for n in nomes:
                fh.write(n + "\n")

    # -------------------------------------------------------------------- CSV

    @staticmethod
    def escrever_csv(caminho, cabecalho, linhas, anexar=False):
        """Grava CSV ';' em UTF-8 com BOM (o Excel reconhece os acentos)."""
        caminho = Path(caminho)
        novo = (not caminho.is_file()) or (not anexar)
        modo = "a" if anexar else "w"
        with open(caminho, modo, encoding="utf-8-sig" if novo else "utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter=";")
            if novo and cabecalho:
                w.writerow(cabecalho)
            for l in linhas:
                w.writerow(l)

    # ------------------------------------------------------------- importação

    def importar_imagens(self, fontes, log):
        """Copia imagens para dados/imagens. Aceita arquivos, pastas e .zip.
        Imagens repetidas (mesmo conteúdo) são ignoradas."""
        hashes = self.hashes_existentes()
        cont = [0, 0, 0]  # novas, repetidas, inválidas
        for f in fontes:
            self._importar_fonte(Path(f), hashes, cont, log)
        log(f"Importação concluída: {cont[0]} novas, {cont[1]} repetidas ignoradas, {cont[2]} inválidas.")
        return cont[0]

    def _importar_fonte(self, f: Path, hashes, cont, log):
        nome = f.name.lower()
        if f.is_dir():
            for g in sorted(f.iterdir()):
                self._importar_fonte(g, hashes, cont, log)
        elif nome.endswith(".zip"):
            with zipfile.ZipFile(f) as z:
                for info in z.infolist():
                    n = info.filename.replace("\\", "/")
                    n = n[n.rfind("/") + 1:]
                    if info.is_dir() or n.startswith("._") or "__MACOSX" in info.filename:
                        continue
                    if self.eh_imagem(n):
                        self._registrar(n, z.read(info), hashes, cont, log)
                    elif n.lower().endswith(".webp"):
                        log(f"  ignorada (WEBP não suportado, converta para JPG): {n}")
                        cont[2] += 1
        elif self.eh_imagem(nome):
            self._registrar(f.name, f.read_bytes(), hashes, cont, log)
        elif nome.endswith(".webp"):
            log(f"  ignorada (WEBP não suportado, converta para JPG): {f.name}")
            cont[2] += 1

    def _registrar(self, nome, dados, hashes, cont, log):
        prep = self.preparar(nome, dados)
        if prep is None:
            log(f"  ignorada (não é uma imagem legível): {nome}")
            cont[2] += 1
            return
        novo_nome, bytes_img = prep
        if self.gravar_imagem_nova(novo_nome, bytes_img, hashes) is None:
            cont[1] += 1
        else:
            cont[0] += 1

    @staticmethod
    def preparar(nome, dados):
        """Valida e reduz fotos grandes; devolve (nome, bytes) ou None."""
        try:
            img = Image.open(io.BytesIO(dados))
            img.load()
        except Exception:  # noqa: BLE001
            return None
        w, h = img.size
        if max(w, h) <= LADO_MAXIMO:
            return nome, dados
        esc = LADO_MAXIMO / float(max(w, h))
        nw, nh = max(1, round(w * esc)), max(1, round(h * esc))
        menor = img.convert("RGB").resize((nw, nh), Image.BILINEAR)
        buf = io.BytesIO()
        menor.save(buf, format="JPEG", quality=92)
        base = re.sub(r"\.[^.]+$", "", nome)
        return base + ".jpg", buf.getvalue()

    def gravar_imagem_nova(self, nome, dados, hashes):
        h = _md5(dados)
        if h in hashes:
            return None
        hashes.add(h)
        limpo = re.sub(r'[\\/:*?"<>|]', "_", nome)
        p = limpo.rfind(".")
        base = limpo[:p] if p > 0 else limpo
        ext = limpo[p:] if p > 0 else ".jpg"
        destino = self.pasta_imagens / limpo
        i = 2
        while destino.exists():
            destino = self.pasta_imagens / f"{base}_{i}{ext}"
            i += 1
        destino.write_bytes(dados)
        return destino

    def hashes_existentes(self):
        return {_md5(f.read_bytes()) for f in self.listar_imagens()}


def _md5(dados: bytes) -> str:
    return hashlib.md5(dados).hexdigest()


def _chave_natural(nome: str):
    """Ordena 'images (2)' antes de 'images (10)'."""
    partes = re.split(r"(\d+)", nome.lower())
    return [(int(p) if p.isdigit() else p) for p in partes]
