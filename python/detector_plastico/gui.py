"""Interface gráfica (Tkinter) para anotar imagens, treinar e detectar.

Ciclo de uso:
  1. selecione uma imagem na lista à esquerda;
  2. pinte de verde o que é plástico (pincel), use a varinha, o retângulo ou o
     "ignorar"; apague com o botão direito;
  3. salve (Ctrl+S) — salvar sem pintar marca a imagem como "sem plástico";
  4. treine a IA e veja a eficiência;
  5. detecte em lote ou gere os filtros passo a passo.

Tkinter faz parte da biblioteca padrão do Python. A GUI usa os mesmos módulos de
PDI/VC do restante do projeto.
"""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as e:  # noqa: BLE001
    raise RuntimeError("Tkinter não está disponível. No Linux Mint/Ubuntu instale: "
                       "sudo apt install python3-tk") from e

from PIL import Image, ImageTk

from . import detector, etapas as etapas_mod, filtros, lote, treino
from .modelo import Modelo
from .projeto import FUNDO, IGNORAR, PLASTICO, Projeto

LADO_VISUAL = 900  # maior lado da imagem exibida no canvas


class App:
    def __init__(self, root, projeto: Projeto):
        self.root = root
        self.projeto = projeto
        self.modelo = None
        self.imagens = []
        self.indice = -1

        self.img_rgb = None          # np.ndarray (H,W,3) no tamanho de trabalho
        self.mascara = None          # np.ndarray (H,W) uint8
        self.escala_vis = 1.0        # trabalho -> visual
        self._tk_img = None
        self._bordas_cache = None

        self.ferramenta = tk.StringVar(value="plastico")
        self.tam_pincel = tk.IntVar(value=14)
        self.limiar = tk.DoubleVar(value=0.5)
        self.mostrar_ia = tk.BooleanVar(value=False)
        self._ia_overlay = None
        self._pintando = False
        self._retang_ini = None

        self._montar()
        self._carregar_modelo()
        self._recarregar_lista()
        root.bind("<Control-s>", lambda e: self.salvar())
        root.bind("b", lambda e: self.ferramenta.set("plastico"))
        root.bind("w", lambda e: self.ferramenta.set("varinha"))
        root.bind("r", lambda e: self.ferramenta.set("retangulo"))
        root.bind("i", lambda e: self.ferramenta.set("ignorar"))

    # ---------------------------------------------------------------- layout

    def _montar(self):
        self.root.title("Detector de Plástico em Rios — PDI + Visão Computacional (Python)")
        self.root.geometry("1200x760")

        barra = ttk.Frame(self.root, padding=4)
        barra.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(barra, text="Importar imagens…", command=self.importar).pack(side=tk.LEFT, padx=2)
        ttk.Button(barra, text="Treinar IA", command=self.treinar).pack(side=tk.LEFT, padx=2)
        ttk.Button(barra, text="Pré-anotar com a IA", command=self.pre_anotar).pack(side=tk.LEFT, padx=2)
        ttk.Button(barra, text="Detectar em lote…", command=self.detectar_lote).pack(side=tk.LEFT, padx=2)
        ttk.Button(barra, text="Filtros passo a passo", command=self.filtros_passo).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(barra, text="Sobrepor detecção da IA", variable=self.mostrar_ia,
                        command=self._redesenhar).pack(side=tk.LEFT, padx=10)

        corpo = ttk.Frame(self.root)
        corpo.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # lista de imagens
        esq = ttk.Frame(corpo, padding=4)
        esq.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(esq, text="Imagens").pack(anchor=tk.W)
        self.lista = tk.Listbox(esq, width=34, activestyle="dotbox")
        self.lista.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        sb = ttk.Scrollbar(esq, orient=tk.VERTICAL, command=self.lista.yview)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        self.lista.config(yscrollcommand=sb.set)
        self.lista.bind("<<ListboxSelect>>", self._selecionar)

        # canvas central
        centro = ttk.Frame(corpo)
        centro.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(centro, bg="#202020", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._mouse_baixo)
        self.canvas.bind("<B1-Motion>", self._mouse_arrasta)
        self.canvas.bind("<ButtonRelease-1>", self._mouse_cima)
        self.canvas.bind("<ButtonPress-3>", self._mouse_baixo_dir)
        self.canvas.bind("<B3-Motion>", self._mouse_arrasta_dir)

        # painel direito de ferramentas
        dir_ = ttk.Frame(corpo, padding=6)
        dir_.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(dir_, text="Ferramenta", font=("", 10, "bold")).pack(anchor=tk.W)
        for valor, rotulo in [("plastico", "Plástico (B) — pincel"),
                              ("ignorar", "Ignorar (I) — amarelo"),
                              ("varinha", "Varinha (W) — por contorno"),
                              ("retangulo", "Retângulo (R)")]:
            ttk.Radiobutton(dir_, text=rotulo, value=valor, variable=self.ferramenta).pack(anchor=tk.W)
        ttk.Label(dir_, text="(botão direito apaga)").pack(anchor=tk.W, pady=(0, 6))

        ttk.Label(dir_, text="Tamanho do pincel").pack(anchor=tk.W)
        ttk.Scale(dir_, from_=3, to=60, variable=self.tam_pincel, orient=tk.HORIZONTAL).pack(fill=tk.X)

        ttk.Separator(dir_).pack(fill=tk.X, pady=8)
        ttk.Button(dir_, text="Salvar anotação (Ctrl+S)", command=self.salvar).pack(fill=tk.X, pady=2)
        ttk.Button(dir_, text="Limpar anotação", command=self.limpar_anotacao).pack(fill=tk.X, pady=2)
        ttk.Button(dir_, text="Remover anotação", command=self.remover_anotacao).pack(fill=tk.X, pady=2)

        ttk.Separator(dir_).pack(fill=tk.X, pady=8)
        ttk.Label(dir_, text="Limiar da IA").pack(anchor=tk.W)
        self.lbl_limiar = ttk.Label(dir_, text="0.50")
        self.lbl_limiar.pack(anchor=tk.W)
        ttk.Scale(dir_, from_=0.1, to=0.9, variable=self.limiar, orient=tk.HORIZONTAL,
                  command=self._mudou_limiar).pack(fill=tk.X)

        ttk.Separator(dir_).pack(fill=tk.X, pady=8)
        self.lbl_modelo = ttk.Label(dir_, text="", justify=tk.LEFT, wraplength=220)
        self.lbl_modelo.pack(anchor=tk.W)

        # log inferior
        inf = ttk.Frame(self.root, padding=4)
        inf.pack(side=tk.BOTTOM, fill=tk.X)
        self.log_txt = tk.Text(inf, height=6, wrap=tk.WORD, bg="#111", fg="#ddd")
        self.log_txt.pack(fill=tk.X)
        self.log("Pronto. Selecione uma imagem e comece a anotar. Pressione B/W/R/I para trocar a ferramenta.")

    # ------------------------------------------------------------------ util

    def log(self, msg):
        self.log_txt.insert(tk.END, str(msg) + "\n")
        self.log_txt.see(tk.END)
        self.root.update_idletasks()

    def _carregar_modelo(self):
        self.modelo = None
        arq = self.projeto.arquivo_modelo()
        if arq.is_file():
            try:
                self.modelo = Modelo.carregar(arq)
                self.limiar.set(self.modelo.limiar)
                self._mudou_limiar(None)
            except Exception as e:  # noqa: BLE001
                self.log(f"Não foi possível carregar o modelo: {e}")
        self._atualizar_label_modelo()

    def _atualizar_label_modelo(self):
        if self.modelo is None:
            self.lbl_modelo.config(text="Nenhum modelo treinado.\nAnote imagens e clique em 'Treinar IA'.")
            return
        m = self.modelo
        txt = f"Modelo: {m.n_imagens} imagens anotadas"
        if m.f1 >= 0:
            txt += (f"\nEficiência (F1): {m.f1 * 100:.1f}%"
                    f"\n  precisão {m.precisao * 100:.1f}% · revocação {m.revocacao * 100:.1f}%"
                    f"\n  filtro HSV: {m.f1_classico * 100:.1f}%")
        self.lbl_modelo.config(text=txt)

    def _recarregar_lista(self):
        self.imagens = self.projeto.listar_imagens()
        self.lista.delete(0, tk.END)
        for f in self.imagens:
            marca = "● " if self.projeto.anotada(f) else "   "
            self.lista.insert(tk.END, marca + f.name)
        self.log(f"{len(self.imagens)} imagem(ns); "
                 f"{len(self.projeto.imagens_anotadas())} anotada(s).")

    # ------------------------------------------------------------- seleção

    def _selecionar(self, _evt=None):
        sel = self.lista.curselection()
        if not sel:
            return
        self._abrir(sel[0])

    def _abrir(self, indice):
        self.indice = indice
        f = self.imagens[indice]
        original = Projeto.ler_imagem(f)
        h0, w0 = original.shape[:2]
        esc = min(1.0, LADO_VISUAL / max(w0, h0))
        w, h = max(1, round(w0 * esc)), max(1, round(h0 * esc))
        self.img_rgb = filtros.redimensionar(original, w, h) if esc < 1 else original
        self.escala_vis = 1.0
        self._bordas_cache = None
        self._ia_overlay = None

        m = self.projeto.carregar_mascara(f, w, h)
        self.mascara = m if m is not None else np.zeros((h, w), dtype=np.uint8)
        self._redesenhar()

    # ----------------------------------------------------------- desenho

    def _redesenhar(self):
        if self.img_rgb is None:
            return
        h, w = self.img_rgb.shape[:2]
        base = (self.img_rgb * 255).astype(np.uint8).copy()

        # sobreposição das anotações
        pl = self.mascara == PLASTICO
        ig = self.mascara == IGNORAR
        base[pl] = (0.4 * base[pl] + np.array([0, 160, 0])).clip(0, 255).astype(np.uint8)
        base[ig] = (0.5 * base[ig] + np.array([150, 150, 0])).clip(0, 255).astype(np.uint8)

        img = Image.fromarray(base, "RGB")

        if self.mostrar_ia.get() and self.modelo is not None:
            if self._ia_overlay is None:
                r = detector.detectar(self.modelo, self.img_rgb)
                self._ia_overlay = lote.desenhar(self.img_rgb, r, float(self.limiar.get()))
            img = self._ia_overlay

        self._tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, w, h))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)

    # ------------------------------------------------------------- mouse

    def _coord(self, evt):
        x = int(self.canvas.canvasx(evt.x))
        y = int(self.canvas.canvasy(evt.y))
        return x, y

    def _mouse_baixo(self, evt):
        if self.img_rgb is None:
            return
        ferr = self.ferramenta.get()
        x, y = self._coord(evt)
        if ferr == "varinha":
            self._aplicar_varinha(x, y)
        elif ferr == "retangulo":
            self._retang_ini = (x, y)
        else:
            self._pintando = True
            self._pintar(x, y, apagar=False)

    def _mouse_arrasta(self, evt):
        if self._pintando:
            x, y = self._coord(evt)
            self._pintar(x, y, apagar=False)

    def _mouse_cima(self, evt):
        if self.ferramenta.get() == "retangulo" and self._retang_ini:
            x0, y0 = self._retang_ini
            x1, y1 = self._coord(evt)
            self._retangulo(x0, y0, x1, y1, apagar=False)
            self._retang_ini = None
        self._pintando = False

    def _mouse_baixo_dir(self, evt):
        if self.img_rgb is None:
            return
        self._pintando = True
        x, y = self._coord(evt)
        self._pintar(x, y, apagar=True)

    def _mouse_arrasta_dir(self, evt):
        x, y = self._coord(evt)
        self._pintar(x, y, apagar=True)

    def _valor_ferramenta(self):
        return IGNORAR if self.ferramenta.get() == "ignorar" else PLASTICO

    def _pintar(self, x, y, apagar):
        h, w = self.mascara.shape
        r = max(1, self.tam_pincel.get())
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        if y0 >= y1 or x0 >= x1:
            return
        ys, xs = np.ogrid[y0:y1, x0:x1]
        disco = (ys - y) ** 2 + (xs - x) ** 2 <= r * r
        valor = FUNDO if apagar else self._valor_ferramenta()
        reg = self.mascara[y0:y1, x0:x1]
        reg[disco] = valor
        self.mascara[y0:y1, x0:x1] = reg
        self._redesenhar()

    def _retangulo(self, x0, y0, x1, y1, apagar):
        h, w = self.mascara.shape
        xa, xb = sorted((max(0, min(w, x0)), max(0, min(w, x1))))
        ya, yb = sorted((max(0, min(h, y0)), max(0, min(h, y1))))
        self.mascara[ya:yb, xa:xb] = FUNDO if apagar else self._valor_ferramenta()
        self._redesenhar()

    def _aplicar_varinha(self, x, y):
        if self._bordas_cache is None:
            self._bordas_cache = filtros.bordas(self.img_rgb)
        h, w = self.mascara.shape
        reg = filtros.varinha(self.img_rgb, self._bordas_cache, x, y,
                              tolerancia=0.18, max_pixels=w * h // 3)
        if reg is None:
            self.log("Varinha: a região vazou (contorno aberto). Tente outro ponto ou o pincel.")
            return
        self.mascara[reg] = self._valor_ferramenta()
        self._redesenhar()

    # ----------------------------------------------------------- ações

    def salvar(self):
        if self.indice < 0:
            return
        f = self.imagens[self.indice]
        self.projeto.salvar_mascara(f, self.mascara)
        plast = int((self.mascara == PLASTICO).sum())
        self.log(f"Anotação salva: {f.name}"
                 + (f" ({plast} px de plástico)." if plast else " (sem plástico)."))
        # atualiza a marca ● na lista
        self.lista.delete(self.indice)
        self.lista.insert(self.indice, "● " + f.name)
        self.lista.selection_set(self.indice)

    def limpar_anotacao(self):
        if self.mascara is not None:
            self.mascara[:] = FUNDO
            self._redesenhar()

    def remover_anotacao(self):
        if self.indice < 0:
            return
        f = self.imagens[self.indice]
        self.projeto.remover_mascara(f)
        self.limpar_anotacao()
        self.lista.delete(self.indice)
        self.lista.insert(self.indice, "   " + f.name)
        self.lista.selection_set(self.indice)
        self.log(f"Anotação removida: {f.name}")

    def _mudou_limiar(self, _):
        self.lbl_limiar.config(text=f"{float(self.limiar.get()):.2f}")
        self._ia_overlay = None
        if self.mostrar_ia.get():
            self._redesenhar()

    def importar(self):
        arqs = filedialog.askopenfilenames(
            title="Selecionar imagens ou .zip",
            filetypes=[("Imagens e zip", "*.jpg *.jpeg *.png *.bmp *.gif *.zip"), ("Todos", "*.*")])
        if not arqs:
            return
        self._em_thread(lambda: self._importar_fim(
            self.projeto.importar_imagens([Path(a) for a in arqs], self.log)))

    def _importar_fim(self, _n):
        self.root.after(0, self._recarregar_lista)

    def pre_anotar(self):
        if self.modelo is None:
            messagebox.showinfo("Pré-anotar", "Treine a IA primeiro.")
            return
        if self.img_rgb is None:
            return
        r = detector.detectar(self.modelo, self.img_rgb)
        limiar = float(self.limiar.get())
        # pinta como plástico onde a probabilidade interpolada passa do limiar
        h, w = self.mascara.shape
        mapa = np.asarray(Image.fromarray((r.prob * 255).astype(np.uint8))
                          .resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
        self.mascara[mapa >= limiar] = PLASTICO
        self._redesenhar()
        self.log("Pré-anotação aplicada. Revise e corrija antes de salvar (Ctrl+S).")

    def treinar(self):
        if not self.projeto.imagens_anotadas():
            messagebox.showinfo("Treinar", "Anote pelo menos uma imagem e salve antes de treinar.")
            return
        self.log("Iniciando treino… (pode levar de segundos a minutos)")
        self._em_thread(self._treinar_bg)

    def _treinar_bg(self):
        try:
            treino.treinar(self.projeto, self.log)
            self.root.after(0, self._carregar_modelo)
        except Exception as e:  # noqa: BLE001
            self.log("ERRO no treino: " + str(e))
            self.log(traceback.format_exc())

    def detectar_lote(self):
        if self.modelo is None:
            messagebox.showinfo("Detectar", "Treine a IA primeiro.")
            return
        pasta = filedialog.askdirectory(title="Pasta com imagens para detectar")
        if not pasta:
            return
        imgs = Projeto.listar_imagens_em(Path(pasta), recursivo=False)
        if not imgs:
            messagebox.showinfo("Detectar", "Nenhuma imagem encontrada na pasta.")
            return
        from datetime import datetime
        saida = self.projeto.pasta_resultados / ("lote_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.log(f"Detectando {len(imgs)} imagens…")
        self._em_thread(lambda: self._detectar_fim(
            lote.detectar_pasta(self.modelo, imgs, saida, float(self.limiar.get()), self.log), saida))

    def _detectar_fim(self, n, saida):
        self.log(f"{n} imagens analisadas. Veja {saida}")

    def filtros_passo(self):
        if self.img_rgb is None:
            messagebox.showinfo("Filtros", "Selecione uma imagem primeiro.")
            return
        et = etapas_mod.calcular(self.img_rgb)
        if self.modelo is not None:
            r = detector.detectar(self.modelo, self.img_rgb)
            et.append(lote.etapa_ia(len(et) + 1, self.img_rgb, r, float(self.limiar.get())))
        self._janela_etapas(et)

    def _janela_etapas(self, etapas):
        win = tk.Toplevel(self.root)
        win.title("Filtros passo a passo (PDI)")
        win.geometry("980x720")
        nb = ttk.Notebook(win)
        nb.pack(fill=tk.BOTH, expand=True)
        self._refs_img = []
        for e in etapas:
            aba = ttk.Frame(nb)
            nb.add(aba, text=e.titulo.split(".")[0])
            img = e.imagem
            esc = min(1.0, 900 / max(img.size))
            if esc < 1:
                img = img.resize((int(img.size[0] * esc), int(img.size[1] * esc)))
            tkimg = ImageTk.PhotoImage(img)
            self._refs_img.append(tkimg)
            ttk.Label(aba, text=e.titulo, font=("", 11, "bold")).pack(anchor=tk.W, padx=6, pady=4)
            ttk.Label(aba, text=e.explicacao, wraplength=900, justify=tk.LEFT).pack(anchor=tk.W, padx=6)
            ttk.Label(aba, text="Resultado: " + e.info, foreground="#06c").pack(anchor=tk.W, padx=6, pady=2)
            tk.Label(aba, image=tkimg, bg="#202020").pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        def salvar_tudo():
            if self.indice < 0:
                return
            import re
            base = re.sub(r"[^A-Za-z0-9_-]+", "_", self.imagens[self.indice].stem).strip("_")
            pasta = self.projeto.pasta_resultados / f"filtros_{base}"
            etapas_mod.salvar(etapas, self.imagens[self.indice].name, pasta)
            self.log(f"Etapas salvas em PNG: {pasta}")
        ttk.Button(win, text="Salvar as etapas em PNG", command=salvar_tudo).pack(pady=4)

    # --------------------------------------------------------------- threads

    def _em_thread(self, fn):
        def alvo():
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                self.log("ERRO: " + str(e))
                self.log(traceback.format_exc())
        threading.Thread(target=alvo, daemon=True).start()


def abrir(projeto: Projeto):
    root = tk.Tk()
    App(root, projeto)
    root.mainloop()
