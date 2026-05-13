"""
App sencilla para probar la CNN de reconocimiento de dígitos.
Dibuja un número en el lienzo, pulsa "Predecir" y comprueba si la red acierta.

Requisitos: tensorflow, numpy, pillow  (todos ya instalados en el entorno).
Modelo: debe existir 'modelo_digitos_cnn_final.keras' en la misma carpeta.
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image, ImageDraw
import os, sys

# ── Cargar modelo ─────────────────────────────────────────────────────
MODELO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "modelo_digitos_cnn_final.keras")

try:
    import tensorflow as tf
    modelo = tf.keras.models.load_model(MODELO_PATH)
    print(f"✅ Modelo cargado correctamente desde: {MODELO_PATH}")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    sys.exit(1)

# ── Constantes de la interfaz ─────────────────────────────────────────
CANVAS_SIZE   = 280          # Tamaño del lienzo (px)
BRUSH_RADIUS  = 10           # Grosor del trazo
BG_COLOR      = "black"
BRUSH_COLOR   = "white"

# ── Clase principal ───────────────────────────────────────────────────
class AppDigito:
    def __init__(self, root):
        self.root = root
        self.root.title("🔢 Prueba de CNN - Reconocimiento de Dígitos")
        self.root.resizable(False, False)

        # --- Imagen interna para capturar el dibujo ---
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw  = ImageDraw.Draw(self.image)

        # --- Frame principal ---
        main = tk.Frame(root, bg="#1e1e1e")
        main.pack(padx=10, pady=10)

        # --- Título ---
        tk.Label(main, text="Dibuja un dígito (0-9)",
                 font=("Segoe UI", 16, "bold"), fg="white", bg="#1e1e1e"
                 ).pack(pady=(0, 8))

        # --- Lienzo ---
        self.canvas = tk.Canvas(main, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg=BG_COLOR, cursor="cross", highlightthickness=2,
                                highlightbackground="#555")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self._pintar)
        self.canvas.bind("<ButtonRelease-1>", self._soltar)

        # --- Botones ---
        btn_frame = tk.Frame(main, bg="#1e1e1e")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="🔍  Predecir", font=("Segoe UI", 12, "bold"),
                  bg="#4CAF50", fg="white", width=14,
                  command=self._predecir).pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame, text="🗑  Borrar", font=("Segoe UI", 12, "bold"),
                  bg="#f44336", fg="white", width=14,
                  command=self._borrar).pack(side=tk.LEFT, padx=5)

        # --- Resultado ---
        self.lbl_resultado = tk.Label(main, text="",
                                      font=("Segoe UI", 18, "bold"),
                                      fg="#00e676", bg="#1e1e1e")
        self.lbl_resultado.pack(pady=(5, 0))

        self.lbl_confianza = tk.Label(main, text="",
                                      font=("Segoe UI", 12),
                                      fg="#aaa", bg="#1e1e1e")
        self.lbl_confianza.pack()

        # --- Confirmación del usuario ---
        self.confirm_frame = tk.Frame(main, bg="#1e1e1e")
        # Se muestra solo después de predecir

    # ── Dibujar en el lienzo ──────────────────────────────────────────
    def _pintar(self, event):
        x, y = event.x, event.y
        r = BRUSH_RADIUS
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                fill=BRUSH_COLOR, outline=BRUSH_COLOR)
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def _soltar(self, _event):
        pass  # Se podría usar para segmentar trazos

    # ── Borrar lienzo ─────────────────────────────────────────────────
    def _borrar(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw  = ImageDraw.Draw(self.image)
        self.lbl_resultado.config(text="")
        self.lbl_confianza.config(text="")
        self.confirm_frame.pack_forget()

    # ── Predecir ──────────────────────────────────────────────────────
    def _predecir(self):
        # 1. Redimensionar a 28×28 y normalizar
        img = self.image.resize((28, 28), Image.LANCZOS)
        arr = np.array(img, dtype="float32") / 255.0
        arr = arr.reshape(1, 28, 28, 1)

        # 2. Comprobar que no esté vacío
        if arr.max() < 0.05:
            messagebox.showinfo("Aviso", "¡Dibuja algo primero!")
            return

        # Guardar la imagen preprocesada para posible reentrenamiento
        self._ultima_imagen = arr

        # 3. Predecir
        probs = modelo.predict(arr, verbose=0)[0]
        pred  = int(np.argmax(probs))
        conf  = probs[pred] * 100

        # 4. Mostrar resultado
        self.lbl_resultado.config(
            text=f"La red predice:  {pred}",
            fg="#00e676"
        )
        self.lbl_confianza.config(
            text=f"Confianza: {conf:.1f}%"
        )

        # 5. Pedir confirmación al usuario
        self._mostrar_confirmacion(pred)

    # ── Confirmación ──────────────────────────────────────────────────
    def _mostrar_confirmacion(self, prediccion):
        # Limpiar frame anterior
        self.confirm_frame.pack_forget()
        for w in self.confirm_frame.winfo_children():
            w.destroy()

        self.confirm_frame = tk.Frame(self.root.winfo_children()[0], bg="#1e1e1e")
        self.confirm_frame.pack(pady=5)

        tk.Label(self.confirm_frame,
                 text="¿Qué número dibujaste realmente?",
                 font=("Segoe UI", 11), fg="white", bg="#1e1e1e"
                 ).pack()

        btn_row = tk.Frame(self.confirm_frame, bg="#1e1e1e")
        btn_row.pack(pady=5)

        for d in range(10):
            color = "#4CAF50" if d == prediccion else "#333"
            tk.Button(btn_row, text=str(d), width=3,
                      font=("Segoe UI", 11, "bold"),
                      bg=color, fg="white",
                      command=lambda digit=d: self._confirmar(prediccion, digit)
                      ).pack(side=tk.LEFT, padx=2)

    def _confirmar(self, prediccion, real):
        if prediccion == real:
            self.lbl_resultado.config(
                text=f"✅  ¡Correcto!  Predicción: {prediccion}  |  Real: {real}",
                fg="#00e676"
            )
            self.lbl_confianza.config(text="Dibuja otro número y pulsa Predecir.")
        else:
            self.lbl_resultado.config(
                text=f"❌  Incorrecto.  Predicción: {prediccion}  |  Real: {real}",
                fg="#f44336"
            )
            # ── Reentrenar el modelo con la corrección ────────────────
            self.lbl_confianza.config(
                text="🧠 Aprendiendo de la corrección...",
                fg="#ffeb3b"
            )
            self.root.update()  # Actualizar la interfaz

            try:
                # Preparar la etiqueta correcta
                etiqueta = np.array([real])
                imagen   = self._ultima_imagen  # (1, 28, 28, 1)

                # Recompilar con learning rate bajo para fine-tuning
                modelo.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )

                # Entrenar unas pocas épocas con esta imagen
                modelo.fit(imagen, etiqueta, epochs=5, verbose=0)

                # Guardar el modelo actualizado
                modelo.save(MODELO_PATH)

                self.lbl_confianza.config(
                    text=f"🧠 ¡Modelo actualizado! Aprendió que era un {real}. Dibuja otro.",
                    fg="#00e676"
                )
                print(f"🧠 Modelo reentrenado: {prediccion} → {real} (modelo guardado)")

            except Exception as e:
                self.lbl_confianza.config(
                    text=f"⚠️ Error al reentrenar: {e}",
                    fg="#f44336"
                )
                print(f"⚠️ Error al reentrenar: {e}")

        self.confirm_frame.pack_forget()


# ── Arrancar la app ───────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = AppDigito(root)
    root.mainloop()
