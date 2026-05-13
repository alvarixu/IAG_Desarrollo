# Cómo funciona el aprendizaje cuando corriges al modelo

## El flujo completo

```
Dibujas un 3 → El modelo predice 8 → Tú pulsas "3" → El modelo aprende
```

---

## ¿Qué pasa internamente?

### Paso 1: Tu dibujo se convierte en datos

```
Lienzo 280×280 px → Redimensionar a 28×28 px → Normalizar (0-1) → Forma (1, 28, 28, 1)
```

El modelo solo entiende imágenes de **28×28 en escala de grises**, así que tu dibujo se transforma a ese formato. La normalización divide cada píxel entre 255 para que los valores queden entre 0.0 y 1.0, lo cual ayuda al modelo a converger más rápido.

---

### Paso 2: El modelo predice (Forward Pass)

La imagen pasa por las capas de la CNN en orden:

```
Imagen → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(64) → Flatten → Dense(64) → Dense(10)
```

La última capa (`Dense(10)` con activación **softmax**) produce **10 probabilidades**, una por cada dígito (0-9). Ejemplo:

```
Clase:        0     1     2     3     4     5     6     7     8     9
Probabilidad: 0.01  0.02  0.05  0.08  0.01  0.03  0.02  0.03  0.72  0.03
                                                              ↑
                                                    Dígito 8 = 72%
                                                    → Predicción: 8
```

El modelo elige el dígito con la **probabilidad más alta** como su predicción.

---

### Paso 3: Tú corriges → El modelo calcula su error

Cuando dices "era un 3", el modelo calcula la **función de pérdida** (*loss*) usando `sparse_categorical_crossentropy`:

```
Predicción del modelo: [0.01, 0.02, 0.05, 0.08, 0.01, 0.03, 0.02, 0.03, 0.72, 0.03]
Etiqueta correcta:     [0,    0,    0,    1,    0,    0,    0,    0,    0,    0   ]
                                          ↑
                                   Debería ser 3 = 100%
```

La función de pérdida mide **qué tan lejos está la predicción de la realidad**. Cuanto mayor es el loss, más se equivocó el modelo.

---

### Paso 4: Backpropagation (el aprendizaje real)

El error calculado se propaga **hacia atrás** por todas las capas de la red:

```
Dense(10) ← Dense(64) ← Flatten ← Conv2D(64) ← MaxPool ← Conv2D(32) ← Imagen
```

Para **cada peso** (cada conexión entre neuronas) de cada capa, se calcula:

1. **¿Cuánto contribuyó este peso al error?** → Se calcula el **gradiente** (derivada parcial del error respecto a ese peso).
2. **¿En qué dirección debo ajustarlo para reducir el error?** → El gradiente indica si el peso debe subir o bajar.

Este proceso se llama **backpropagation** (retropropagación) y es el corazón del aprendizaje en redes neuronales.

---

### Paso 5: Actualización de pesos (Gradiente Descendente)

Con los gradientes calculados, el optimizador **Adam** actualiza cada peso:

```
peso_nuevo = peso_viejo - learning_rate × gradiente
```

- **`learning_rate = 0.0001`** (10 veces más pequeño que el original de 0.001): ajusta los pesos **suavemente** para no destruir el conocimiento previo.
- Se repite **5 veces** (epochs) con la misma imagen para reforzar la corrección.

#### ¿Por qué un learning rate tan bajo?

Si fuera grande (ej. 0.01), el modelo cambiaría **drásticamente** sus pesos para acertar ese único ejemplo, pero **olvidaría** todo lo que aprendió durante el entrenamiento original con 60.000 imágenes. Con 0.0001, los ajustes son sutiles.

---

### Paso 6: Guardar el modelo

```python
modelo.save(MODELO_PATH)
```

Los pesos actualizados se guardan en `modelo_digitos_cnn_final.keras`. La próxima vez que abras la app, el modelo ya "recuerda" la corrección.

---

## Resumen visual del proceso

```
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│  Dibujas     │ → │  Modelo      │ → │  Predicción: 8   │
│  un 3        │    │  predice     │    │  (72% confianza) │
└─────────────┘    └──────────────┘    └──────────────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │  Tú dices: "Es   │
                                    │  un 3"           │
                                    └──────────────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │  Calcular error  │
                                    │  (Loss)          │
                                    └──────────────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │  Backpropagation │
                                    │  (propagar error │
                                    │  hacia atrás)    │
                                    └──────────────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │  Ajustar pesos   │
                                    │  (×5 épocas)     │
                                    └──────────────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │  Guardar modelo  │
                                    │  actualizado     │
                                    └──────────────────┘
```

---

## ⚠️ Limitación importante: Olvido Catastrófico

Este enfoque se llama **online learning** (aprendizaje en línea). Tiene un riesgo conocido: si corriges muchas veces con el mismo dígito, el modelo se **sesga** hacia esa clase.

Ejemplo de lo que puede pasar:

```
🧠 Modelo reentrenado: 2 → 8  (×3 veces)
🧠 Modelo reentrenado: 0 → 8  (×4 veces)
🧠 Modelo reentrenado: 3 → 8  (×2 veces)
```

Después de tantas correcciones hacia "8", los pesos del modelo se han ajustado demasiado para favorecer la clase 8. El modelo probablemente empezará a predecir "8" para casi todo.

### ¿Por qué ocurre?

El modelo fue entrenado originalmente con **60.000 imágenes** equilibradas. Cuando lo reentrenamos con **una sola imagen** muchas veces, esa imagen tiene un peso desproporcionado frente a todo el entrenamiento previo.

### ¿Cómo solucionarlo?

- **Solución rápida**: Vuelve a ejecutar las secciones 5 y 7 del notebook para regenerar el modelo original.
- **Solución avanzada** (en producción): Se usarían técnicas como:
  - **Experience Replay**: Guardar un buffer de ejemplos anteriores y reentrenar con ellos también.
  - **Regularización**: Limitar cuánto pueden cambiar los pesos en cada corrección.
  - **Entrenamiento por lotes**: Acumular varias correcciones y reentrenar en batch.

---

## Resumen en una frase

> El modelo calcula cuánto se equivocó, propaga ese error hacia atrás por todas sus capas (backpropagation), y ajusta ligeramente sus pesos para que la próxima vez que vea algo parecido, acierte.

---

## Explicación del código de la app (`app_dibujar_digito.py`)

A continuación se explica **cada parte** del código que realiza la predicción y el reentrenamiento.

### Función `_predecir()` — Preprocesar imagen y obtener predicción

```python
def _predecir(self):
```
Esta función se ejecuta cuando el usuario pulsa el botón **"Predecir"**.

---

```python
    img = self.image.resize((28, 28), Image.LANCZOS)
```
- `self.image` es la imagen PIL del lienzo (280×280 px, escala de grises).
- `.resize((28, 28))` la reduce a **28×28 px**, que es el tamaño que espera el modelo MNIST.
- `Image.LANCZOS` es un filtro de alta calidad para redimensionar (suaviza los píxeles al reducir).

---

```python
    arr = np.array(img, dtype="float32") / 255.0
```
- Convierte la imagen PIL en un **array NumPy** de números decimales.
- Divide entre 255 para **normalizar** los valores de `[0, 255]` a `[0.0, 1.0]`.
- El modelo fue entrenado con datos normalizados, así que es obligatorio hacerlo igual.

---

```python
    arr = arr.reshape(1, 28, 28, 1)
```
- Cambia la forma del array a **(1, 28, 28, 1)**:
  - `1` = un solo ejemplo (batch de tamaño 1)
  - `28, 28` = alto y ancho de la imagen
  - `1` = un solo canal de color (escala de grises)
- La CNN espera exactamente esta forma como entrada.

---

```python
    if arr.max() < 0.05:
        messagebox.showinfo("Aviso", "¡Dibuja algo primero!")
        return
```
- Si el valor máximo del array es menor que 0.05, significa que el lienzo está **prácticamente vacío** (todo negro).
- Muestra un aviso y no hace la predicción.

---

```python
    self._ultima_imagen = arr
```
- **Guarda** la imagen preprocesada en un atributo de la clase.
- Se usará más adelante si el usuario corrige la predicción y hay que reentrenar.

---

```python
    probs = modelo.predict(arr, verbose=0)[0]
```
- `modelo.predict(arr)` pasa la imagen por todas las capas de la CNN y devuelve las **10 probabilidades** (una por dígito).
- `verbose=0` silencia los mensajes de progreso de TensorFlow.
- `[0]` extrae el primer (y único) resultado del batch.
- `probs` queda como un array de 10 valores, ej: `[0.01, 0.02, 0.05, 0.08, 0.01, 0.03, 0.02, 0.03, 0.72, 0.03]`

---

```python
    pred = int(np.argmax(probs))
    conf = probs[pred] * 100
```
- `np.argmax(probs)` devuelve el **índice** del valor más alto del array (el dígito predicho).
- `probs[pred] * 100` convierte la probabilidad a **porcentaje** para mostrarla.

---

### Función `_mostrar_confirmacion()` — Botones 0-9 para confirmar

```python
def _mostrar_confirmacion(self, prediccion):
```
Crea un panel con **10 botones** (del 0 al 9) para que el usuario diga qué número dibujó realmente.

```python
    for d in range(10):
        color = "#4CAF50" if d == prediccion else "#333"
        tk.Button(btn_row, text=str(d), ...,
                  command=lambda digit=d: self._confirmar(prediccion, digit)
                  ).pack(side=tk.LEFT, padx=2)
```
- El botón correspondiente a la predicción del modelo se pinta en **verde** (`#4CAF50`).
- Los demás se pintan en **gris oscuro** (`#333`).
- Al pulsar cualquier botón, se llama a `_confirmar(prediccion, digit)` con la predicción del modelo y el dígito que eligió el usuario.

---

### Función `_confirmar()` — Reentrenamiento si hay error

```python
def _confirmar(self, prediccion, real):
```
Recibe dos parámetros:
- `prediccion`: lo que dijo el modelo (ej: 8)
- `real`: lo que dijo el usuario (ej: 3)

---

```python
    if prediccion == real:
        # ✅ Acierto: solo muestra mensaje de éxito
```
Si el modelo acertó, simplemente muestra un mensaje verde y **no reentrena** (no hay nada que corregir).

---

```python
    else:
        # ❌ Error: hay que reentrenar
```
Si el modelo se equivocó, comienza el proceso de reentrenamiento:

---

```python
        self.lbl_confianza.config(
            text="🧠 Aprendiendo de la corrección...",
            fg="#ffeb3b"
        )
        self.root.update()
```
- Muestra un mensaje **amarillo** avisando que el modelo está aprendiendo.
- `self.root.update()` fuerza a la interfaz a actualizarse **inmediatamente** (sin esto, el mensaje no se vería hasta que termine el entrenamiento).

---

```python
        etiqueta = np.array([real])
```
- Crea un array NumPy con la **etiqueta correcta** (el dígito que el usuario confirmó).
- Ejemplo: si el usuario dijo "3", `etiqueta = [3]`.
- Es un array porque `model.fit()` espera un lote de etiquetas, aunque sea de tamaño 1.

---

```python
        imagen = self._ultima_imagen  # (1, 28, 28, 1)
```
- Recupera la imagen preprocesada que se guardó en `_predecir()`.
- Ya está en el formato correcto: normalizada, 28×28, con dimensión de canal.

---

```python
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
```
- **Recompila** el modelo con un nuevo optimizador.
- `Adam(learning_rate=0.0001)`: learning rate **10 veces menor** que el original (0.001). Esto es crucial para que los ajustes sean suaves y el modelo no "olvide" todo lo que aprendió antes.
- `sparse_categorical_crossentropy`: la misma función de pérdida usada en el entrenamiento original. Mide cuánto se equivoca el modelo comparando su predicción con la etiqueta correcta.
- `metrics=['accuracy']`: para que internamente también calcule el porcentaje de acierto.

---

```python
        modelo.fit(imagen, etiqueta, epochs=5, verbose=0)
```
Este es el **corazón del reentrenamiento**:
- `imagen`: la imagen del dígito dibujado, ya preprocesada `(1, 28, 28, 1)`.
- `etiqueta`: la clase correcta `[3]`.
- `epochs=5`: repite el proceso de forward pass + backpropagation + actualización de pesos **5 veces** con esa misma imagen.
- `verbose=0`: silencia la salida de TensorFlow.

En cada época:
1. **Forward pass**: la imagen pasa por toda la red y produce una predicción.
2. **Cálculo del loss**: se mide cuánto se equivocó respecto a la etiqueta correcta.
3. **Backpropagation**: se calculan los gradientes (cuánto contribuyó cada peso al error).
4. **Actualización de pesos**: cada peso se ajusta ligeramente en la dirección que reduce el error.

---

```python
        modelo.save(MODELO_PATH)
```
- **Guarda** el modelo con los pesos actualizados en el archivo `.keras`.
- La próxima vez que se abra la app, el modelo cargará con los nuevos pesos, es decir, "recordará" la corrección.

---

```python
            except Exception as e:
                self.lbl_confianza.config(
                    text=f"⚠️ Error al reentrenar: {e}",
                    fg="#f44336"
                )
```
- Si ocurre cualquier error durante el reentrenamiento (falta de memoria, archivo corrupto, etc.), se captura la excepción y se muestra un mensaje en **rojo** sin que la app se cierre.
