# 🧠 RNA Base con Keras — Guía conceptual y práctica


## 📑 Índice

| Sección | Contenido |
|---|---|
| 1️⃣ | [Qué es una RNA y por qué usarla](#1️⃣-qué-es-una-rna-y-por-qué-usarla) |
| 2️⃣ | [Anatomía de una red neuronal (entrada → salida) + ejemplo Keras](#2️⃣-anatomía-de-una-red-neuronal-entrada--salida) |
| 3️⃣ | [Activaciones y no linealidad + ejemplo Keras](#3️⃣-activaciones-y-no-linealidad) |
| 4️⃣ | [Salida y función de pérdida según tarea + ejemplo Keras](#4️⃣-salida-y-función-de-pérdida-según-tarea) |
| 5️⃣ | [Cómo aprende la red: backprop y optimización + ejemplo Keras](#5️⃣-cómo-aprende-la-red-backprop-y-optimización) |
| 6️⃣ | [Regularización y estabilidad del entrenamiento + ejemplo Keras](#6️⃣-regularización-y-estabilidad-del-entrenamiento) |
| 7️⃣ | [Métricas e interpretación de resultados + ejemplo Keras](#7️⃣-métricas-e-interpretación-de-resultados) |
| 8️⃣ | [Ejemplo completo y explicativo en Keras (pipeline end-to-end)](#8️⃣-ejemplo-completo-y-explicativo-en-keras-end-to-end) |
| 9️⃣ | [Plantillas rápidas de arquitectura en Keras](#9️⃣-plantillas-rápidas-de-arquitectura-en-keras) |
| 🔟 | [Checklist de depuración y buenas prácticas](#-checklist-de-depuración-y-buenas-prácticas) |

---

## 1️⃣ Qué es una RNA y por qué usarla

Una **red neuronal artificial (RNA)** es un modelo compuesto por capas de transformaciones matemáticas que aprende relaciones entre entradas y salidas a partir de datos.  

La idea central es esta:
- cada capa transforma la representación anterior,
- las capas intermedias aprenden patrones útiles,
- la salida produce una predicción (clase o valor numérico).

Una RNA destaca cuando la relación entre variables es **no lineal** o compleja, y cuando un modelo lineal simple se queda corto.

> 💡 **Idea clave:** una RNA no “memoriza reglas fijas”; aprende parámetros (pesos y sesgos) que minimizan una función de error.

---

## 2️⃣ Anatomía de una red neuronal (entrada → salida)

Una RNA densa (MLP) en Keras se compone de:
1. **Capa de entrada (`Input`)**: define forma de los datos.
2. **Capas ocultas (`Dense`)**: extraen representaciones.
3. **Activaciones**: introducen no linealidad.
4. **Capa de salida**: adapta el modelo al tipo de problema.

Cada neurona calcula:

$$
z = (w \cdot x) + b \rightarrow a = f(z)
$$

donde `w` son pesos, `b` sesgo y `f` la activación.

### Ejemplo Keras (estructura de la red)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

modelo = keras.Sequential([
    layers.Input(shape=(20,)),          # 20 variables de entrada
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # salida binaria
])

modelo.summary()
```

---

## 3️⃣ Activaciones y no linealidad

Sin activación no lineal, varias capas densas seguidas equivalen a una sola transformación lineal. Por eso la activación es crítica.

Activaciones más usadas ([más info](./REFS.md#funciones-de-activacion)):
- **ReLU**: estándar en capas ocultas (`max(0, x)`), rápida y estable en la práctica. [más info](./REFS.md#relu)
- **Sigmoid**: salida de probabilidad para clasificación binaria. [más info](./REFS.md#sigmoid)
- **Softmax**: distribución de probabilidad entre clases (multiclase). [más info](./REFS.md#softmax)
- **Tanh**: alternativa menos usada que ReLU en redes modernas densas. [más info](./REFS.md#tanh)

> ⚠️ **Error frecuente:** usar `softmax` para binaria (2 clases) cuando bastaba `sigmoid` + `binary_crossentropy`.

### Ejemplo Keras (mismo cuerpo, diferentes salidas)

```python
# Binaria
salida_binaria = layers.Dense(1, activation="sigmoid")

# Multiclase (5 clases)
salida_multiclase = layers.Dense(5, activation="softmax")
```

---

## 4️⃣ Salida y función de pérdida según tarea

La capa de salida y la `loss` deben ser coherentes con el objetivo ([más info](./REFS.md#metricas-y-funciones-de-perdida)):

| Tarea | Capa de salida | Loss recomendada |
|---|---|---|
| Regresión | `Dense(1, activation="linear")` | `mse` o `mae` |
| Clasificación binaria | `Dense(1, activation="sigmoid")` | `binary_crossentropy` |
| Clasificación multiclase (one-hot) | `Dense(n, activation="softmax")` | `categorical_crossentropy` |
| Clasificación multiclase (labels enteros) | `Dense(n, activation="softmax")` | `sparse_categorical_crossentropy` |

> 💡 **Codificación de etiquetas:** diferencias entre *one-hot* y *label encoding* [más info](./REFS.md#one-hot-encoding-y-label-encoding).

### Ejemplo Keras (compilación según tarea)

```python
# Binaria
modelo_bin = keras.Sequential([
    layers.Input(shape=(20,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
modelo_bin.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Regresión
modelo_reg = keras.Sequential([
    layers.Input(shape=(20,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="linear")
])
modelo_reg.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)
```

---

## 5️⃣ Cómo aprende la red: backprop y optimización

Proceso de aprendizaje (conceptual):
1. **Forward pass**: se calcula la predicción.
2. **Cálculo de pérdida**: se mide el error.
3. **Backpropagation**: se calcula cómo afecta cada peso al error.
4. **Actualización de pesos** con un optimizador (SGD, Adam...). [más info](./REFS.md#optimizadores-adam-vs-sgd)

Conceptos que debes dominar:
- **Learning rate (`lr`)**: tamaño del paso al actualizar pesos.
- **Batch size**: ejemplos por actualización.
- **Epoch**: una pasada completa por el conjunto de entrenamiento.

`Adam` suele ser un buen punto de partida para comenzar rápido. [más info](./REFS.md#optimizadores-adam-vs-sgd)

### Ejemplo Keras (optimizador y tasa de aprendizaje)

```python
opt = keras.optimizers.Adam(learning_rate=1e-3)

modelo.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

---

## 6️⃣ Regularización y estabilidad del entrenamiento

Una red puede sobreajustar: aprende muy bien entrenamiento pero generaliza mal en validación/test.

Técnicas básicas:
- **Dropout**: desactiva neuronas aleatoriamente durante entrenamiento.
- **L2 (weight decay)**: penaliza pesos grandes.
- **EarlyStopping**: para entrenamiento cuando validación deja de mejorar.
- **Normalización/escalado** de entrada: mejora estabilidad numérica. [más info](./REFS.md#normalizers-normalizacion)

### Ejemplo Keras (Dropout + L2 + EarlyStopping)

```python
from tensorflow.keras import regularizers

modelo_reg = keras.Sequential([
    layers.Input(shape=(20,)),
    layers.Dense(64, activation="relu",
                 kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

---

## 7️⃣ Métricas e interpretación de resultados

La métrica depende del problema ([más info](./REFS.md#metricas-y-funciones-de-perdida)):
- **Accuracy**: útil en clases balanceadas.
- **Precision/Recall/F1**: clave en desbalance y costes asimétricos.
- **AUC**: calidad de ranking probabilístico.
- **MAE/RMSE**: error en regresión.

Nunca tomes decisiones solo con una métrica si el problema es sensible al tipo de error.

### Ejemplo Keras (múltiples métricas)

```python
modelo.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc")
    ]
)
```

---

## 8️⃣ Ejemplo completo y explicativo en Keras (end-to-end)

Objetivo: clasificación binaria con pipeline completo.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1) Datos ficticios (reemplazar por tus datos reales)
rng = np.random.default_rng(42)
X = rng.normal(size=(2000, 20))
y = (X[:, 0] + 0.5 * X[:, 1] - 0.2 * X[:, 2] > 0).astype(int)

# 2) Split train/val/test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

# 3) Escalado (fit SOLO con train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 4) Modelo
model = keras.Sequential([
    layers.Input(shape=(20,)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# 5) Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

# 6) Entrenamiento
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# 7) Evaluación final
resultados = model.evaluate(X_test, y_test, verbose=0)
print(dict(zip(model.metrics_names, resultados)))

# 8) Predicciones
probas = model.predict(X_test[:5], verbose=0)
print("Probabilidades:", probas.ravel())
print("Clases:", (probas.ravel() >= 0.5).astype(int))
```

### Qué está pasando en cada fase

1. **Datos y split:** separas entrenamiento/validación/test para medir generalización real.  
2. **Escalado:** evita que magnitudes distintas dominen el aprendizaje. [más info](./REFS.md#normalizers-normalizacion)  
3. **Arquitectura:** bloques `Dense` aprenden representaciones; `Dropout` reduce overfitting.  
4. **Compile:** defines cómo aprende (`optimizer`), qué optimiza (`loss`) y cómo lo mides (`metrics`). [más info](./REFS.md#metricas-y-funciones-de-perdida)  
5. **Fit:** entrenamiento iterativo; `EarlyStopping` evita seguir cuando ya no mejora validación.  
6. **Evaluate/Predict:** obtienes rendimiento final y usas probabilidades para decisiones.

---

## 9️⃣ Plantillas rápidas de arquitectura en Keras

### A) Binaria

```python
modelo = keras.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```

### B) Multiclase (labels enteros)

```python
modelo = keras.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(n_classes, activation="softmax")
])
modelo.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

### C) Regresión

```python
modelo = keras.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="linear")
])
modelo.compile(optimizer="adam", loss="mse", metrics=["mae"])
```

---

## 🔟 Checklist de depuración y buenas prácticas

- [ ] ¿La **salida** y la **loss** son coherentes con la tarea?
- [ ] ¿Has hecho split antes de cualquier `fit` de preprocesado?
- [ ] ¿Tus datos de entrada están escalados/normalizados?
- [ ] ¿Monitoreas `train` y `val` para detectar overfitting/underfitting?
- [ ] ¿Usas `EarlyStopping` cuando procede?
- [ ] ¿Comparas contra un baseline simple?
- [ ] ¿Controlas semilla y configuración para reproducibilidad?
- [ ] ¿Eliges métricas alineadas con el coste real del error?

> ✅ **Regla de oro:** para mejorar un modelo, cambia **una sola cosa por experimento** (LR, batch size, capas, regularización) y registra resultado.

---

## 📚 Referencias

| Fuente | Enlace |
|---|---|
| Keras — Sequential model guide | https://keras.io/guides/sequential_model/ |
| Keras — Training & evaluation | https://keras.io/guides/training_with_built_in_methods/ |
| TensorFlow — Keras API | https://www.tensorflow.org/api_docs/python/tf/keras |
| Deep Learning (Goodfellow, Bengio, Courville) | https://www.deeplearningbook.org/ |

