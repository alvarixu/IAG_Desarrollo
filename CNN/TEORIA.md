# 🎥 Redes Neuronales Convolucionales (CNN)

> **📖 Resumen:** Las CNN son redes diseñadas para trabajar con datos con estructura espacial, como imágenes. Aprenden primero bordes y texturas simples y después patrones más complejos, por eso son la arquitectura base en visión por computador.

---

## 📑 Índice

| Sección | Contenido |
|---|---|
| 📌 | [Conceptos clave](#-conceptos-clave) |
| 🧠 | [Qué es una CNN y cómo piensa](#-qué-es-una-cnn-y-cómo-piensa) |
| 🔬 | [Cómo funciona paso a paso](#-cómo-funciona-paso-a-paso) |
| ⚙️ | [Ejemplo de implementación en Keras](#-ejemplo-de-implementación-en-keras) |
| 🏭 | [Casos de uso reales](#-casos-de-uso-reales) |
| ✅ | [Mejores prácticas](#-mejores-prácticas) |
| 📚 | [Referencias](#-referencias) |

---

## 📌 Conceptos Clave

> 💡 **Idea principal:** una CNN no “lee” una imagen como una lista de números sueltos; la procesa conservando la relación entre píxeles vecinos.

### 1️⃣ **Convolución**

**📝 Explicación:** es una operación que desliza un pequeño filtro (kernel) sobre la imagen para detectar patrones locales. Cada filtro aprende a reconocer algo distinto: bordes, esquinas, texturas, formas, etc.

```markdown
💬 Ejemplo/Uso:
Un kernel 3x3 puede detectar un borde horizontal en una foto.
```

---

### 2️⃣ **Mapa de características (feature map)**

**📝 Explicación:** es el resultado de aplicar un filtro. Si el filtro “encuentra” el patrón que busca, el valor activado será alto.

```markdown
💬 Ejemplo/Uso:
Si un filtro busca bordes y la imagen contiene uno, el mapa de características resaltará esa zona.
```

---

### 3️⃣ **Pooling**

**📝 Explicación:** reduce el tamaño espacial de los mapas de características. Sirve para simplificar, acelerar y hacer el modelo más robusto a pequeños desplazamientos.

```markdown
💬 Ejemplo/Uso:
MaxPooling(2x2) conserva el valor más fuerte de cada bloque.
```

---

## 🧠 Qué es una CNN y cómo piensa

Una **CNN (Convolutional Neural Network)** es una red neuronal pensada para datos con estructura en rejilla: imágenes, vídeo, mapas de calor o incluso series temporales transformadas a formato bidimensional.

La gran diferencia frente a una red densa es esta:

- una red densa mezcla todos los píxeles desde el inicio;
- una CNN observa primero **pequeñas zonas locales**;
- después combina esas señales locales para construir ideas más complejas.

Eso la hace especialmente buena en visión, porque en una imagen importa mucho la posición relativa de los píxeles. Un borde, un ojo o una rueda no se interpretan por un píxel aislado, sino por el patrón que forman sus vecinos.

### La intuición más importante

Piensa en una CNN como una cadena de detectores:

1. las primeras capas detectan cosas simples, como líneas y bordes;
2. las capas intermedias combinan esas señales y detectan texturas, curvas o partes de objetos;
3. las capas profundas reconocen objetos más completos, como caras, coches o perros.

> 🧠 **Regla mental útil:** cuanto más profunda es la red, más abstractos suelen ser los patrones que aprende.

---

## 🔬 Cómo funciona paso a paso

### 1. Entrada

La imagen entra como un tensor. Por ejemplo:

- imagen en escala de grises: `alto x ancho x 1`
- imagen RGB: `alto x ancho x 3`

Por ejemplo, una imagen `28x28x1` de MNIST tiene 784 píxeles, pero la CNN no la aplana al principio: mantiene la forma 2D.

### 2. Convolución

Un kernel pequeño, como `3x3`, se desliza por la imagen. En cada posición calcula una combinación ponderada de los píxeles cercanos. Así obtiene un valor nuevo que resume “qué tan presente” está un patrón local.

### 3. Activación

Después de la convolución se aplica una función no lineal, normalmente **ReLU**. Esto permite que la red no se limite a combinaciones lineales.

### 4. Pooling

Se reduce la resolución espacial. Esto:

- disminuye el número de parámetros;
- acelera el entrenamiento;
- reduce sensibilidad a pequeños cambios de posición.

### 5. Repetición de bloques

Una CNN suele encadenar varios bloques:

- `Conv2D`
- `ReLU`
- `MaxPooling2D`

Cada bloque aprende representaciones más ricas.

### 6. Clasificación o predicción final

Al final, se usa:

- `Flatten` o `GlobalAveragePooling2D`;
- una o varias capas densas;
- una capa de salida con `sigmoid` o `softmax`.

> ⚠️ **Muy importante:** la CNN no termina “viendo imágenes”; termina produciendo una salida útil para la tarea: clase, probabilidad, máscara, coordenadas, etc.

---

## ⚙️ Ejemplo de implementación en Keras

Ejemplo clásico para clasificación de imágenes simples:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
```

### Qué está pasando aquí

- `Conv2D(32, (3,3))`: aprende 32 filtros distintos.
- `MaxPooling2D`: reduce tamaño y ruido.
- `Flatten`: convierte mapas 2D en vector.
- `Dense(10, softmax)`: devuelve probabilidades para 10 clases.

### Flujo típico de entrenamiento

1. normalizar imágenes a `[0, 1]`;
2. dividir en train/val/test;
3. entrenar con `fit`;
4. revisar validación para detectar overfitting;
5. ajustar arquitectura o añadir regularización si hace falta.

[Notebook de ejemplo](https://colab.research.google.com/drive/1oKydKFpR6IzEdORvRAKtATxuagNzOIMA?usp=sharing)

---

## 🏭 Casos de uso reales

Las CNN se usan cuando la estructura espacial importa mucho.

| Caso de uso | Qué hace la CNN | Ejemplo real |
|---|---|---|
| Clasificación de imágenes | Asigna una clase a cada imagen | Detectar si una radiografía muestra una patología |
| Detección de objetos | Localiza y clasifica objetos | Coches y peatones en conducción autónoma |
| Segmentación semántica | Clasifica cada píxel | Separar tumores en una imagen médica |
| OCR y lectura de documentos | Extrae rasgos visuales de texto | Leer matrículas o documentos escaneados |
| Inspección industrial | Detecta defectos visuales | Buscar grietas en piezas de fábrica |
| Seguridad y biometría | Reconoce rostros o patrones | Verificación facial en dispositivos |

### Ejemplos muy habituales

- **Medicina:** análisis de radiografías, TAC o resonancias.
- **Retail:** clasificación de productos, control de inventario visual.
- **Agricultura:** detección de plagas, enfermedades o madurez del cultivo.
- **Industria:** control de calidad por imagen.
- **Movilidad:** percepción visual en coches autónomos.

> 💡 Si el problema depende de “ver” patrones en una imagen, una CNN suele ser el primer candidato serio.

---

## ✅ Mejores Prácticas

| Práctica | Recomendación | 🎯 |
|---|---|---|
| **Entrada bien preparada** | Redimensiona, normaliza y asegura canales coherentes | ✓ |
| **Arquitectura progresiva** | Empieza con pocos bloques y aumenta solo si lo necesitas | ✓ |
| **Regularización** | Usa `Dropout`, data augmentation o weight decay si hay overfitting | ✓ |
| **Validación constante** | Supervisa train y val para no engañarte con accuracy alta | ✓ |
| **Elegir la loss correcta** | Binaria, multiclase o segmentación requieren pérdidas distintas | ✓ |

> 💡 **Tip:** en visión, mejorar datos y augmentations suele rendir más que hacer la red innecesariamente más grande.

---

## 📚 Referencias

| Fuente | Enlace |
|---|---|
| Keras — Conv2D | https://keras.io/api/layers/convolution_layers/convolution2d/ |
| Keras — MaxPooling2D | https://keras.io/api/layers/pooling_layers/max_pooling2d/ |
| TensorFlow — CNN tutoriales | https://www.tensorflow.org/tutorials/images |
| Deep Learning (Goodfellow, Bengio, Courville) | https://www.deeplearningbook.org/ |

---
