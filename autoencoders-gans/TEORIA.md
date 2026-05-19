# 🧩 Autoencoders y GANs

> **📖 Resumen:** Autoencoders y GANs son dos familias clave de modelos generativos: los primeros aprenden representaciones comprimidas y reconstrucción, mientras que los segundos aprenden a generar datos realistas mediante un juego adversarial.

---

## 📑 Índice

| Sección | Contenido |
|---|---|
| 📌 | [Conceptos clave](#-conceptos-clave) |
| 🎯 | [Autoencoders: compresión y representación latente](#-autoencoders-compresión-y-representación-latente) |
| ⚔️ | [GANs: generación adversarial](#-gans-generación-adversarial) |
| 🔍 | [Comparación entre ambas arquitecturas](#-comparación-entre-ambas-arquitecturas) |
| 🔧 | [Ejemplos prácticos en Keras](#-ejemplos-prácticos-en-keras) |
| ✅ | [Mejores prácticas](#-mejores-prácticas) |
| 📚 | [Referencias](#-referencias) |

---

## 📌 Conceptos clave

> 💡 **Idea central:** ambas arquitecturas aprenden a capturar estructura en los datos, pero lo hacen con objetivos distintos: reconstruir o generar.

### 1️⃣ **Espacio latente**

**📝 Explicación:** es una representación comprimida de la información original. En autoencoders, el codificador transforma la entrada en un vector más pequeño que conserva lo esencial.

```markdown
💬 Ejemplo/Uso:
Una imagen 28x28 se resume en un vector de 16 dimensiones para retener solo los rasgos relevantes.
```

---

### 2️⃣ **Reconstrucción**

**📝 Explicación:** el modelo intenta reproducir su entrada lo mejor posible. Si reconstruye bien, significa que aprendió una representación útil.

```markdown
💬 Ejemplo/Uso:
Un autoencoder de ruido recibe una imagen corrupta y aprende a devolver la versión limpia.
```

---

### 3️⃣ **Aprendizaje adversarial**

**📝 Explicación:** en un GAN dos redes compiten: el generador crea muestras falsas y el discriminador intenta distinguirlas de las reales. Esa competencia mejora ambas.

```markdown
💬 Ejemplo/Uso:
Un generador produce rostros sintéticos mientras el discriminador aprende a detectar si son reales.
```

---

## 🎯 Autoencoders: compresión y representación latente

> 💡 **Contexto:** un autoencoder aprende una función de ida y vuelta: comprimir y reconstruir.

Un autoencoder suele tener tres partes:

1. **Encoder**: reduce la dimensión de la entrada.
2. **Bottleneck**: contiene la representación latente.
3. **Decoder**: reconstruye la entrada original.

### ¿Para qué sirven?

- **Compresión** de datos.
- **Denoising**: eliminar ruido de imágenes o señales.
- **Detección de anomalías**: si reconstruye mal una muestra, puede ser sospechosa.
- **Extracción de features**: aprender representaciones útiles sin etiquetas.

### Qué hay que vigilar

| Aspecto | Detalle |
|---|---|
| ✓ **Capacidad del modelo** | Si es demasiado grande, memoriza en vez de aprender una representación útil. |
| ✓ **Tamaño del bottleneck** | Si es demasiado pequeño, pierde información; si es demasiado grande, no comprime. |

### Variantes habituales

- **Autoencoder clásico**: reconstruye la entrada original.
- **Denoising autoencoder**: aprende a limpiar ruido.
- **Sparse autoencoder**: fuerza representaciones más compactas.
- **Variational autoencoder (VAE)**: introduce una formulación probabilística para generar muestras nuevas.

---

## ⚔️ GANs: generación adversarial

> 💡 **Contexto:** un GAN no intenta reconstruir la entrada; intenta aprender a generar datos nuevos indistinguibles de los reales.

### Componentes

- **Generador (G):** transforma ruido aleatorio en una muestra sintética.
- **Discriminador (D):** clasifica si una muestra parece real o falsa.

### Dinámica de entrenamiento

1. se muestrea ruido aleatorio;
2. el generador crea datos sintéticos;
3. el discriminador evalúa reales vs falsos;
4. ambos modelos se actualizan para mejorar;
5. el objetivo final es que el generador produzca muestras cada vez más creíbles.

### Problemas típicos

- **Inestabilidad de entrenamiento**.
- **Mode collapse**: el generador produce poca variedad.
- **Sensibilidad a hiperparámetros**.
- **Difícil evaluación**: “que parezca bueno” no siempre es fácil de medir.

### Casos de uso

- generación de imágenes sintéticas,
- super-resolución,
- restauración de imágenes,
- data augmentation sintética,
- transferencia de estilo y edición generativa.

---

## 🔍 Comparación entre ambas arquitecturas

| Criterio | Autoencoders | GANs |
|---|---|---|
| **Objetivo** | Reconstruir la entrada | Generar muestras nuevas y realistas |
| **Señal de entrenamiento** | Error de reconstrucción | Juego adversarial entre G y D |
| **Salida típica** | Datos parecidos a la entrada | Nuevos datos sintéticos |
| **Estabilidad** | Más estable | Más difícil de entrenar |
| **Uso principal** | Compresión, denoising, anomalías | Generación de contenido, realismo visual |
| **Evaluación** | Reconstrucción y error | Calidad, diversidad y realismo |

> ⚠️ **Regla práctica:** si quieres entender o comprimir datos, piensa en autoencoders; si quieres crear datos nuevos con apariencia realista, piensa en GANs.

---

## 🔧 Ejemplos prácticos en Keras

### Autoencoder simple para imágenes

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

input_img = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(input_img)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
encoded = layers.MaxPooling2D((2, 2), padding="same")(x)

x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
```

**💡 Explicación:**
- el encoder reduce la resolución;
- el bottleneck concentra la información;
- el decoder reconstruye la imagen original.

### GAN básica conceptual

```python
noise_dim = 100

generator = keras.Sequential([
    layers.Input(shape=(noise_dim,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(784, activation="sigmoid"),
    layers.Reshape((28, 28, 1))
])

discriminator = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
```

**💡 Explicación:**
- el generador parte de ruido;
- el discriminador aprende a validar si la muestra es real o falsa;
- en entrenamiento real, ambos se optimizan con pasos alternos.

---

## ✅ Mejores prácticas

| Práctica | Recomendación | 🎯 |
|---|---|---|
| **Elegir bien el objetivo** | Autoencoder para representación/reconstrucción; GAN para generación | ✓ |
| **Controlar la capacidad** | Evita modelos sobredimensionados que memoricen o colapsen | ✓ |
| **Monitorizar diversidad** | En GANs, no basta con calidad: también importa variedad | ✓ |
| **Usar métricas adecuadas** | Reconstruction loss, FID, evaluación visual o análisis de anomalías | ✓ |
| **Entrenar con estabilidad** | Ajusta learning rate, batch size y regularización con cuidado | ✓ |

> 💡 **Tip:** empieza con un autoencoder simple para entender la estructura latente y luego pasa a GANs si necesitas síntesis realista.

---

## 📚 Referencias

| Fuente | Enlace |
|---|---|
| TensorFlow — Tutorials de generación | https://www.tensorflow.org/tutorials/generative |
| TensorFlow — DCGAN tutorial | https://www.tensorflow.org/tutorials/generative/dcgan |
| Keras — API de capas | https://keras.io/api/layers/ |
| Goodfellow et al. — Generative Adversarial Nets | https://arxiv.org/abs/1406.2661 |
| Deep Learning Book | https://www.deeplearningbook.org/ |

---
