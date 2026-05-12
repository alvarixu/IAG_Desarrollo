# 📘 REFS — Referencias técnicas de RNA en Keras

> Documento de apoyo para profundizar en activaciones, optimizadores, métricas, funciones de pérdida, codificación de etiquetas y normalización.

---

## 📑 Índice

| Sección | Contenido |
|---|---|
| 1 | [Funciones de activación](#funciones-de-activacion) |
| 2 | [Optimizadores: Adam vs SGD](#optimizadores-adam-vs-sgd) |
| 3 | [Métricas y funciones de pérdida](#metricas-y-funciones-de-perdida) |
| 4 | [One-hot encoding y label encoding](#one-hot-encoding-y-label-encoding) |
| 5 | [Normalizers (normalización)](#normalizers-normalizacion) |

---

## Funciones de activación

Las activaciones introducen no linealidad en la red y permiten modelar relaciones complejas.

### ReLU

**Fórmula**  

$$
\text{ReLU}(x) = \max(0, x)
$$

**Definición**  
Devuelve 0 para valores negativos y deja pasar valores positivos.

**Para qué sirve**  
Es la activación más usada en capas ocultas por su simplicidad, eficiencia y buen comportamiento del gradiente en la práctica.

### Sigmoid

**Fórmula**  

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Definición**  
Comprime la salida al rango $(0,1)$.

**Para qué sirve**  
Muy útil en salida de clasificación binaria para interpretar la predicción como probabilidad.

### Softmax

**Fórmula**  

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

**Definición**  
Transforma un vector de logits en una distribución de probabilidades que suma 1.

**Para qué sirve**  
Salida estándar en clasificación multiclase de etiqueta única.

### Tanh

**Fórmula**  

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

**Definición**  
Mapea valores al rango $(-1,1)$, centrado en 0.

**Para qué sirve**  
Puede funcionar bien en algunos problemas, aunque hoy suele preferirse ReLU en capas ocultas.

---

## Optimizadores: Adam vs SGD

### SGD (Stochastic Gradient Descent)

**Actualización base**

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t)
$$

**Puntos clave**
- Simple y estable conceptualmente.
- Requiere más ajuste de `learning_rate`.
- Con momentum puede mejorar convergencia.

### Adam (Adaptive Moment Estimation)

**Idea de actualización (resumen)**  
Combina media móvil del gradiente (primer momento) y de su cuadrado (segundo momento), con corrección de sesgo.

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
$$

**Puntos clave**
- Suele converger rápido al inicio.
- Menos sensible que SGD al escalado de gradientes.
- Buen optimizador por defecto para comenzar.

### Diferencias prácticas (resumen)

| Aspecto | SGD | Adam |
|---|---|---|
| Velocidad inicial de convergencia | Media | Alta |
| Sensibilidad a `learning_rate` | Alta | Media |
| Facilidad para empezar | Media | Alta |
| Ajuste fino para generalización | Suele dar buen control | Muy cómodo para prototipos |

---

## Métricas y funciones de pérdida

> Incluye las más usadas en Keras para clasificación y regresión.

### Funciones de pérdida (loss)

#### MSE (Mean Squared Error)

$$
\text{MSE} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat y_i)^2
$$

**Uso:** regresión; penaliza mucho errores grandes.

#### MAE (Mean Absolute Error)

$$
\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat y_i|
$$

**Uso:** regresión; más robusta ante outliers que MSE.

#### Binary Crossentropy

$$
\text{BCE} = -\frac{1}{N}\sum_{i=1}^N \big(y_i\log(\hat y_i) + (1-y_i)\log(1-\hat y_i)\big)
$$

**Uso:** clasificación binaria con salida `sigmoid`.

#### Categorical Crossentropy

$$
\text{CCE} = -\frac{1}{N}\sum_{i=1}^N\sum_{c=1}^{K} y_{ic}\log(\hat y_{ic})
$$

**Uso:** multiclase con etiquetas one-hot y salida `softmax`.

#### Sparse Categorical Crossentropy

$$
\text{SCCE} = -\frac{1}{N}\sum_{i=1}^N \log(\hat y_{i, y_i})
$$

**Uso:** multiclase con etiquetas enteras (sin one-hot) y salida `softmax`.

### Métricas

#### Accuracy

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**Uso:** clasificación balanceada.

#### Precision

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**Uso:** cuando importa reducir falsos positivos.

#### Recall

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**Uso:** cuando importa capturar positivos reales.

#### F1-score

$$
F1 = 2\cdot \frac{\text{Precision}\cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Uso:** balance entre precisión y recall en clases desbalanceadas.

#### AUC-ROC

$$
\text{AUC} = \int_0^1 TPR(FPR)\,d(FPR)
$$

**Uso:** calidad del ranking probabilístico independiente de un umbral fijo.

#### RMSE

$$
\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat y_i)^2}
$$

**Uso:** regresión; error en mismas unidades que la variable objetivo.

---

## One-hot encoding y label encoding

### Label encoding

Convierte cada clase en un entero:
- rojo -> 0
- verde -> 1
- azul -> 2

**Uso práctico**
- Necesario para `sparse_categorical_crossentropy`.
- No introduce columnas extra.

### One-hot encoding

Cada clase se representa como un vector binario:
- rojo -> [1,0,0]
- verde -> [0,1,0]
- azul -> [0,0,1]

**Uso práctico**
- Se usa con `categorical_crossentropy`.
- Evita imponer orden artificial entre clases.

---

## Normalizers (normalización)

Normalizar ayuda a estabilizar y acelerar el entrenamiento.

### Estandarización (z-score)

$$
x' = \frac{x - \mu}{\sigma}
$$

**Uso:** opción por defecto frecuente en datos tabulares.

### Min-Max scaling

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

**Uso:** escala al rango [0,1], útil cuando se quiere rango acotado.

### Robust scaling

$$
x' = \frac{x - \text{mediana}}{IQR}
$$

**Uso:** más resistente a outliers.

### Normalización L2 por muestra

$$
x' = \frac{x}{\|x\|_2}
$$

**Uso:** frecuente cuando importa magnitud relativa del vector.

### Capa Normalization de Keras

Permite aprender estadísticas de normalización dentro del pipeline del modelo:

```python
from tensorflow.keras.layers import Normalization

norm = Normalization()
norm.adapt(X_train)  # aprende media y varianza en train
```

---

## Referencias

- Keras activations: https://keras.io/api/layers/activations/
- Keras optimizers: https://keras.io/api/optimizers/
- Keras losses: https://keras.io/api/losses/
- Keras metrics: https://keras.io/api/metrics/
- scikit-learn preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
