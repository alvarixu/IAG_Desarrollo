# =============================================================================
# PRÁCTICA: Predicción de Precios de Viviendas con Red Neuronal
# Caso de estudio completo - RNA con Keras
# =============================================================================

# %% [markdown]
# # 🏠 Predicción de Precios de Viviendas con Redes Neuronales
# 
# **Objetivo:** Desarrollar un modelo de redes neuronales para predecir precios
# de viviendas con precisión superior a la regresión lineal.
#
# **Métricas de éxito:**
# - RMSE inferior al 15% del precio promedio
# - R² mayor a 0.60

# %% Paso 0 - Instalación de dependencias
# !pip install tensorflow numpy pandas scikit-learn matplotlib seaborn kagglehub

# %% Paso 1 - Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
import warnings
warnings.filterwarnings('ignore')

# Semilla para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

print("✅ Librerías importadas correctamente")
print(f"TensorFlow versión: {tf.__version__}")

# %% Paso 2 - Cargar el dataset
# Opción A: Descargar desde kagglehub
try:
    import kagglehub
    path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
    import os
    csv_file = os.path.join(path, "Housing.csv")
    df = pd.read_csv(csv_file)
    print(f"✅ Dataset cargado desde kagglehub: {csv_file}")
except Exception as e:
    print(f"⚠️ kagglehub falló: {e}")
    print("Descarga manual desde: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset")
    print("Coloca Housing.csv en la misma carpeta que este notebook")
    df = pd.read_csv("Housing.csv")

print(f"\n📊 Dimensiones del dataset: {df.shape}")
print(f"   - {df.shape[0]} registros")
print(f"   - {df.shape[1]} columnas")

# %% Paso 3 - Exploración inicial de los datos
print("=" * 60)
print("EXPLORACIÓN INICIAL DE DATOS")
print("=" * 60)

# Primeras filas
print("\n📋 Primeras 5 filas:")
print(df.head())

# Información del dataset
print("\n📋 Información del dataset:")
print(df.info())

# Estadísticas descriptivas
print("\n📋 Estadísticas descriptivas:")
print(df.describe())

# Valores nulos
print("\n📋 Valores nulos por columna:")
print(df.isnull().sum())
print(f"\n   Total valores nulos: {df.isnull().sum().sum()}")

# %% Paso 4 - Visualización de distribuciones
# Histogramas de variables numéricas
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribución de Variables Numéricas', fontsize=16, fontweight='bold')

num_cols = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
for i, col in enumerate(num_cols):
    ax = axes[i // 3, i % 3]
    ax.hist(df[col], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_title(col, fontsize=12, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Frecuencia')
    # Línea de media
    ax.axvline(df[col].mean(), color='red', linestyle='--', label=f'Media: {df[col].mean():.0f}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('histogramas_variables.png', dpi=100, bbox_inches='tight')
plt.show()
print("✅ Histogramas guardados en 'histogramas_variables.png'")

# %% Paso 5 - Diagramas de caja (boxplots) para detectar outliers
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle('Boxplots - Detección de Outliers', fontsize=16, fontweight='bold')

for i, col in enumerate(['price', 'area', 'bedrooms', 'bathrooms']):
    sns.boxplot(data=df, y=col, ax=axes[i], color='lightcoral')
    axes[i].set_title(col, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('boxplots_outliers.png', dpi=100, bbox_inches='tight')
plt.show()
print("✅ Boxplots guardados")

# %% Paso 6 - Análisis de correlación
print("=" * 60)
print("ANÁLISIS DE CORRELACIÓN")
print("=" * 60)

# Codificamos temporalmente las variables categóricas para la correlación
df_encoded = df.copy()
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df_encoded[col] = df_encoded[col].map({'yes': 1, 'no': 0})

# furnishingstatus: label encoding
furnish_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
df_encoded['furnishingstatus'] = df_encoded['furnishingstatus'].map(furnish_map)

# Matriz de correlación
corr_matrix = df_encoded.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, vmin=-1, vmax=1)
plt.title('Matriz de Correlación', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('matriz_correlacion.png', dpi=100, bbox_inches='tight')
plt.show()

# Correlación con price
print("\n📊 Correlación de cada variable con 'price':")
corr_with_price = corr_matrix['price'].drop('price').sort_values(ascending=False)
print(corr_with_price)
print(f"\n🔍 Variable más correlacionada: {corr_with_price.index[0]} ({corr_with_price.iloc[0]:.3f})")

# %% Paso 7 - Preprocesamiento de datos
print("=" * 60)
print("PREPROCESAMIENTO DE DATOS")
print("=" * 60)

# 7.1 - Manejo de valores faltantes
print(f"\n📋 Valores faltantes: {df.isnull().sum().sum()}")
# Si hubiera nulos, imputamos con mediana (numéricos) o moda (categóricos)
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"   Imputado {col} con mediana: {df[col].median()}")

for col in df.select_dtypes(include=['object']).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"   Imputado {col} con moda: {df[col].mode()[0]}")

# 7.2 - Codificación de variables categóricas
print("\n📋 Codificación de variables categóricas:")

# Variables binarias (yes/no -> 1/0)
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})
    print(f"   ✅ {col}: yes->1, no->0")

# furnishingstatus: one-hot encoding (para evitar orden artificial)
print(f"   ✅ furnishingstatus: one-hot encoding")
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True, dtype=int)

print(f"\n📊 Columnas finales ({len(df.columns)}):")
print(df.columns.tolist())
print(f"\n{df.head()}")

# %% Paso 8 - Separar features y target
# Separamos X (features) e y (target = price)
X = df.drop('price', axis=1)
y = df['price']

print(f"📊 Features (X): {X.shape}")
print(f"📊 Target (y): {y.shape}")
print(f"📊 Precio medio: ${y.mean():,.0f}")
print(f"📊 Precio mediano: ${y.median():,.0f}")

# %% Paso 9 - División Train-Test (80/20)
print("=" * 60)
print("DIVISIÓN TRAIN-TEST (80/20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"📊 Train: {X_train.shape[0]} muestras ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"📊 Test:  {X_test.shape[0]} muestras ({X_test.shape[0]/len(X)*100:.0f}%)")

# %% Paso 10 - Normalización con StandardScaler
# Escalamos SOLO con datos de train para evitar data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform en train
X_test_scaled = scaler.transform(X_test)          # solo transform en test

print("✅ Normalización aplicada con StandardScaler")
print(f"   Media train (tras escalar): {X_train_scaled.mean(axis=0).round(2)}")
print(f"   Std train (tras escalar):   {X_train_scaled.std(axis=0).round(2)}")

# %% Paso 11 - Arquitectura de la Red Neuronal
print("=" * 60)
print("MODEL PLANNING - ARQUITECTURA DE LA RED")
print("=" * 60)

n_features = X_train_scaled.shape[1]
print(f"📊 Número de features de entrada: {n_features}")

# Construimos el modelo según las especificaciones de la práctica:
# - Capa entrada: n_features neuronas
# - 2 capas ocultas densas con ReLU
# - Capa salida: 1 neurona con activación lineal (regresión)
model = keras.Sequential([
    # Capa de entrada
    layers.Input(shape=(n_features,)),
    # Primera capa oculta: 64 neuronas, ReLU, regularización L2
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.2),  # Dropout para evitar sobreajuste
    # Segunda capa oculta: 32 neuronas, ReLU
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.2),
    # Capa de salida: 1 neurona, activación lineal (regresión)
    layers.Dense(1, activation='linear')
], name='modelo_precios_viviendas')

model.summary()

# %% Paso 12 - Compilar el modelo (Adam)
# Loss: MSE (error cuadrático medio) -> estándar para regresión
# Optimizador: Adam con learning_rate=0.001
# Métricas: MAE para seguimiento durante entrenamiento
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)
print("✅ Modelo compilado con Adam (lr=0.001), loss=MSE, metrics=[MAE]")

# %% Paso 13 - Callbacks (EarlyStopping)
# EarlyStopping: para el entrenamiento si val_loss no mejora en 10 épocas
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# ReduceLROnPlateau: reduce learning rate si val_loss se estanca
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

print("✅ Callbacks configurados: EarlyStopping + ReduceLROnPlateau")

# %% Paso 14 - Entrenamiento del modelo
print("=" * 60)
print("ENTRENAMIENTO DEL MODELO")
print("=" * 60)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,        # 20% de train como validación
    epochs=100,                   # Máximo 100 épocas
    batch_size=32,                # Tamaño de batch
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print(f"\n✅ Entrenamiento completado en {len(history.history['loss'])} épocas")

# %% Paso 15 - Curvas de entrenamiento (loss y MAE)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history.history['loss'], label='Train Loss', color='steelblue', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Val Loss', color='coral', linewidth=2)
axes[0].set_title('Evolución del Loss (MSE)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('MSE')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAE
axes[1].plot(history.history['mae'], label='Train MAE', color='steelblue', linewidth=2)
axes[1].plot(history.history['val_mae'], label='Val MAE', color='coral', linewidth=2)
axes[1].set_title('Evolución del MAE', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Curvas de Entrenamiento', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('curvas_entrenamiento.png', dpi=100, bbox_inches='tight')
plt.show()

# %% Paso 16 - Experimentación: Modelo con SGD
print("=" * 60)
print("EXPERIMENTACIÓN - MODELO CON SGD")
print("=" * 60)

model_sgd = keras.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.2),
    layers.Dense(1, activation='linear')
], name='modelo_sgd')

# SGD con momentum
model_sgd.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
    loss='mse',
    metrics=['mae']
)

history_sgd = model_sgd.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[
        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ],
    verbose=0
)
print(f"✅ SGD completado en {len(history_sgd.history['loss'])} épocas")

# %% Paso 17 - Evaluación en Test
print("=" * 60)
print("EVALUACIÓN EN CONJUNTO DE TEST")
print("=" * 60)

# Predicciones
y_pred_adam = model.predict(X_test_scaled, verbose=0).flatten()
y_pred_sgd = model_sgd.predict(X_test_scaled, verbose=0).flatten()

# Métricas para Adam
rmse_adam = np.sqrt(mean_squared_error(y_test, y_pred_adam))
mae_adam = mean_absolute_error(y_test, y_pred_adam)
r2_adam = r2_score(y_test, y_pred_adam)

# Métricas para SGD
rmse_sgd = np.sqrt(mean_squared_error(y_test, y_pred_sgd))
mae_sgd = mean_absolute_error(y_test, y_pred_sgd)
r2_sgd = r2_score(y_test, y_pred_sgd)

# Umbral de éxito: RMSE < 15% del precio promedio
precio_promedio = y_test.mean()
umbral_rmse = 0.15 * precio_promedio

print(f"\n{'Métrica':<20} {'Adam':>15} {'SGD':>15}")
print("-" * 50)
print(f"{'RMSE':<20} {rmse_adam:>15,.0f} {rmse_sgd:>15,.0f}")
print(f"{'MAE':<20} {mae_adam:>15,.0f} {mae_sgd:>15,.0f}")
print(f"{'R²':<20} {r2_adam:>15.4f} {r2_sgd:>15.4f}")
print("-" * 50)
print(f"{'Precio promedio':<20} {precio_promedio:>15,.0f}")
print(f"{'Umbral RMSE (15%)':<20} {umbral_rmse:>15,.0f}")
print(f"\n{'✅' if rmse_adam < umbral_rmse else '❌'} Adam RMSE {'<' if rmse_adam < umbral_rmse else '>'} umbral")
print(f"{'✅' if r2_adam > 0.60 else '❌'} Adam R² {'>' if r2_adam > 0.60 else '<'} 0.60")

# %% Paso 18 - Gráficos de evaluación
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Dispersión: Real vs Predicho
axes[0].scatter(y_test, y_pred_adam, alpha=0.5, color='steelblue', edgecolors='white', s=50)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Línea ideal')
axes[0].set_xlabel('Precio Real', fontsize=12)
axes[0].set_ylabel('Precio Predicho', fontsize=12)
axes[0].set_title(f'Real vs Predicho (R²={r2_adam:.3f})', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Distribución de errores
errores = y_test.values - y_pred_adam
axes[1].hist(errores, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Error (Real - Predicho)', fontsize=12)
axes[1].set_ylabel('Frecuencia', fontsize=12)
axes[1].set_title('Distribución de Errores', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Errores porcentuales
error_pct = np.abs(errores / y_test.values) * 100
axes[2].hist(error_pct, bins=30, color='coral', edgecolor='white', alpha=0.8)
axes[2].axvline(10, color='red', linestyle='--', linewidth=2, label='Umbral 10%')
axes[2].set_xlabel('Error Porcentual (%)', fontsize=12)
axes[2].set_ylabel('Frecuencia', fontsize=12)
axes[2].set_title('Distribución Error Porcentual', fontsize=14, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('Evaluación de Predicciones', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('evaluacion_predicciones.png', dpi=100, bbox_inches='tight')
plt.show()

# %% Paso 19 - Análisis de errores significativos (>10%)
print("=" * 60)
print("ANÁLISIS DE ERRORES SIGNIFICATIVOS (>10%)")
print("=" * 60)

error_pct_series = pd.Series(error_pct, index=y_test.index)
mask_high_error = error_pct_series > 10

print(f"\n📊 Predicciones con error > 10%: {mask_high_error.sum()} de {len(y_test)} ({mask_high_error.sum()/len(y_test)*100:.1f}%)")

if mask_high_error.sum() > 0:
    high_error_df = pd.DataFrame({
        'Precio_Real': y_test[mask_high_error],
        'Precio_Predicho': y_pred_adam[mask_high_error.values],
        'Error_%': error_pct_series[mask_high_error].round(1)
    }).sort_values('Error_%', ascending=False)
    print(f"\n📋 Top 10 predicciones con mayor error:")
    print(high_error_df.head(10))

# %% Paso 20 - Comparación con Baseline (Regresión Lineal)
print("=" * 60)
print("COMPARACIÓN CON BASELINE - REGRESIÓN LINEAL")
print("=" * 60)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"\n{'Métrica':<20} {'Red Neuronal':>15} {'Reg. Lineal':>15} {'Diferencia':>15}")
print("-" * 65)
print(f"{'RMSE':<20} {rmse_adam:>15,.0f} {rmse_lr:>15,.0f} {rmse_lr-rmse_adam:>+15,.0f}")
print(f"{'MAE':<20} {mae_adam:>15,.0f} {mae_lr:>15,.0f} {mae_lr-mae_adam:>+15,.0f}")
print(f"{'R²':<20} {r2_adam:>15.4f} {r2_lr:>15.4f} {r2_adam-r2_lr:>+15.4f}")

if r2_adam > r2_lr:
    print(f"\n✅ La Red Neuronal supera a la Regresión Lineal en R² por {r2_adam-r2_lr:.4f}")
else:
    print(f"\n⚠️ La Regresión Lineal obtiene mejor R². Considerar ajustar hiperparámetros.")

# %% Paso 21 - Validación Cruzada K-Fold (k=5)
print("=" * 60)
print("VALIDACIÓN CRUZADA K-FOLD (k=5)")
print("=" * 60)

X_all_scaled = scaler.fit_transform(X)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all_scaled)):
    X_fold_train, X_fold_val = X_all_scaled[train_idx], X_all_scaled[val_idx]
    y_fold_train, y_fold_val = y.values[train_idx], y.values[val_idx]
    
    fold_model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(1, activation='linear')
    ])
    fold_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    fold_model.fit(X_fold_train, y_fold_train, epochs=100, batch_size=32,
                   validation_data=(X_fold_val, y_fold_val),
                   callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                   verbose=0)
    
    y_fold_pred = fold_model.predict(X_fold_val, verbose=0).flatten()
    fold_r2 = r2_score(y_fold_val, y_fold_pred)
    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_fold_pred))
    fold_scores.append({'fold': fold+1, 'R2': fold_r2, 'RMSE': fold_rmse})
    print(f"   Fold {fold+1}: R²={fold_r2:.4f}, RMSE={fold_rmse:,.0f}")

scores_df = pd.DataFrame(fold_scores)
print(f"\n📊 Media K-Fold: R²={scores_df['R2'].mean():.4f} (±{scores_df['R2'].std():.4f})")
print(f"📊 Media K-Fold: RMSE={scores_df['RMSE'].mean():,.0f} (±{scores_df['RMSE'].std():,.0f})")

# %% Paso 22 - Guardar el modelo
print("=" * 60)
print("SERIALIZACIÓN DEL MODELO")
print("=" * 60)

# Guardar en formato .keras (recomendado en TF2+)
model.save('modelo_precios_viviendas.keras')
print("✅ Modelo guardado como 'modelo_precios_viviendas.keras'")

# Verificar carga
modelo_cargado = keras.models.load_model('modelo_precios_viviendas.keras')
y_pred_check = modelo_cargado.predict(X_test_scaled[:3], verbose=0).flatten()
print(f"✅ Verificación de carga: predicciones = {y_pred_check}")

# %% Paso 23 - Resumen final
print("\n" + "=" * 60)
print("📋 RESUMEN FINAL DEL MODELO")
print("=" * 60)
print(f"""
🏠 Modelo: Red Neuronal Densa (MLP)
📐 Arquitectura: {n_features} → 64 (ReLU+Dropout) → 32 (ReLU+Dropout) → 1 (Linear)
🎯 Optimizador: Adam (lr=0.001)
📉 Loss: MSE

📊 Resultados en Test:
   RMSE:  {rmse_adam:,.0f}  (umbral: {umbral_rmse:,.0f})
   MAE:   {mae_adam:,.0f}
   R²:    {r2_adam:.4f}  (objetivo: >0.60)

📊 Validación Cruzada (5-Fold):
   R² medio: {scores_df['R2'].mean():.4f} (±{scores_df['R2'].std():.4f})

📊 vs Baseline (Reg. Lineal):
   R² RNA: {r2_adam:.4f} vs R² LR: {r2_lr:.4f}

{'✅ MODELO CUMPLE OBJETIVOS' if r2_adam > 0.60 and rmse_adam < umbral_rmse else '⚠️ MODELO NECESITA AJUSTES'}
""")
