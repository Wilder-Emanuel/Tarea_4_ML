import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# 1. Carga de datos desde tu CSV
df = pd.read_csv('caso_CPU.csv')
usuarios = df['Usuarios'].values
cpu = df['CPU'].values

# 2. Gráfica comparativa
plt.figure(figsize=(12, 5))

# Subplot 1: Datos Crudos (Se debe ver una curva que se achata)
plt.subplot(1, 2, 1)
plt.scatter(usuarios, cpu, color='blue')
plt.title('Usuarios vs CPU (Original)')
plt.xlabel('Usuarios')
plt.ylabel('CPU (%)')

# Subplot 2: Datos con ln(u) (Se debe ver una tendencia lineal clara)
plt.subplot(1, 2, 2)
plt.scatter(np.log(usuarios), cpu, color='green')
plt.title('ln(Usuarios) vs CPU (Transformado)')
plt.xlabel('ln(Usuarios)')
plt.ylabel('CPU (%)')

plt.tight_layout()
plt.show() 

# ===========================================================================
# PREGUNTA 1
# ===========================================================================

# Definir features originales y target
features_cols = ['Usuarios', 'Tareas', 'Memoria']
target_col = 'CPU'

X_raw = df[features_cols].values.astype(float)
y = df[target_col].values

# Aplicar ln() solo a la primera columna (Usuarios)
X_trans = np.copy(X_raw)
X_trans[:, 0] = np.log(X_raw[:, 0]) 

# Matriz para Ecuación Normal (Sin normalizar, con columna de unos)
X_ec_normal = np.c_[np.ones(X_trans.shape[0]), X_trans]

# ===========================================================================
# PREGUNTA 2 - GRADIENT DESCENT
# ===========================================================================

def normalizar_datos(X):
    # Z-Score: (X - media) / desviacion
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

# Normalizamos las features transformadas (sin la columna de unos aún)
X_norm_features, mu, sigma = normalizar_datos(X_trans)

# Matriz para GD (Normalizada, con columna de unos)
X_gd = np.c_[np.ones(X_norm_features.shape[0]), X_norm_features]

print("\n--- RESULTADOS GRADIENT DESCENT (NORMALIZADO Z-SCORE) ---")

def gradient_descent_multivariable(X, y, lr=0.005, epocas=3000):
    n_muestras, n_features = X.shape
    w = np.zeros(n_features) # 1. INICIALIZACIÓN CON CEROS
    historial_costo = []

    for i in range(epocas):
        # 2. FORWARD: Predicciones
        y_pred = np.dot(X, w)
        
        # 3. COSTO (MSE)
        error = y - y_pred
        costo = np.mean(error**2)
        historial_costo.append(costo)

        # 4. GRADIENTE
        gradiente = (-2 / n_muestras) * np.dot(X.T, error)

        # 5. ACTUALIZACIÓN
        w = w - lr * gradiente

    return w, historial_costo

# Entrenar el modelo
W_gd, historial = gradient_descent_multivariable(X_gd, y, lr=0.005, epocas=3000)

print(f"w0_norm (Bias): {W_gd[0]:.5f}")
print(f"w1_norm (ln(Usuarios)): {W_gd[1]:.5f}")
print(f"w2_norm (Tareas): {W_gd[2]:.5f}")
print(f"w3_norm (Memoria): {W_gd[3]:.5f}")

# Graficar la curva de aprendizaje (opcional pero recomendado)
plt.figure(figsize=(8, 4))
plt.plot(historial, color='purple', linewidth=2)
plt.title('Caída del Error en Gradient Descent (lr=0.005)')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# ===========================================================================
# PREGUNTA 3 - SOLUCIÓN ANALÍTICA (ECUACIÓN NORMAL)
# ===========================================================================

print("\n--- RESULTADOS EC. NORMAL (SIN NORMALIZAR) ---")

# Fórmula: W = (X^T * X)^-1 * X^T * Y
X_T_X = np.dot(X_ec_normal.T, X_ec_normal)
Inversa = np.linalg.inv(X_T_X)
Inversa_X_T = np.dot(Inversa, X_ec_normal.T)
W_normal = np.dot(Inversa_X_T, y)

print(f"w0 (Bias): {W_normal[0]:.5f}")
print(f"w1 (ln(Usuarios)): {W_normal[1]:.5f}")
print(f"w2 (Tareas): {W_normal[2]:.5f}")
print(f"w3 (Memoria): {W_normal[3]:.5f}")

# Cálculo del R2
y_pred_normal = np.dot(X_ec_normal, W_normal)
# SS_res = suma de errores al cuadrado
SS_res = np.sum((y - y_pred_normal)**2)
# SS_tot = suma de distancias al promedio al cuadrado
SS_tot = np.sum((y - np.mean(y))**2)
R2_normal = 1 - (SS_res / SS_tot)

print(f"R² (Ecuación Normal): {R2_normal:.5f}")

# ===========================================================================
# PREGUNTA 4
# ===========================================================================

print("\n--- PREGUNTA 4: PREDICCIÓN Y ANÁLISIS ---")

# Datos del nuevo caso
usuarios_nuevo = 400
tareas_nuevo = 10
memoria_nuevo = 8

# 1. Aplicar feature engineering (ln a los usuarios)
ln_usuarios_nuevo = np.log(usuarios_nuevo)
x_nuevo_trans = np.array([ln_usuarios_nuevo, tareas_nuevo, memoria_nuevo])

# 2. Normalizar ESTRICTAMENTE con el mu y sigma del conjunto de entrenamiento
x_nuevo_norm = (x_nuevo_trans - mu) / sigma

# 3. Agregar la columna de 1 (Bias Trick)
x_nuevo_final = np.insert(x_nuevo_norm, 0, 1.0)

# 4. Predecir (Producto punto con los pesos de GD)
prediccion_cpu = np.dot(x_nuevo_final, W_gd)

print(f"Nuevo servidor -> Usuarios: {usuarios_nuevo}, Tareas: {tareas_nuevo}, Memoria: {memoria_nuevo}GB")
print(f"Predicción de consumo de CPU: {prediccion_cpu:.2f}%")