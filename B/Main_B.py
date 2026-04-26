import matplotlib.pyplot as plt
import Funciones_Univariable as fu
import Funciones_Multivariable as fm
import Funciones_Analisis as fa

# ===========================================================================
# EXTRAER DATOS
# ===========================================================================

ruta = "Ejercicio_B.csv"
datos = []

with open(ruta, "r", encoding="utf-8") as f:
    encabezado = f.readline().strip().split(",")

    idx_x1 = encabezado.index("area")
    idx_x2 = encabezado.index("bedrooms")
    idx_x3 = encabezado.index("age")
    idx_y = encabezado.index("price")

    for linea in f:
        partes = linea.strip().split(",")
        x1 = float(partes[idx_x1])   # area
        x2 = float(partes[idx_x2])   # bedrooms
        x3 = float(partes[idx_x3])   # age
        y  = float(partes[idx_y])    # price
        datos.append((x1, x2, x3, y))

# ===========================================================================
# ANALISIS DE COMPORTAMIENTO DE LOS DATOS
# ===========================================================================

x1_vals, x2_vals, x3_vals, y_vals = [], [], [], []

for x1, x2, x3, y in datos:
    x1_vals.append(x1)
    x2_vals.append(x2)
    x3_vals.append(x3)
    y_vals.append(y)

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# area vs price
axs[0].scatter(x1_vals, y_vals)
axs[0].set_title("area vs price")
axs[0].set_xlabel("area")
axs[0].set_ylabel("price")

# bedrooms vs price
axs[1].scatter(x2_vals, y_vals)
axs[1].set_title("bedrooms vs price")
axs[1].set_xlabel("bedrooms")
axs[1].set_ylabel("price")

# age vs price
axs[2].scatter(x3_vals, y_vals)
axs[2].set_title("age vs price")
axs[2].set_xlabel("age")
axs[2].set_ylabel("price")

plt.tight_layout()
plt.show()

# ===========================================================================
# ANALISIS DE MODELO UNIVARIABLE PARA CADA FEATURE
# ===========================================================================

datos_norm, _ = fm.normalizar_zscore_multivariable(datos, num_features=3)

datos_x1 = [(fila[0], fila[3]) for fila in datos_norm]  # area
datos_x2 = [(fila[1], fila[3]) for fila in datos_norm]  # bedrooms
datos_x3 = [(fila[2], fila[3]) for fila in datos_norm]  # age

w1, b1, _ = fu.gradient_descent(datos_x1, lr=0.01, iteraciones=2000)
w2, b2, _ = fu.gradient_descent(datos_x2, lr=0.01, iteraciones=2000)
w3, b3, _ = fu.gradient_descent(datos_x3, lr=0.01, iteraciones=2000)

fu.plot_modelo(datos_x1, w1, b1, "area vs price (normalizado)")
fu.plot_modelo(datos_x2, w2, b2, "bedrooms vs price (normalizado)")
fu.plot_modelo(datos_x3, w3, b3, "age vs price (normalizado)")

# ===========================================================================
# ANALISIS R² POR FEATURE
# ===========================================================================
 
r2_x1 = fa.calcular_r2(datos_x1, w1, b1)
r2_x2 = fa.calcular_r2(datos_x2, w2, b2)
r2_x3 = fa.calcular_r2(datos_x3, w3, b3)
 
print("\n" + "=" * 50)
print(" R² — Coeficiente de Determinación por Feature")
print("=" * 50)
print(f"  area     (x1): R² = {r2_x1:.4f}")
print(f"  bedrooms (x2): R² = {r2_x2:.4f}")
print(f"  age      (x3): R² = {r2_x3:.4f}")
print("=" * 50)

# ===========================================================================
# MODELO MULTIVARIABLE
# ===========================================================================

# Construir matrices X (con columna de bias) e y a partir de datos normalizados
X, y = fm.construir_matrices(datos_norm, num_features=3)
 
# Inicializar pesos: [w0_bias, w1_area, w2_bedrooms, w3_age]
w_init = [1.0, 61980.5256, 14924.2253, 13308.9262]
 
# Entrenar con Gradient Descent multivariable
w_multi, historial_multi = fm.gradient_descent_multivariable(
    X, y,
    w_init      = w_init,
    lr          = 0.05,
    iteraciones = 2000,
    epsilon     = 1e-8,
    verbose     = True,
    mostrar_cada= 300,
    nombre      = "area (x1), bedrooms (x2), age (x3) → price (y)"
)

# ===========================================================================
# ANALISIS R² — MODELO MULTIVARIABLE
# ===========================================================================
 
r2_multi = fa.calcular_r2_multivariable(X, y, w_multi)
 
print("\n" + "=" * 50)
print(" R² - Modelo Multivariable")
print("=" * 50)
print(f"  area (x1), bedrooms (x2), age (x3): R² = {r2_multi:.4f}")
print("=" * 50)
 
fm.graficar_convergencia_3_lr(X, y, w_init)

