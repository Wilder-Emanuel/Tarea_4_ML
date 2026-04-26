import matplotlib.pyplot as plt
import Funciones_Univariable as fu
import Funciones_Multivariable as fm
import Funciones_Analisis as fa
import Funciones_Polinomial as fp
import Funciones_Logistica as flog

# ===========================================================================
# EXTRAER DATOS
# ===========================================================================

ruta = "Ejercicio_A.csv"
datos = []

with open(ruta, "r", encoding="utf-8") as f:
    encabezado = f.readline().strip().split(",")

    idx_x = encabezado.index("distance")
    idx_y = encabezado.index("levelL")

    for linea in f:
        partes = linea.strip().split(",")
        x = float(partes[idx_x])
        y = float(partes[idx_y])
        datos.append((x, y))

# ===========================================================================
# ANALISIS DE COMPORTAMIENTO DE LOS DATOS
# ===========================================================================

xs = [x for x, y in datos]
ys = [y for x, y in datos]

plt.scatter(xs, ys)
plt.title("distance vs levelL (datos reales)")
plt.xlabel("distance")
plt.ylabel("levelL")
plt.show()

# ===========================================================================
# MODELO LINEAL
# ===========================================================================

datos_norm, _ = fm.normalizar_zscore_multivariable(datos, num_features=1)
datos_x = [(fila[0], fila[1]) for fila in datos_norm]

w, b, _ = fu.gradient_descent(datos_x, lr=0.05, iteraciones=2000)
fu.plot_modelo(datos_x, w, b, "Modelo lineal (distance vs levelL)")

r2_x1 = fa.calcular_r2(datos_x, w, b)
print("\n" + "=" * 50)
print(" R² — Coeficiente de Determinación por Feature")
print("=" * 50)
print(f"  area     (x1): R² = {r2_x1:.4f}")
print("=" * 50)

# ===========================================================================
# MODELO POLINOMIAL
# ===========================================================================

datos_norm_pol, _ = fm.normalizar_zscore_multivariable(datos, num_features=1)
datos_pol = [(fila[0], fila[1]) for fila in datos_norm_pol]

w2, w1, b_pol, historial_pol = fp.gradient_descent(
    datos_pol,
    lr=0.001,
    iteraciones=5000
)

fp.plot_modelo(datos_pol, w2, w1, b_pol, "Modelo polinomial orden 2 (distance vs levelL)")
fp.plot_historial_mse(historial_pol, "Evolución del MSE — Modelo Polinomial")

r2_pol = fp.calcular_r2(datos_pol, w2, w1, b_pol)

print(f"\n  Resultado GD : w2 = {w2:.4f} | w1 = {w1:.4f} | b = {b_pol:.4f}")
print(f"  MSE final    : {fp.calcular_mse(datos_pol, w2, w1, b_pol):.4f}")
print(f"  R²           : {r2_pol:.4f}")

# ===========================================================================
# MODELO LOGISTICO
# ===========================================================================

# Solución analitica para comparar dspues (Pregunta 3)
w0_a, w1_a = flog.solucion_analitica_log(datos)

# GD con 2000 épocas (Pregunta 2)
w0_g, w1_g, historial_log = flog.gradient_descent_log(
    datos,
    lr=0.05,
    iteraciones=2000
)

# evolución de la Loss cada 200 épocas
print("\n" + "="*60)
print(" EVOLUCIÓN DE LA LOSS (MSE) CADA 200 ÉPOCAS (Pregunta 2)")
print("="*60)
print(f"{'Época':>10} | {'Loss (MSE)':>15}")
print("-" * 60)

for h in historial_log:
    epoca = h[0]
    mse = h[3] 
    if epoca % 200 == 0:
        print(f"{epoca:>10} | {mse:>15.4f}")

# Graficar y calcular R2
flog.plot_modelo_log(datos, w0_g, w1_g, "Modelo logarítmico")
r2_log = flog.calcular_r2_log(datos, w0_g, w1_g)

# Comparación final (Pregunta 3)
print("\n" + "="*60)
print("RESULTADOS FINALES MODELO LOGARITMICO")
print("="*60)
print(f"GD       : w0 = {w0_g:.4f} | w1 = {w1_g:.4f}")
print(f"Analítica: w0 = {w0_a:.4f} | w1 = {w1_a:.4f}")
print(f"R² Logarítmico: {r2_log:.4f}")
print("="*60)