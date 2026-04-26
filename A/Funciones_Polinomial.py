import matplotlib.pyplot as plt
import numpy as np

# ===========================================================================
# FUNCIONES BASE
# ===========================================================================

def predecir(x, w2, w1, b):
    """y = w2*x^2 + w1*x + b"""
    return w2 * x**2 + w1 * x + b

def calcular_mse(datos, w2, w1, b):
    n = len(datos)
    total = sum((y - predecir(x, w2, w1, b))**2 for x, y in datos)
    return total / n

def calcular_mae(datos, w2, w1, b):
    n = len(datos)
    total = sum(abs(y - predecir(x, w2, w1, b)) for x, y in datos)
    return total / n

def gradiente_mse(datos, w2, w1, b):
    n = len(datos)
    grad_w2 = 0.0
    grad_w1 = 0.0
    grad_b  = 0.0
    for x, y in datos:
        error = y - predecir(x, w2, w1, b)
        grad_w2 += -2 * (x**2) * error
        grad_w1 += -2 * x      * error
        grad_b  += -2          * error
    return grad_w2 / n, grad_w1 / n, grad_b / n

def gradiente_una_muestra(x, y, w2, w1, b):
    error = y - predecir(x, w2, w1, b)
    grad_w2 = -2 * (x**2) * error
    grad_w1 = -2 * x      * error
    grad_b  = -2          * error
    return grad_w2, grad_w1, grad_b

def calcular_r2(datos, w2, w1, b):
    y_media = sum(y for x, y in datos) / len(datos)
    ss_tot  = sum((y - y_media)**2             for x, y in datos)
    ss_res  = sum((y - predecir(x, w2, w1, b))**2 for x, y in datos)
    return 1 - (ss_res / ss_tot)

def solucion_analitica(datos):
    X    = np.array([[x**2, x, 1] for x, y in datos])
    y    = np.array([y for x, y in datos])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    w2, w1, b = beta
    return w2, w1, b

def separador(titulo):
    linea = "=" * 60
    print(f"\n{linea}")
    print(f"  {titulo}")
    print(linea)

# ===========================================================================
# GRADIENT DESCENT (GD)
# ===========================================================================

def gradient_descent(datos, lr=0.001, iteraciones=5000):
    w2, w1, b = 0.0, 0.0, 0.0
    historial = []

    for i in range(iteraciones):
        mse = calcular_mse(datos, w2, w1, b)
        historial.append((i, w2, w1, b, mse))

        gw2, gw1, gb = gradiente_mse(datos, w2, w1, b)

        w2 -= lr * gw2
        w1 -= lr * gw1
        b  -= lr * gb

    mse_final = calcular_mse(datos, w2, w1, b)
    historial.append((iteraciones, w2, w1, b, mse_final))

    return w2, w1, b, historial

# ===========================================================================
# GRÁFICOS
# ===========================================================================

def plot_modelo(datos, w2, w1, b, titulo="Regresión polinómica (orden 2)"):
    xs = [x for x, y in datos]
    ys = [y for x, y in datos]

    plt.figure(figsize=(8, 5))
    plt.scatter(xs, ys, color="steelblue", label="Datos reales", zorder=3)

    xs_linea = np.linspace(min(xs), max(xs), 300)
    ys_pred  = [predecir(x, w2, w1, b) for x in xs_linea]

    etiqueta = f"y = {w2:.3f}x² + {w1:.3f}x + {b:.3f}"
    plt.plot(xs_linea, ys_pred, color="tomato", linewidth=2, label=etiqueta)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_historial_mse(historial, titulo="Evolución del MSE"):
    iters = [h[0] for h in historial]
    mses  = [h[4] for h in historial]

    plt.figure(figsize=(8, 4))
    plt.plot(iters, mses, color="darkorange", linewidth=1.5)
    plt.xlabel("Iteración")
    plt.ylabel("MSE")
    plt.title(titulo)
    plt.tight_layout()
    plt.show()