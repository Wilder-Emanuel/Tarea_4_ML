import math
import matplotlib.pyplot as plt
import numpy as np
 
def predecir(x, w, b):
    return w * x + b
 
def calcular_mse(datos, w, b):
    n = len(datos)
    return sum((y - predecir(x, w, b))**2 for x, y in datos) / n
 
def calcular_mae(datos, w, b):
    n = len(datos)
    return sum(abs(y - predecir(x, w, b)) for x, y in datos) / n
 
def gradiente_mse(datos, w, b):
    n = len(datos)
    grad_w, grad_b = 0.0, 0.0
    for x, y in datos:
        error   = y - predecir(x, w, b)
        grad_w += -2 * x * error
        grad_b += -2 * error
    return grad_w / n, grad_b / n
 
def gradiente_una_muestra(x, y, w, b):
    error  = y - predecir(x, w, b)
    grad_w = -2 * x * error
    grad_b = -2 * error
    return grad_w, grad_b
 
def solucion_analitica(datos):
    n   = len(datos)
    sx  = sum(x     for x, y in datos)
    sy  = sum(y     for x, y in datos)
    sxy = sum(x * y for x, y in datos)
    sx2 = sum(x**2  for x, y in datos)
    m   = (n * sxy - sx * sy) / (n * sx2 - sx**2)
    b   = (sy - m * sx) / n
    return m, b
 
def calcular_r2(datos, w, b):
    y_med  = sum(y for _, y in datos) / len(datos)
    ss_tot = sum((y - y_med)**2           for _, y in datos)
    ss_res = sum((y - predecir(x,w,b))**2 for x, y in datos)
    return 1 - (ss_res / ss_tot)
 
def gradient_descent(datos, lr=0.01, iteraciones=1000):
    w, b      = 0.0, 0.0
    historial = []
    for i in range(iteraciones):
        historial.append((i, w, b, calcular_mse(datos, w, b)))
        gw, gb = gradiente_mse(datos, w, b)
        w -= lr * gw
        b -= lr * gb
    historial.append((iteraciones, w, b, calcular_mse(datos, w, b)))
    return w, b, historial
 
def plot_modelo(datos, w, b, titulo="Regresión lineal"):
    xs, ys = zip(*datos)
    xs_ord = sorted(xs)
    plt.figure()
    plt.scatter(xs, ys, label="Datos reales", zorder=3)
    plt.plot(xs_ord, [predecir(x, w, b) for x in xs_ord],
             label=f"Modelo: y = {w:.2f}x + {b:.2f}")
    plt.xlabel("X"); plt.ylabel("Y"); plt.title(titulo)
    plt.legend(); plt.tight_layout(); plt.show()
 
def predecir_log(d, w0, w1):
    return w0 + w1 * math.log(d)
 
def calcular_mse_log(datos, w0, w1):
    n = len(datos)
    return sum((y - predecir_log(d, w0, w1))**2 for d, y in datos) / n
 
def calcular_mae_log(datos, w0, w1):
    n = len(datos)
    return sum(abs(y - predecir_log(d, w0, w1)) for d, y in datos) / n
 
def calcular_r2_log(datos, w0, w1):
    y_med  = sum(y for _, y in datos) / len(datos)
    ss_tot = sum((y - y_med)**2                    for _, y in datos)
    ss_res = sum((y - predecir_log(d, w0, w1))**2  for d, y in datos)
    return 1 - (ss_res / ss_tot)
 
def gradiente_mse_log(datos, w0, w1):
    n = len(datos)
    acc_w0, acc_w1 = 0.0, 0.0
    for d, y in datos:
        residuo = predecir_log(d, w0, w1) - y
        acc_w0 += residuo
        acc_w1 += residuo * math.log(d)
    factor = 2.0 / n
    return factor * acc_w0, factor * acc_w1
 
def solucion_analitica_log(datos):
    n   = len(datos)
    zs  = [math.log(d) for d, _ in datos]
    ys  = [y           for _, y in datos]
    sz  = sum(zs)
    sy  = sum(ys)
    szy = sum(z * y for z, y in zip(zs, ys))
    sz2 = sum(z**2  for z    in zs)
    w1  = (n * szy - sz * sy) / (n * sz2 - sz**2)
    w0  = (sy - w1 * sz) / n
    return w0, w1
 
def gradient_descent_log(datos, lr=0.05, iteraciones=1000):
    w0, w1    = 0.0, 0.0
    historial = []
    for i in range(iteraciones):
        historial.append((i, w0, w1, calcular_mse_log(datos, w0, w1)))
        gw0, gw1 = gradiente_mse_log(datos, w0, w1)
        w0 -= lr * gw0
        w1 -= lr * gw1
    historial.append((iteraciones, w0, w1, calcular_mse_log(datos, w0, w1)))
    return w0, w1, historial
 
def plot_modelo_log(datos, w0, w1, titulo="Regresión logarítmica"):
    ds, ys  = zip(*datos)
    d_curva = np.linspace(min(ds), max(ds), 300)
    y_curva = [predecir_log(d, w0, w1) for d in d_curva]
    plt.figure()
    plt.scatter(ds, ys, label="Datos reales", zorder=3)
    plt.plot(d_curva, y_curva, color="tomato",
             label=f"L(d) = {w0:.3f} + {w1:.3f}·ln(d)")
    plt.xlabel("d"); plt.ylabel("L(d)"); plt.title(titulo)
    plt.legend(); plt.tight_layout(); plt.show()
 
def plot_historial_loss(historial, titulo="Curva de pérdida (MSE)"):
    iters = [h[0] for h in historial]
    mses  = [h[3] for h in historial]
    plt.figure()
    plt.plot(iters, mses, color="steelblue")
    plt.xlabel("Iteración"); plt.ylabel("MSE"); plt.title(titulo)
    plt.tight_layout(); plt.show()