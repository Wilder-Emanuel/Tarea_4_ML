import Funciones_Univariable as fu
import Funciones_Multivariable as fm
 
# ===========================================================================
# R CUADRADO (R²)
# ===========================================================================
 
def calcular_r2(datos, w, b):
    n = len(datos)
 
    y_media = sum(y for _, y in datos) / n
    ss_tot = sum((y - y_media) ** 2 for _, y in datos)
    ss_res = sum((y - fu.predecir(x, w, b)) ** 2 for x, y in datos)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
 
    return 1 - ss_res / ss_tot
 
def calcular_r2_multivariable(X, y, w):

    n = len(y)
    y_media = sum(y) / n
    ss_tot = sum((yi - y_media) ** 2 for yi in y)
    y_hat = fm.predecir_matricial(X, w)
    ss_res = sum((y[i] - y_hat[i]) ** 2 for i in range(n))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
 
    return 1 - ss_res / ss_tot
