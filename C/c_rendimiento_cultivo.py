import numpy as np
import matplotlib.pyplot as plt

def aplicar_normal(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y
    
# Aplicar gradiente descendente

def normalizar(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def gradiente_descendente(X, y, alpha=0.0001, max_iter=10000, epsilon=1e-6):
    n_muestras, n_features = X.shape

    # Inicializar pesos (ceros)
    w = np.zeros(n_features)  # Inicializar pesos
    
    historial_costos = []
    historial_pesos = []

    for i in range(max_iter):
        # Predicción
        y_pred = np.dot(X, w)

        # Calcular el error (MSE)
        error = y - y_pred
        costo = np.mean(error**2)
        historial_costos.append(costo)
        historial_pesos.append(w.copy())

        # Calular gradiente
        gradiente = (-2 / n_muestras) * np.dot(X.T, error)

        # Actualizar pesos
        w_nuevo = w - alpha * gradiente

        # Verificar convergencia
        if i > 0 and abs(historial_costos[-2] - historial_costos[-1]) < epsilon:
            print(f"Convergencia alcanzada en la iteración {i}.")
            historial_pesos.append(w_nuevo.copy())
            break
            
        w = w_nuevo

    return w, historial_costos, historial_pesos


# 4. Encontrar el agua y temperatura óptima

if __name__ == "__main__":
    
    # Datos del caso
    agua = np.array([256, 783, 490, 346, 519, 371, 718, 225, 265, 660, 231, 665])
    temp = np.array([15.2, 27.4, 31.4, 33.0, 34.7, 24.9, 22.0, 32.3, 22.9, 27.9, 15.4, 31.2])
    rendimiento = np.array([990.8, 1015.5, 1268.4, 1226.4, 1226.5, 1283.1, 1168.1, 1063.9, 1088.2, 1207.8, 904.1, 1212.8])

    # ==========================================
    # Paso 2: Aplicar Ecuación Normal y Gradiente Descendente
    # ==========================================

    # 1. Definir el vector de features

    m = len(agua)
    x0 = np.ones(m)
    x1 = agua
    x2 = temp
    x3 = agua**2
    x4 = temp**2

    # 2. Preparar matrices (X e y)

    X = np.column_stack((x0, x1, x2, x3, x4))
    y = rendimiento.flatten()

    # Aplicar ecuación normal

    ec_normal = aplicar_normal(X, y)

    print("Pesos (Theta) obtenidos con la Ecuación Normal:")
    print(f"w0 (Bias):   {ec_normal[0]:.4f}")
    print(f"w1 (Agua):   {ec_normal[1]:.4f}")
    print(f"w2 (Temp):   {ec_normal[2]:.4f}")
    print(f"w3 (Agua^2): {ec_normal[3]:.6f}")
    print(f"w4 (Temp^2): {ec_normal[4]:.4f}")

    # Aplicar gradiente descendente

    # Normalizar solo las características (sin el bias)
    X_norm, mu, sigma = normalizar(X[:, 1:])  

    # Agregar la columna de unos para el bias (w0)
    X_norm = np.column_stack((np.ones(m), X_norm))

    w_gd, costos, pesos = gradiente_descendente(X_norm, y, alpha = 0.1)

    print("Pesos (Theta) obtenidos con Gradiente Descendente:")
    print(f"w0 (Bias):   {w_gd[0]:.4f}")
    print(f"w1 (Agua):   {w_gd[1]:.4f}")
    print(f"w2 (Temp):   {w_gd[2]:.4f}")
    print(f"w3 (Agua^2): {w_gd[3]:.6f}")
    print(f"w4 (Temp^2): {w_gd[4]:.4f}")
    print("-" * 50) 

    # ==========================================
    # Gráfica de la Caída del Gradiente
    # ==========================================
    print("\nGenerando gráfica de la curva de costo...")
    
    plt.figure(figsize=(8, 5))
    
    plt.plot(costos, color='royalblue', linewidth=2)
    
    plt.yscale('log') 
    
    plt.title('Curva de Aprendizaje (Escala Logarítmica)', fontsize=14, fontweight='bold')
    plt.xlabel('Número de Iteraciones', fontsize=12)
    plt.ylabel('Costo o Error (Log MSE)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

    # ==========================================
    # Paso 3: Cálculo del R²
    # ==========================================

    # 1. Calculamos las predicciones del modelo (y_hat)
    y_pred = np.dot(X, ec_normal)
    
    # 2. Calculamos la Suma de Cuadrados de los Residuos (SS_res)
    # Es el error que nuestro modelo NO pudo explicar
    ss_res = np.sum((y - y_pred)**2)
    
    # 3. Calculamos la Suma de Cuadrados Totales (SS_tot)
    # Es la variabilidad total de los datos respecto a su promedio
    ss_tot = np.sum((y - np.mean(y))**2)
    
    # 4. Fórmula del R²
    r_cuadrado = 1 - (ss_res / ss_tot)
    
    print(f"Resultado del R²: {r_cuadrado:.4f}")

    # ==========================================
    # Comparación: ¿El modelo es mejor que una regresión lineal simple?
    # ==========================================
    print("\n--- COMPARACIÓN CON REGRESIÓN LINEAL SIMPLE ---")

    # 1. Creamos la matriz X solo con los términos lineales
    # Vector: [1, agua, temp]
    X_lineal = np.column_stack((x0, x1, x2))

    # 2. Encontramos los pesos usando la Ecuación Normal
    w_lineal = aplicar_normal(X_lineal, y)

    # 3. Calculamos las predicciones de este modelo lineal
    y_pred_lineal = np.dot(X_lineal, w_lineal)

    # 4. Calculamos su R²
    ss_res_lineal = np.sum((y - y_pred_lineal)**2)
    # Usamos el ss_tot de antes
    r_cuadrado_lineal = 1 - (ss_res_lineal / ss_tot)

    print(f"R² del Modelo Cuadrático (M=2): {r_cuadrado:.4f}")
    print(f"R² del Modelo Lineal     (M=1): {r_cuadrado_lineal:.4f}")
    print("-" * 50)
    
    # Conclusión
    if r_cuadrado > r_cuadrado_lineal:
        mejora = (r_cuadrado - r_cuadrado_lineal) * 100
        print(f"CONCLUSIÓN: El modelo cuadrático es mejor.")
        print(f"Explica un {mejora:.1f}% más de la variabilidad de los datos que el lineal.")
    else:
        print("CONCLUSIÓN: El modelo lineal es igual o mejor.")

    # ==========================================
    # Paso 4: Encontrar el agua y temperatura óptima
    # ==========================================
    print("\n--- PASO 4: CONDICIONES ÓPTIMAS ---")

    # Extraemos los pesos calculados por la ecuación normal
    w0 = ec_normal[0]
    w1 = ec_normal[1]
    w2 = ec_normal[2]
    w3 = ec_normal[3]
    w4 = ec_normal[4]

    # Aplicamos la fórmula del vértice para cada variable
    agua_optima = -w1 / (2 * w3)
    temp_optima = -w2 / (2 * w4)

    # Opcional: Calcular cuál sería el rendimiento máximo con estas condiciones
    rendimiento_max = w0 + (w1 * agua_optima) + (w2 * temp_optima) + (w3 * agua_optima**2) + (w4 * temp_optima**2)

    print(f"Cantidad de agua óptima:     {agua_optima:.2f}")
    print(f"Temperatura óptima:          {temp_optima:.2f} grados")
    print(f"Rendimiento máximo esperado: {rendimiento_max:.2f}")
    print("-" * 50)

    # ==========================================
    # Comparación: Condiciones Óptimas con Gradiente Descendente
    # ==========================================
    print("\n--- COMPARACIÓN: ÓPTIMOS CON GRADIENTE DESCENDENTE ---")

    # Extraemos los pesos calculados por GD
    w0_gd = w_gd[0]
    w1_gd = w_gd[1]
    w2_gd = w_gd[2]
    w3_gd = w_gd[3]
    w4_gd = w_gd[4]

    # Ajustar fórmula del vértice usando las desviaciones estándar (sigma) que guardamos de la función normalizar()
    agua_optima_gd = - (w1_gd * sigma[2]) / (2 * w3_gd * sigma[0])
    temp_optima_gd = - (w2_gd * sigma[3]) / (2 * w4_gd * sigma[1])

    # Normalizar estos valores óptimos con las medias y desviaciones estándar originales
    agua_opt_norm = (agua_optima_gd - mu[0]) / sigma[0]
    temp_opt_norm = (temp_optima_gd - mu[1]) / sigma[1]
    agua2_opt_norm = (agua_optima_gd**2 - mu[2]) / sigma[2]
    temp2_opt_norm = (temp_optima_gd**2 - mu[3]) / sigma[3]

    # Calculamos el rendimiento máximo
    rendimiento_max_gd = w0_gd + (w1_gd * agua_opt_norm) + (w2_gd * temp_opt_norm) + (w3_gd * agua2_opt_norm) + (w4_gd * temp2_opt_norm)

    print(f"Cantidad de agua óptima (GD):     {agua_optima_gd:.2f}")
    print(f"Temperatura óptima (GD):          {temp_optima_gd:.2f} grados")
    print(f"Rendimiento máximo esperado (GD): {rendimiento_max_gd:.2f}")
    print("-" * 50)
    