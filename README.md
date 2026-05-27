# PELE - Lenguaje de Programación para Deep Learning y Machine Learning

Este repositorio contiene el desarrollo del lenguaje de programación **PELE**, diseñado bajo un **paradigma funcional puro** y enfocado enteramente en el diseño, preparación de datos, entrenamiento y evaluación de modelos de **Machine Learning y Deep Learning**. 

El núcleo del lenguaje utiliza **ANTLR4** para el análisis léxico y sintáctico (Parser/Lexer), con **Python** únicamente como máquina virtual / intérprete de bajo nivel (AST Evaluator). Siguiendo una **regla de oro estricta**, el intérprete no contiene lógica externa ni dependencias de Python; todo el ecosistema matemático, estructural y predictivo de PELE está escrito de forma nativa en archivos `.pele` bajo abstracciones funcionales puras.

---

## Características y Capacidades de Deep Learning

PELE cuenta con un ecosistema completo y autónomo de librerías para resolver problemas complejos de ciencia de datos:

1. **Estructuras de Datos 100% Nativas (`pele_structs.pele`):**
   * Pilas y Colas funcionales mediante cortes de arreglos.
   * Conjuntos inmutables con operaciones de pertenencia y unicidad.
   * Árboles con recorridos en preorden e inorden recursivos.
   * Grafos mediante listas de adyacencia dinámicas con algoritmos BFS (búsqueda a lo ancho) y DFS (búsqueda a lo profundo) recursivos.
2. **Cómputo Multidimensional y Tensores (`pele_numpy.pele`, `pele_tensor.pele`):**
   * Creación de arrays de NumPy nativos (`np_array`).
   * Operaciones matriciales complejas: transposición, suma, resta, producto escalar, producto por escalares y multiplicación de matrices en 2D (`np_matmul`).
   * Atributos de dimensiones (`np_shape`).
3. **Data Wrangling & ETL con Pandas (`pele_pandas.pele`):**
   * **`pd_read_csv`**: Carga y parsea textos planos CSV de manera dinámica reconociendo cabeceras e infiriendo tipos numéricos (enteros y decimales).
   * Proyección y selección de subconjuntos de columnas, eliminación inmutable de características y limpieza de nulos (`pd_fill_na`).
   * **`pd_one_hot_encode`**: Codificación One-Hot para convertir columnas de categorías de texto a matrices binarias ($0$/$1$), indispensables para alimentar clasificadores.
4. **Algoritmos y Modelos Predictivos (`pele_ml.pele`):**
   * **Perceptrón de Rosenblatt:** Clasificador lineal entrenado nativamente.
   * **Regresión Lineal Simple y Múltiple:** Ajuste de pesos mediante Descenso de Gradiente estocástico.
   * **K-Nearest Neighbors (KNN):** Clasificador de vecinos más cercanos usando métricas de distancia Euclidiana.
   * **K-Means Clustering:** Agrupación espacial no supervisada de centroides dinámicos.
5. **Métricas de Rendimiento Avanzadas (`pele_metrics.pele`):**
   * Métricas de clasificación: Exactitud (Accuracy), Precisión, Sensibilidad (Recall) y Puntuación F1.
   * Matriz de confusión multiclasificación de dos dimensiones generada dinámicamente.
   * Coeficiente de determinación ($R^2$) para evaluar modelos de regresión.
6. **Redes Neuronales Artificiales (`pele_nn.pele`):**
   * Inicialización de capas neuronales (pesos y sesgos) y propagación hacia adelante.
7. **Visualización y Gráficos Nativos (`pele_plot.pele`):**
   * Exportación de curvas de aprendizaje y gráficos de dispersión (Scatter Plots) a archivos vectoriales SVG nativos.

---

## Estructura del Proyecto

La arquitectura del lenguaje se divide en la infraestructura del intérprete y las librerías nativas:

### Núcleo del Intérprete
*   `PELE.g4`: Gramática formal en ANTLR4. Define la precedencia matemática correcta (desde acceso a índices hasta tuberías logicas) y la sintaxis.
*   `visitorPELE.py`: Árbol de Sintaxis Abstracta (AST). Evaluador semántico puro libre de lógica de librerías.
*   `pele.py`: Archivo ejecutable del compilador. Inyecta el preludio de librerías en orden de dependencia y corre el código.
*   `programa.txt`: Suite integradora de pruebas y dashboard visual.

### Librerías Nativas (`.pele`)
*   `pele_math.pele`: Funciones matemáticas basales (potencia, exp, log, min, max, absoluto y funciones de activación como Sigmoide y ReLU).
*   `pele_numpy.pele`: Abstracción de tensores multidimensionales y álgebra lineal.
*   `pele_structs.pele`: Estructuras de datos puras (pilas, colas, conjuntos, árboles y grafos).
*   `pele_pandas.pele`: Parsea y manipula DataFrames cargados de strings CSV.
*   `pele_tensor.pele`: Estructura base para el flujo de gradientes de tensores.
*   `pele_random.pele`: Generador pseudoaleatorio lineal congruente (LCG) nativo para reproducibilidad de modelos.
*   `pele_losses.pele`: Pérdidas de entropía cruzada binaria (BCE) y error cuadrático medio (MSE).
*   `pele_metrics.pele`: Matriz de confusión y cálculo de F1 Score, Precision, Recall y R2.
*   `pele_data.pele`: Auxiliares de data preprocesing (normalización Min-Max, estandarización Z-score).
*   `pele_ml.pele`: Algoritmos de regresión y modelos clasificadores clásicos.
*   `pele_nn.pele`: Componentes neuronales artificiales.
*   `pele_plot.pele`: Generación directa de reportes gráficos en SVG.

---

## Requisitos Previos e Instalación

1. **Entorno de Software:**
   * Python 3.x
   * Java JRE/JDK (solo si deseas compilar la gramática `PELE.g4`)
   * Herramienta ANTLR4 instalada.

2. **Instalación de Dependencias:**
   ```bash
   # Crear entorno virtual
   python3 -m venv env
   source env/bin/activate

   # Instalar runtime de ANTLR
   pip install antlr4-python3-runtime
   ```

3. **Compilación de la Gramática (Opcional si editas `PELE.g4`):**
   ```bash
   antlr4 -Dlanguage=Python3 -visitor PELE.g4
   ```

4. **Ejecutar el Dashboard de Pruebas:**
   ```bash
   python3 pele.py
   ```

---

## Ejemplo de Flujo de ML Completo en PELE

Puedes cargar un dataset CSV, limpiar características nulas, codificar variables cualitativas a binarias, entrenar un clasificador y evaluar métricas de precisión directamente en PELE:

```text
// 1. Cargar Datos con Pandas
csv_txt = "edad,salario,compra,categoria\n25,40000.0,0,Premium\n32,54000.0,1,Standard\n47,80000.0,1,Premium\n";
df = pd_read_csv(csv_txt);

// 2. Codificación One-Hot
df_codificado = pd_one_hot_encode(df, "categoria");

// 3. Selección de características y objetivos
X = pd_select_columns(df_codificado, ["edad", "salario"]);
y = pd_get_column(df_codificado, "compra");

// 4. Entrenar y Evaluar Métricas
y_real = [1, 0, 1, 1, 0, 1, 0, 0];
y_pred = [1, 0, 1, 0, 0, 1, 1, 0];

mostrar("Metricas del Modelo:");
mostrar("Precision:");
mostrar(precision(y_real, y_pred, 1)); // Retorna 0.75
mostrar("Confusion Matrix:");
mostrar(confusion_matrix(y_real, y_pred)); // Retorna [[3, 1], [1, 3]]
```

---

## Historial de Avances

*   **Fase 1 (Sintaxis Base):** Control de flujo (`si`, `mientras`, `por`), funciones, ámbitos de memoria (Scopes) y recursividad.
*   **Fase 2 (Autonomía de Librerías):** Remoción absoluta de las dependencias externas del visitor. Todo el software estructurado (grafos, árboles, pilas) fue reescrito de forma nativa en PELE.
*   **Fase 3 (Corrección del Parser):** Reordenamiento de la gramática `PELE.g4` para corregir la precedencia de ANTLR4 de arriba a abajo. Esto permitió evaluar expresiones complejas (ej. comparadores relacionales seguidos de sumas `tp + fp == 0`) de forma matemáticamente exacta.
*   **Fase 4 (Data Science & Pandas):** Creación de `pele_pandas.pele` e integración de la suite integradora visual `programa.txt` que corre y reporta el paso del 100% de los tests en la consola.