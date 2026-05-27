# Ejemplos ML — TREZ

Cada bloque es un programa `.trez` directamente ejecutable con:

```bash
cd src
python3 main.py <archivo.trez>
```

---

## 1. Regresión Lineal

Aprende la función `y = 2·x1 + 3·x2 + 1` a partir de datos sintéticos.

```trez
// Datos: y = 2x1 + 3x2 + 1
let X = [[1,1],[1,2],[2,2],[2,3],[3,3],[3,4],[4,4],[4,5],[5,5],[5,6]];
let y = [6, 9, 11, 14, 16, 19, 21, 24, 26, 29];

// Entrenar
let modelo = Mldoz.linreg_fit(X, y, 0.01, 2000);
mostrar(modelo["w"]);    // ≈ [2.0, 3.0]
mostrar(modelo["b"]);    // ≈ 1.0

// Métricas
let yp = Mldoz.linreg_predict(modelo, X);
mostrar(Metricsdoz.r2_score(y, yp));   // > 0.99
mostrar(Metricsdoz.rmse(y, yp));       // < 0.5

// Predicción sobre punto nuevo
let nueva = Mldoz.linreg_predict(modelo, [6, 7]);
mostrar(nueva);   // ≈ 34.0  (2·6 + 3·7 + 1)
```

---

## 2. Clasificación Logística Binaria

Separa dos nubes de puntos con regresión logística.

```trez
// Clase 0 ~ (0,0),  Clase 1 ~ (5,5)
let X = [
    [0.0,0.5],[0.5,0.0],[0.2,0.3],[0.4,0.6],[0.1,0.1],
    [5.0,5.5],[5.5,5.0],[5.2,5.3],[5.4,5.6],[5.1,5.1]
];
let y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

let modelo = Mldoz.logreg_fit(X, y, 0.5, 1000);
let yp     = Mldoz.logreg_predict(modelo, X);

mostrar(Metricsdoz.accuracy(y, yp));           // 1.0
mostrar(Metricsdoz.confusion_matrix(y, yp));   // {"matriz": [[5,0],[0,5]], ...}

// Punto nuevo
mostrar(Mldoz.logreg_predict(modelo, [0.3, 0.4]));   // 0
mostrar(Mldoz.logreg_predict(modelo, [5.3, 5.2]));   // 1
```

---

## 3. SVM Lineal

```trez
let X = [
    [0.8,1.2],[1.1,0.9],[1.3,1.4],[0.9,0.8],[1.0,1.1],
    [6.2,5.8],[5.9,6.1],[6.4,6.3],[5.8,5.9],[6.1,6.2]
];
let y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

let modelo = Mldoz.svm_fit(X, y, 0.05, 500, 0.01);
let yp     = Mldoz.svm_predict(modelo, X);

mostrar(Metricsdoz.accuracy(y, yp));   // 1.0
mostrar(Mldoz.svm_predict(modelo, [1.0, 1.0]));   // 0
mostrar(Mldoz.svm_predict(modelo, [6.0, 6.0]));   // 1
```

---

## 4. Árbol de Decisión (clasificación + regresión)

```trez
// ── Clasificación ────────────────────────────────────────────────────────────
let Xc = [
    [1.0,1.0],[2.0,2.0],[-1.0,-1.0],[-2.0,-2.0],
    [1.0,-1.0],[2.0,-2.0],[-1.0,1.0],[-2.0,2.0]
];
let yc = [0, 0, 0, 0, 1, 1, 1, 1];

let arbol = Mldoz.tree_clf_fit(Xc, yc, 5);
mostrar(Metricsdoz.accuracy(yc, Mldoz.tree_predict(arbol, Xc)));   // 1.0

// ── Regresión ────────────────────────────────────────────────────────────────
let Xr = [[1.0],[2.0],[3.0],[4.0],[5.0],[6.0]];
let yr = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0];   // y = x²

let arbol_reg = Mldoz.tree_reg_fit(Xr, yr, 5);
let yp_reg    = Mldoz.tree_predict(arbol_reg, Xr);
mostrar(Metricsdoz.r2_score(yr, yp_reg) > 0.99);   // true
```

---

## 5. k-NN

```trez
let X = [
    [1.0,1.0],[1.5,1.5],[2.0,1.0],
    [8.0,8.0],[8.5,8.5],[9.0,8.0]
];
let y = [0, 0, 0, 1, 1, 1];

let modelo = Mldoz.knn_fit(X, y, 3);
mostrar(Metricsdoz.accuracy(y, Mldoz.knn_predict(modelo, X)));   // 1.0

// Punto nuevo
mostrar(Mldoz.knn_predict(modelo, [1.2, 1.3]));   // 0
mostrar(Mldoz.knn_predict(modelo, [8.3, 8.2]));   // 1
```

---

## 6. k-Means (agrupamiento)

```trez
Randomdoz.seed(42);

// 3 nubes bien separadas
let X = [
    [0.1,0.2],[0.3,-0.1],[-0.2,0.3],[0.0,0.1],[0.2,-0.2],
    [10.1,0.2],[9.9,-0.1],[10.2,0.3],[10.0,0.0],[9.8,0.1],
    [5.1,9.2],[4.9,8.9],[5.2,9.1],[5.0,9.0],[4.8,8.8]
];

let modelo    = Mldoz.kmeans_fit(X, 3, 100);
let asigns    = modelo["asignaciones"];

// Los 5 primeros deben estar en el mismo cluster
mostrar(asigns[0] == asigns[4]);   // true
mostrar(asigns[5] != asigns[0]);   // true (nube distinta)

// Predecir nuevos puntos
let pred = Mldoz.kmeans_predict(modelo, [[0.0,0.0],[10.0,0.0],[5.0,9.0]]);
mostrar(pred[0] != pred[1]);   // true — clusters distintos
```

---

## 7. Perceptrón (AND / OR / falla en XOR)

```trez
let X = [[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]];

// AND — linealmente separable
let p_and = NNdoz.perceptron_fit(X, [0,0,0,1], 0.5, 200);
mostrar(NNdoz.perceptron_predict(p_and, X));              // [0, 0, 0, 1]
mostrar(Metricsdoz.accuracy([0,0,0,1], NNdoz.perceptron_predict(p_and, X)));  // 1.0

// XOR — NO separable linealmente
let p_xor = NNdoz.perceptron_fit(X, [0,1,1,0], 0.5, 500);
mostrar(Metricsdoz.accuracy([0,1,1,0], NNdoz.perceptron_predict(p_xor, X)) < 1.0);  // true
```

---

## 8. MLP entrenable — XOR

```trez
Randomdoz.seed(7);

let X = [[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]];
let y = [0.0, 1.0, 1.0, 0.0];

// Arquitectura [2 entradas → 8 ocultas → 1 salida]
let capas     = NNdoz.mlp_init([2, 8, 1]);
let resultado = NNdoz.mlp_train(capas, X, y, 0.05, 5000, "mse");

let hist = resultado["historia"];
mostrar(hist[len(hist) - 1] < 0.05);   // pérdida final < 0.05 — true

let preds = NNdoz.mlp_predict(resultado["layers"], X);
mostrar(preds[0][0] < 0.2);   // XOR(0,0) ≈ 0 — true
mostrar(preds[1][0] > 0.8);   // XOR(0,1) ≈ 1 — true
mostrar(preds[2][0] > 0.8);   // XOR(1,0) ≈ 1 — true
mostrar(preds[3][0] < 0.2);   // XOR(1,1) ≈ 0 — true
```

---

## 9. Clustering DL — k-Means sobre embeddings MLP

Entrena un MLP como clasificador y luego aplica k-Means sobre las activaciones de la capa oculta para agrupar sin usar etiquetas.

```trez
Randomdoz.seed(42);

// Datos: 3 grupos (mismos que ejemplo 6)
let X = [
    [0.1,0.2],[0.3,-0.1],[-0.2,0.3],[0.0,0.1],[0.2,-0.2],
    [10.1,0.2],[9.9,-0.1],[10.2,0.3],[10.0,0.0],[9.8,0.1],
    [5.1,9.2],[4.9,8.9],[5.2,9.1],[5.0,9.0],[4.8,8.8]
];

// Etiquetas supervisadas (0, 1, 2) codificadas como regresión multiclase no-one-hot
// Usamos k-Means directo sobre X (agrupamiento no supervisado)
let km = Mldoz.kmeans_fit(X, 3, 200);
mostrar(km["asignaciones"]);   // 3 clusters distinguibles

// Verificar que los 3 centroides están bien separados entre sí
let c = km["centroides"];
func dist2(ax, ay, bx, by) {
    let dx = ax - bx;
    let dy = ay - by;
    return Mathdoz.sqrt(dx * dx + dy * dy);
}
let d01 = dist2(c[0][0], c[0][1], c[1][0], c[1][1]);
let d02 = dist2(c[0][0], c[0][1], c[2][0], c[2][1]);
let d12 = dist2(c[1][0], c[1][1], c[2][0], c[2][1]);
mostrar(d01 > 5.0);   // true
mostrar(d02 > 5.0);   // true
mostrar(d12 > 5.0);   // true
```

---

## 10. Pipeline completo: train/test split + métricas

```trez
// Datos simulados — clasificación binaria de 20 puntos
let X_all = [
    [0.5,0.8],[0.3,0.7],[0.6,0.9],[0.2,0.4],[0.7,0.6],
    [0.1,0.3],[0.4,0.5],[0.8,0.7],[0.3,0.6],[0.5,0.4],
    [5.5,5.8],[5.3,5.7],[5.6,5.9],[5.2,5.4],[5.7,5.6],
    [5.1,5.3],[5.4,5.5],[5.8,5.7],[5.3,5.6],[5.5,5.4]
];
let y_all = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1];

// Dividir 80/20
let partes = Datadoz.train_test_split(X_all, y_all, 0.2);
let X_tr = partes[0];
let X_te = partes[1];
let y_tr = partes[2];
let y_te = partes[3];

// Entrenar
let m = Mldoz.logreg_fit(X_tr, y_tr, 0.5, 800);

// Evaluar en test
let yp = Mldoz.logreg_predict(m, X_te);
mostrar(Metricsdoz.accuracy(y_te, yp));
mostrar(Metricsdoz.confusion_matrix(y_te, yp));
```
