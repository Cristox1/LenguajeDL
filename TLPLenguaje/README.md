# TREZ Language

**TREZ** es un Lenguaje de Dominio Específico (DSL) funcional diseñado para la definición, entrenamiento y evaluación de modelos de Machine Learning y Deep Learning, construido completamente desde cero — cero dependencias a NumPy, PyTorch, scikit-learn o cualquier librería de álgebra lineal o ML externa.

El lenguaje adopta un **paradigma estrictamente funcional** donde las transformaciones de datos se expresan mediante composición de funciones y el operador pipe `|>`. El sistema de compilación usa **ANTLR4** para análisis léxico/sintáctico y un intérprete **Python** que implementa el patrón **Visitor** sobre el AST generado.

---

## Características Principales

- **Cero dependencias externas:** Motor matemático, tensores, funciones de activación, ML clásico, backpropagation y visualización implementados en Python puro.
- **Paradigma funcional:** Bindings inmutables (`let`), funciones puras, closures con recursión, lambdas anónimas (`\x -> expr`).
- **Patrón Visitor:** `TrezVisitor` recorre el AST nodo a nodo. Cada construcción del lenguaje tiene su `visitX()` correspondiente.
- **Operador pipe `|>`:** Encadena transformaciones: `datos |> normalizar |> relu` equivale a `relu(normalizar(datos))`.
- **Namespaces:** La stdlib se invoca como `Tensordoz.dot(A, B)`, `Mldoz.linreg_fit(X, y, lr, epochs)`, `NNdoz.mlp_train(capas, X, y, lr, epochs, "mse")`.
- **Construido con ANTLR4:** Gramática Tipo 2 (Libre de Contexto) compilada a Python.

---

## Requisitos

- Python >= 3.10
- `antlr4-python3-runtime == 4.13.2`  (`pip install antlr4-python3-runtime==4.13.2`)
- Java JRE/JDK (solo para regenerar el parser desde las gramáticas `.g4`)

---

## Uso

```bash
cd src
python3 main.py <archivo.trez>
```

Correr la suite de tests:

```bash
python3 tests/run_tests.py
```

---

## Sintaxis Rápida

```trez
// Variables (bindings inmutables)
let lr = 0.01;
let pesos = [0.5, 0.3, 0.1];
let matriz = [[1, 2], [3, 4]];

// Funciones con recursión
func factorial(n) {
    if (n <= 1) { return 1; }
    return n * factorial(n - 1);
}
mostrar(factorial(6));   // 720

// Lambda anónima
let doble = \x -> x * 2;

// Operador pipe
let resultado = [1.0, -2.0, 3.0] |> Mathdoz.relu;

// Namespace de módulo
let salida = Tensordoz.dot(W, X);
let error  = Metricsdoz.mse(y_real, y_pred);

// Condicionales encadenados
func clasificar(x) {
    if (x < 0)      { return "negativo"; }
    else if (x == 0){ return "cero"; }
    else            { return "positivo"; }
}

// Bucles
for i in range(10) { mostrar(i); }
while (cond) { mostrar("iterando"); }

// Diccionarios
let d = {nombre: "TREZ", version: 3};
mostrar(d["nombre"]);

// Estructuras de datos
let q = Queue();
let q = q.enqueue(42);
let s = Stack();
let s = s.push(10);
```

---

## Librería Estándar Nativa

Todos los módulos están implementados en Python puro — ninguna función llama a librería externa.

| Módulo | Funciones disponibles |
|---|---|
| `Mathdoz` | `relu`, `sigmoid`, `exp`, `log`, `sin`, `cos`, `tan`, `sqrt`, `abs`, `pow`, `factorial`; constantes `PI`, `E` |
| `Tensordoz` | `dot`, `transpose`, `add`, `sub`, `scale`, `zeros`, `ones`, `flatten`, `reshape`, `concat` |
| `Activationsdoz` | `relu`, `sigmoid`, `tanh`, `softmax`, `linear` y sus derivadas |
| `Randomdoz` | `seed`, `random`, `uniform`, `randint`, `choice`, `sample`, `shuffle`, `gauss` |
| `Metricsdoz` | `accuracy`, `precision`, `recall`, `f1_score`, `confusion_matrix`, `mse`, `rmse`, `mae`, `r2_score`, `cross_entropy`, `cross_entropy_grad` |
| `Mldoz` | `linreg_fit/predict`, `logreg_fit/predict/predict_proba`, `svm_fit/predict`, `tree_clf_fit`, `tree_reg_fit`, `tree_predict`, `knn_fit/predict/predict_clf/predict_reg`, `kmeans_fit/predict` |
| `NNdoz` | `linear_init/forward/backward`, `relu_forward/backward`, `softmax`, `sequential`, `get_params`, `param_count`, `perceptron_fit/predict`, `mlp_init`, `mlp_train`, `mlp_predict` |
| `Optimdoz` | `sgd`, `adam`, `zeros_like` |
| `Datadoz` | `from_lists`, `make_loader`, `get_batches`, `train_test_split`, `read_csv`, `read_xlsx`, `columna`, `fila`, `num_filas`, `num_columnas`, `columnas` |
| `Plotdoz` | `learning_curve`, `multi_curve`, `histogram`, `bar_chart`, `scatter`, `scatter_classes`, `line_chart`, `confusion_matrix`, `heatmap`, `cluster_scatter`, `learning_curve_ascii` |
| `IOdoz` | `leer`, `escribir` |
| `Inspectdoz` | `spy`, `shape` |
| `Structsdoz` | `Queue`, `Stack` |

---

## Estructura del Proyecto

```
TREZ/
├── README.md
├── src/
│   ├── main.py                  # Punto de entrada del intérprete
│   ├── visitor.py               # Patrón Visitor — evaluación del AST + _NAMESPACES
│   ├── math_utilsdoz.py         # Re-exporta la stdlib al visitor
│   ├── autograd.py              # Grafo computacional + backprop nativo
│   ├── errors.py                # TrezError, TrezSyntaxError, TrezRuntimeError
│   ├── error_listener.py        # TrezErrorListener para ANTLR4
│   ├── parser/
│   │   ├── TrezLexer.g4         # Gramática léxica
│   │   ├── TrezParser.g4        # Gramática sintáctica (CFG Tipo 2)
│   │   ├── TrezLexer.py         # Generado por ANTLR4 — no editar
│   │   ├── TrezParser.py        # Generado por ANTLR4 — no editar
│   │   └── TrezParserVisitor.py # Generado por ANTLR4 — no editar
│   └── lib/
│       ├── mathdoz/             # core_mathdoz.py + tensor_mathdoz.py
│       ├── activationsdoz/      # relu, sigmoid, tanh, softmax, linear
│       ├── lossesdoz/           # mse, mse_grad, cross_entropy
│       ├── optimdoz/            # sgd, adam, zeros_like
│       ├── randomdoz/           # Xorshift64: seed, random, uniform, gauss, shuffle...
│       ├── metricsdoz/          # accuracy, precision, recall, f1, confusion_matrix, rmse, r2...
│       ├── mldoz/               # linreg, logreg, svm, tree, knn, kmeans
│       ├── nndoz/               # capas + perceptron + mlp entrenable
│       ├── datadoz/             # from_lists, loader, read_csv, read_xlsx, columna, fila
│       ├── plotdoz/             # SVG nativo: scatter, bar, histogram, learning_curve...
│       ├── iodoz/               # leer, escribir
│       ├── inspectdoz/          # spy, shape
│       └── structsdoz/          # Queue, Stack
├── tests/
│   ├── features/                # 18 tests de lenguaje y stdlib base
│   ├── ml/                      # 9 tests de ML/DL: svm, tree, knn, kmeans, perceptron, mlp...
│   └── run_tests.py             # Runner: python3 tests/run_tests.py
└── docs/
    └── diseno/
        ├── Design.md            # Arquitectura detallada
        └── PLAN_DESARROLLO.md   # Roadmap y API completa
```

---

## Estado de Entregas

| Entrega | Objetivo | Estado |
|---|---|---|
| 1 | Aritmética, arrays, variables, funciones nativas, errores | ✅ Completa |
| 2 | Pipe `\|>`, lambdas, namespaces, dicts, Queue, Stack, closures, for/while | ✅ Completa |
| 3 | Tensordoz completo, Optimdoz, Metricsdoz, Datadoz CSV/XLSX, Plotdoz | ✅ Completa |
| **T3 / Final** | ML clásico (linreg, logreg, SVM, árbol, k-NN, k-Means), MLP entrenable, Perceptrón, Randomdoz, Plotdoz SVG nativo, suite ml/ | ✅ **Completa** |

**Suite de tests: 27/27 pasan.** Sin imports de `matplotlib`, `numpy`, `random`, `math` ni ninguna otra librería externa en `src/`.

---

Julián David Cristancho Bustos — Universidad Sergio Arboleda
