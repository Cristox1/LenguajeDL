# Arquitectura y Diseño de TREZ

**TREZ** — Lenguaje de Dominio Específico Funcional para Deep Learning
Julián David Cristancho Bustos — Universidad Sergio Arboleda

---

## 1. Visión General

TREZ es un DSL estrictamente funcional orientado a la definición, entrenamiento y evaluación de modelos de Deep Learning, construido desde cero sin dependencias externas para el cómputo numérico. Cada expresión es una función; no existen variables mutables ni bucles imperativos en el núcleo del lenguaje. El entrenamiento de una red neuronal se expresa como una cadena de transformaciones funcionales puras encadenadas con `|>`.

**Principio de diseño central:** En TREZ, `A |> f` equivale a `f(A)`. Un pipeline de inferencia completo es una composición declarativa de funciones sin efectos secundarios.

---

## 2. Arquitectura del Sistema

El compilador e intérprete de TREZ opera en dos niveles:

```
Script .trez
    │
    ▼
 [Lexer]  ← TrezLexer.g4  (ANTLR4)
    │
    ▼
 [Parser] ← TrezParser.g4 (ANTLR4, CFG Tipo 2)
    │
    ▼
   AST (Parse Tree)
    │
    ▼
 [Visitor] ← TrezVisitor (Python, patrón Visitor)
    │         Entorno: Environment (scopes encadenados)
    ▼
 [Stdlib nativa] ← mathdoz, tensordoz, metricsdoz, iodoz, structsdoz...
    │
    ▼
 Resultado
```

### 2.1 Front-end — Análisis Léxico y Sintáctico (ANTLR4)

- **Lexer (`TrezLexer.g4`):** Segmenta el flujo de caracteres en tokens: `ID`, `NUMBER`, `STRING`, keywords (`let`, `func`, `return`, `if`, `else`, `while`, `for`, `in`, `not`), operadores (`**`, `&&`, `||`, `|>`, `->`, `==`, `!=`, `<=`, `>=`), separadores y literales.
- **Parser (`TrezParser.g4`):** Aplica la gramática CFG Tipo 2 (Chomsky) para construir el Parse Tree. ANTLR4 genera automáticamente `TrezParser.py` y `TrezParserVisitor.py`.

### 2.2 Back-end — Motor de Evaluación (Python)

El `TrezVisitor` extiende `TrezParserVisitor` (generado por ANTLR4) e implementa un `visitX()` para cada regla de la gramática. El entorno de ejecución (`Environment`) es una cadena de scopes enlazados que implementa lookup funcional: cada bloque/función crea un nuevo scope hijo, y `let` actualiza el binding más cercano en la cadena via `update()`.

---

## 3. Decisiones de Diseño

### 3.1 Paradigma Funcional

- Bindings con `let`: inmutables en intención; `update()` sube la cadena de scopes para simular reasignación controlada dentro de loops.
- Funciones como closures (`TrezFunction`): capturan el `Environment` de definición, permiten recursión (self-reference inyectado en el call env).
- Lambdas anónimas (`\x -> expr`): funciones de primer orden pasables como argumentos y al operador `|>`.
- Sin side-effects en la stdlib: todas las funciones de `lib/` retornan nuevos valores, nunca mutan en lugar.

### 3.2 Patrón Visitor

Cada nodo del AST tiene un `visitNodo()` correspondiente en `TrezVisitor`. La evaluación es puramente recursiva: visitar un nodo dispara la visita de sus hijos, compone los resultados y retorna un valor inmutable. No hay tablas de símbolos globales mutables — el estado vive en el `Environment` encadenado.

### 3.3 Cero Librerías Externas (para cómputo)

| Necesidad | Implementación nativa |
|---|---|
| Constantes PI, E | Literales de 20 dígitos en `core_mathdoz.py` |
| `exp(x)` | Serie de Maclaurin (30 términos) |
| `log(x)` | Serie de Mercator + reducción de rango por E |
| `sin(x)`, `cos(x)` | Serie de Taylor (20 términos) + reducción de periodo |
| `sqrt(x)` | Método de Newton-Raphson (20 iteraciones) |
| `pow(base, exp)` | Producto iterativo (enteros) / `exp(n·log(base))` (floats) |
| Producto matricial | Triple bucle nativo en `tensor_mathdoz.py` |
| Transpose | Lista de comprensión 2D |
| ReLU, Sigmoid | Definiciones elementales sobre `exp_doz` |
| MSE, MSE_grad | Definiciones analíticas directas |
| Queue, Stack | Listas Python puras con semántica inmutable en `structsdoz.py` |
| Backprop | Grafo computacional con topological sort en `autograd.py` |

La única dependencia externa real es `antlr4-python3-runtime` — el motor del parser/lexer, estructural e inevitable.

### 3.4 Operador Pipe `|>`

`A |> f` evalúa `f(A)`. `A |> f |> g` evalúa `g(f(A))`. Tiene la menor precedencia de todos los operadores, garantizando que `a + b |> f` se evalúa como `f(a + b)`.

Con lambdas: `datos |> \x -> Tensordoz.dot(W, x) |> Mathdoz.relu` es un forward pass de una capa.

### 3.5 Namespaces de Módulo

La stdlib se invoca como `Modulo.funcion(args)` para hacer explícito el origen de cada operación y evitar colisiones de nombres. El visitor resuelve `Tensordoz.dot(A, B)` despachando a `tensor_mathdoz.dot(A, B)`.

---

## 4. Gramática (CFG Tipo 2 — ANTLR4)

### 4.1 Estado Actual — Entregas 1 y 2 (Completas)

```antlr
// TrezLexer.g4
LET: 'let';  FUNC: 'func';  RETURN: 'return';
IF: 'if';    ELSE: 'else';  WHILE: 'while';
FOR: 'for';  IN: 'in';      NOT: 'not';
POW: '**';   AND: '&&';     OR: '||';
PIPE: '|>';  ARROW: '->';   BACKSLASH: '\\';
EQEQ: '==';  NEQ: '!=';    LE: '<=';  GE: '>=';
DOT: '.';    COLON: ':';   MOD: '%';
// ... NUMBER, STRING, ID, WS, LINE_COMMENT

// TrezParser.g4
program    : statement+ EOF;
statement  : let_stmt | bind_tuple | func_def | return_stmt
           | expr_stmt | if_stmt | while_stmt | for_stmt | block;

let_stmt   : LET ID EQ expr SEMI;
bind_tuple : LET LBRACK ID (COMMA ID)* RBRACK EQ expr SEMI;
func_def   : FUNC ID LPAREN param_list? RPAREN block;
return_stmt: RETURN expr SEMI;
for_stmt   : FOR ID IN expr block;
if_stmt    : IF LPAREN expr RPAREN block (ELSE (if_stmt | block))?;

expr (menor a mayor precedencia):
    lambdaDef   : BACKSLASH ID ARROW expr
    pipeOp      : expr PIPE expr
    orExpr      : expr OR expr
    andExpr     : expr AND expr
    notExpr     : NOT expr
    eqExpr      : expr (EQEQ | NEQ) expr
    compareExpr : expr (LT | LE | GT | GE) expr
    mulDivExpr  : expr (MUL | DIV | MOD) expr
    addSubExpr  : expr (PLUS | MINUS) expr
    powExpr     : expr POW expr
    unaryMinus  : MINUS expr
    indexExpr   : expr LBRACK expr RBRACK
    moduleCall  : ID DOT ID LPAREN args? RPAREN
    methodCall  : expr DOT ID LPAREN args? RPAREN
    funcCall    : ID LPAREN args? RPAREN
    // ... literals, arrays, dicts
```

### 4.2 Precedencia de Operadores (mayor prioridad abajo)

| Nivel | Operador | Asociatividad |
|---|---|---|
| 1 (menor) | `\x ->` lambda | derecha |
| 2 | `\|>` pipe | izquierda |
| 3 | `\|\|` | izquierda |
| 4 | `&&` | izquierda |
| 5 | `not` | prefijo |
| 6 | `==`, `!=` | izquierda |
| 7 | `<`, `<=`, `>`, `>=` | izquierda |
| 8 | `+`, `-` | izquierda |
| 9 | `*`, `/`, `%` | izquierda |
| 10 | `**` | derecha |
| 11 | `-` unario | prefijo |
| 12 (mayor) | `[]`, `.método()` | izquierda |

---

## 5. Librería Estándar Nativa

### 5.1 Mathdoz / `core_mathdoz.py`

Constantes: `PI`, `E`.
Funciones: `abs`, `pow`, `factorial`, `sqrt` (Newton-Raphson), `exp` (Maclaurin), `log` (Mercator), `sin`, `cos`, `tan` (Taylor). Ninguna usa `import math`.

### 5.2 Tensordoz / `tensor_mathdoz.py`

`dot`, `transpose`, `add`, `sub`, `scale`, `zeros`, `ones`, `flatten`, `reshape`, `concat`. Operaciones matriciales en Python puro (con fallback opcional a Cython si el `.so` existe).

### 5.3 Activationsdoz / `activationsdoz.py`

`relu`, `sigmoid`, `tanh`, `softmax` (numéricamente estable), `linear` — todas elemento a elemento. Derivadas: `relu_deriv`, `sigmoid_deriv`, `tanh_deriv`.

### 5.4 Randomdoz / `randomdoz/randomdoz.py`

PRNG nativo Xorshift64 (Marsaglia, 2003). API: `seed`, `random`, `uniform`, `randint`, `choice`, `sample`, `shuffle`, `gauss` (Box-Muller usando `core_mathdoz`). Reemplaza `import random` en todo el proyecto.

### 5.5 Metricsdoz / `metricsdoz/metricsdoz.py`

Clasificación: `accuracy`, `precision`, `recall`, `f1_score`, `confusion_matrix` (K×K).
Regresión: `mse`, `rmse`, `mae`, `r2_score`.
Losses DL (re-exportados de `lossesdoz`): `cross_entropy`, `cross_entropy_grad`.

### 5.6 Mldoz / `mldoz/`

ML clásico, todo Python puro:

| Archivo | Algoritmo | API |
|---|---|---|
| `linear_regression.py` | Regresión lineal por SGD | `linreg_fit`, `linreg_predict` |
| `logistic_regression.py` | Regresión logística binaria | `logreg_fit`, `logreg_predict`, `logreg_predict_proba` |
| `svm.py` | SVM lineal (subgradiente hinge loss) | `svm_fit`, `svm_predict` |
| `tree.py` | Árbol de decisión (Gini / MSE) | `tree_clf_fit`, `tree_reg_fit`, `tree_predict` |
| `knn.py` | k-Vecinos más cercanos | `knn_fit`, `knn_predict`, `knn_predict_clf`, `knn_predict_reg` |
| `kmeans.py` | k-Means (init Randomdoz) | `kmeans_fit`, `kmeans_predict` |

### 5.7 NNdoz / `nndoz/nndoz.py`

Capas: `linear_init`, `linear_forward`, `linear_backward`, `relu_forward`, `relu_backward`, `softmax_forward`, `sequential_forward`.
Perceptrón de Rosenblatt: `perceptron_fit`, `perceptron_predict`.
MLP entrenable completo: `mlp_init(arch)`, `mlp_train(layers, X, y, lr, epochs, loss)`, `mlp_predict(layers, X)`.

El bucle de `mlp_train` realiza: forward → loss → grad inicial → backward capa a capa → SGD. Soporta `loss='mse'` y `loss='cross_entropy'`.

### 5.8 Optimdoz / `optimdoz/optimdoz.py`

`sgd(params, grads, lr, momentum)` — descenso por gradiente con momentum opcional.
`adam(params, grads, m, v, t, lr, beta1, beta2, eps)` — Adam con corrección de sesgo.
`zeros_like(params)` — inicializa acumuladores.

### 5.9 Datadoz / `datadoz/datadoz.py`

`from_lists`, `make_loader`, `get_batches`, `train_test_split`, `read_csv`, `read_xlsx`, `columna`, `fila`, `num_filas`, `num_columnas`, `columnas`. El barajado usa `Randomdoz.shuffle`.

### 5.10 Plotdoz / `plotdoz/plotdoz.py`

**Implementación: SVG nativo + ASCII. Cero matplotlib.**

Genera archivos `.svg` (texto plano, abre en navegador) usando constructores de cadenas. Si el caller pasa `.png`, la extensión se convierte automáticamente a `.svg`.

Funciones disponibles:

| Función | Descripción |
|---|---|
| `learning_curve(train, val, título, ...)` | Polyline azul (train) + roja (val) con grid |
| `multi_curve(series, labels, ...)` | Múltiples series en un eje |
| `histogram(values, bins, ...)` | Bucketing manual, `rect` por barra |
| `bar_chart(labels, values, ...)` | Barras con etiquetas |
| `scatter(x, y, ...)` | Puntos escalados al canvas |
| `scatter_classes(x, y, ids, ...)` | Puntos coloreados por clase |
| `line_chart(x, y, ...)` | Polyline escalada |
| `confusion_matrix(y_true, y_pred, ...)` | Heatmap K×K con conteos |
| `heatmap(matrix, ...)` | Heatmap genérico |
| `cluster_scatter(X, asigns, centroides, ...)` | Scatter por cluster + marcador centroide |
| `learning_curve_ascii(losses, título)` | Curva en terminal 60×20 con `█` |

### 5.11 IOdoz / `iodoz/iodoz.py`

`leer(ruta)` → string, `escribir(ruta, contenido)` → void.

### 5.12 Inspectdoz / `inspectdoz/inspectdoz.py`

`spy(tensor)` — imprime y retorna. `shape(tensor)` — dimensiones.

---

## 6. Sistema de Errores

```
TrezError (base)
├── TrezSyntaxError      — detectado por TrezErrorListener en ANTLR4
│                          incluye línea, columna y token inesperado
├── TrezRuntimeError     — detectado por TrezVisitor en evaluación
│                          cubre: variable no definida, dimensión incompatible,
│                          división por cero, índice fuera de rango, IO
├── UndefinedSymbolError — (Entrega 2) reemplaza el caso "variable no definida"
│                          con información del símbolo y el entorno actual
├── ShapeMismatchError   — (Entrega 3) para incompatibilidades dimensionales en dot/add
└── MathDomainError      — (Entrega 3) para log(0), sqrt(-1)
```

---

## 7. Hoja de Ruta

### Entrega 1 — ✅ Completa

- [x] Gramática ANTLR4: aritmética, arrays 1D/2D/3D, `let`, funciones nativas globales
- [x] Visitor: `TrezVisitor` + entorno, aritmética vectorial, jerarquía de errores
- [x] Stdlib: `Mathdoz`, `Tensordoz` (dot+transpose), `Metricsdoz` (mse+mse_grad), `IOdoz`
- [x] `autograd.py`: grafo computacional + backprop topológico

### Entrega 2 — ✅ Completa

- [x] Funciones con nombre (`func`/`return`), recursión, closures
- [x] Condicionales encadenados (`else if`), `for..in`, `not`, `while`
- [x] Diccionarios, acceso `[]`, métodos `.metodo()`
- [x] `Queue` y `Stack` nativos, `range()`, `len()`, `append()`, `head()`, `tail()`, `str()`, `num()`
- [x] Pipe `|>`, lambdas `\x -> expr`, namespaces `Modulo.func()`, desestructuración `let [a,b] = expr`
- [x] `Inspectdoz`: `spy()`, `shape()`

### Entrega 3 — ✅ Completa

- [x] `Tensordoz`: `reshape`, `flatten`, `add`, `sub`, `scale`, `concat`, `zeros`, `ones`
- [x] `Activationsdoz`: `softmax`, `tanh`, `linear` y derivadas
- [x] `Optimdoz`: `sgd`, `adam`, `zeros_like`
- [x] `Metricsdoz` completo: `accuracy`, `precision`, `recall`, `f1_score`, `confusion_matrix`, `rmse`, `mae`, `r2_score`
- [x] `Datadoz`: `read_csv`, `read_xlsx`, `train_test_split`

### T3 / Entrega Final — ✅ Completa

- [x] `Randomdoz`: PRNG Xorshift64 nativo (`seed`, `random`, `uniform`, `gauss`, `shuffle`, `sample`…) — reemplaza `import random`
- [x] `Mldoz` completo: regresión lineal y logística, SVM lineal, árbol (clasif + regr), k-NN, k-Means
- [x] `NNdoz` extendido: Perceptrón de Rosenblatt + MLP entrenable (`mlp_init`, `mlp_train`, `mlp_predict`)
- [x] `Plotdoz` reescrito en SVG nativo — cero `matplotlib`; nuevas funciones: `cluster_scatter`, `learning_curve_ascii`
- [x] Suite `tests/ml/` (9 tests): svm, tree, knn, kmeans, perceptron, mlp_train, metrics, linreg, logreg
- [x] `tests/run_tests.py` extendido para cubrir `tests/features/` y `tests/ml/` — **27/27 pasan**
- [x] Verificación dura: `grep -rn "^import random|^import math|matplotlib" src/` → sin resultados

---

## 8. Estructura del Proyecto

```
TREZ/
├── README.md
├── docs/diseno/
│   ├── Design.md                    (este archivo)
│   └── PLAN_DESARROLLO.md
├── src/
│   ├── main.py                      punto de entrada
│   ├── visitor.py                   TrezVisitor + _NAMESPACES (dispatch Modulo.func())
│   ├── math_utilsdoz.py             re-exporta stdlib al visitor
│   ├── autograd.py                  Tensor + backward topológico
│   ├── errors.py                    jerarquía de errores
│   ├── error_listener.py            TrezErrorListener
│   ├── parser/
│   │   ├── TrezLexer.g4             gramática léxica
│   │   ├── TrezParser.g4            gramática sintáctica CFG Tipo 2
│   │   ├── TrezLexer.py             generado ANTLR4 — no editar
│   │   ├── TrezParser.py            generado ANTLR4 — no editar
│   │   └── TrezParserVisitor.py     generado ANTLR4 — no editar
│   └── lib/
│       ├── mathdoz/
│       │   ├── core_mathdoz.py      PI, E, exp, log, sin, cos, tan, sqrt, pow, factorial
│       │   └── tensor_mathdoz.py    dot, transpose, add, sub, scale, flatten, reshape, concat...
│       ├── activationsdoz/
│       │   └── activationsdoz.py    relu, sigmoid, tanh, softmax, linear + derivadas
│       ├── randomdoz/
│       │   └── randomdoz.py         Xorshift64: seed, random, uniform, randint, gauss, shuffle...
│       ├── lossesdoz/
│       │   └── lossdoz.py           mse, mse_grad, cross_entropy, cross_entropy_grad
│       ├── metricsdoz/
│       │   └── metricsdoz.py        accuracy, precision, recall, f1, confusion_matrix, rmse, r2...
│       ├── mldoz/
│       │   ├── linear_regression.py linreg_fit, linreg_predict
│       │   ├── logistic_regression.py logreg_fit, logreg_predict, logreg_predict_proba
│       │   ├── svm.py               svm_fit, svm_predict
│       │   ├── tree.py              tree_clf_fit, tree_reg_fit, tree_predict
│       │   ├── knn.py               knn_fit, knn_predict, knn_predict_clf, knn_predict_reg
│       │   └── kmeans.py            kmeans_fit, kmeans_predict
│       ├── optimdoz/
│       │   └── optimdoz.py          sgd, adam, zeros_like
│       ├── nndoz/
│       │   └── nndoz.py             capas linear/relu/softmax + perceptron + mlp_init/train/predict
│       ├── datadoz/
│       │   └── datadoz.py           from_lists, loader, read_csv, read_xlsx, columna, fila...
│       ├── plotdoz/
│       │   └── plotdoz.py           SVG nativo: scatter, bar, histogram, learning_curve...
│       ├── iodoz/
│       │   └── iodoz.py             leer, escribir
│       ├── inspectdoz/
│       │   └── inspectdoz.py        spy, shape
│       └── structsdoz/
│           └── structsdoz.py        TrezQueue, TrezStack
└── tests/
    ├── features/                    18 tests: lenguaje, stdlib base, DL básico
    ├── ml/                          9 tests: linreg, logreg, svm, tree, knn, kmeans,
    │                                         perceptron, mlp_train, metrics
    └── run_tests.py                 runner unificado — 27/27 pasan
```

---

## 9. Semántica Operacional (reglas clave)

```
-- Binding
Γ ⊢ e ⇒ v
─────────────────────────────
Γ ⊢ let x = e  ⇒  Γ[x ↦ v]

-- Aplicación de función
Γ ⊢ f ⇒ TrezFunction(params, body, env_def)
Γ ⊢ args ⇒ [v1..vn]
env_call = env_def + {param_i ↦ vi} + {f ↦ f}  (self-ref para recursión)
env_call ⊢ body ⇒ v
─────────────────────────────────────────────────
Γ ⊢ f(args) ⇒ v

-- Pipe
Γ ⊢ e1 ⇒ v1
Γ ⊢ e2 ⇒ TrezFunction / builtin f
─────────────────────────────────
Γ ⊢ e1 |> e2  ⇒  f(v1)

-- Lambda
──────────────────────────────────────────
Γ ⊢ \x -> body  ⇒  TrezLambda(x, body, Γ)

-- For
Γ ⊢ iterable ⇒ [v1..vn]
∀ vi: Γ + {var ↦ vi} ⊢ block
─────────────────────────────
Γ ⊢ for var in iterable block  ⇒  ()
```
