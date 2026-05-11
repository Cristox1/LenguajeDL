from PELEVisitor import PELEVisitor
from PELEParser import PELEParser
import pele_numpy_runtime as pnp

class ReturnValue(Exception):
    """Excepción interna usada para implementar 'retornar' en funciones."""
    def __init__(self, value):
        self.value = value

class EvalVisitor(PELEVisitor):
    def __init__(self):
        # pila de scopes (cada scope es un dict). scopes[0] = global
        self.scopes = [{}]
        # almacenamiento de funciones definidas por el usuario: name -> {'params': [...], 'block': block_ctx}
        self.functions = {}
        # control de detención en primer error
        self.stop_on_error = False

    # ----- Helpers de scope -----
    def push_scope(self):
        self.scopes.append({})

    def pop_scope(self):
        if len(self.scopes) > 1:
            self.scopes.pop()
        else:
            # no quitar el scope global
            self.scopes[0].clear()

    def current_scope(self):
        return self.scopes[-1]

    def set_var(self, name, value):
        # asigna en el scope actual (local)
        self.current_scope()[name] = value

    def get_var(self, name):
        # busca desde el scope actual hacia afuera
        for s in reversed(self.scopes):
            if name in s:
                return s[name]
        raise Exception(f"Error: Variable '{name}' no definida.")

    # Programa y bloques
    def visitProgram(self, ctx: PELEParser.ProgramContext):
        return self.visit(ctx.block())

    def visitBlock(self, ctx: PELEParser.BlockContext):
        """
        Ejecuta cada statement del bloque. Captura excepciones y permite continuar
        a menos que stop_on_error sea True. Re-lanza ReturnValue para permitir 'retornar'.
        """
        for stmt in ctx.statement():
            try:
                self.visit(stmt)
            except ReturnValue:
                # re-lanzar para que el flujo de retorno funcione (se captura en llamada de función)
                raise
            except Exception as e:
                # intentar obtener linea para mejor mensaje
                line_no = '?'
                try:
                    if hasattr(stmt, 'start') and stmt.start is not None:
                        line_no = stmt.start.line
                except Exception:
                    pass
                print(f"[Linea {line_no}] Error en statement: {e}")
                if self.stop_on_error:
                    raise
        return None

    # Asignación
    def visitAssignStmt(self, ctx: PELEParser.AssignStmtContext):
        assign_ctx = ctx.assignment()
        var_name = assign_ctx.ID().getText()
        value = self.visit(assign_ctx.expr())
        self.set_var(var_name, value)
        return value

    # Mostrar
    def _format_value(self, v):
        if isinstance(v, dict):
            if 'value' in v and 'children' in v:
                return f"Arbol({self._format_value(v['value'])}, children={len(v['children'])})"
            else:
                items = ", ".join(f"{repr(k)}: {self._format_value(val)}" for k, val in v.items())
                return "{" + items + "}"
        if isinstance(v, list):
            inner = ", ".join(self._format_value(x) for x in v)
            return "[" + inner + "]"
        if isinstance(v, set):
            try:
                inner = ", ".join(self._format_value(x) for x in sorted(v, key=repr))
            except Exception:
                inner = ", ".join(self._format_value(x) for x in v)
            return "{" + inner + "}"
        return repr(v)

    def visitMostrarStmt(self, ctx: PELEParser.MostrarStmtContext):
        expr_ctx = ctx.expr()
        value = self.visit(expr_ctx)
        print("> " + self._format_value(value))
        return None

    def visitExprStmt(self, ctx: PELEParser.ExprStmtContext):
        return self.visit(ctx.expr())

    # Expresiones
    def visitUnaryMinusExpr(self, ctx: PELEParser.UnaryMinusExprContext):
        return -self.visit(ctx.expr())

    def visitPowerExpr(self, ctx: PELEParser.PowerExprContext):
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        return left ** right

    def visitMulDivModExpr(self, ctx: PELEParser.MulDivModExprContext):
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        op = ctx.getChild(1).getText()
        if op == '*': return left * right
        if op == '/': return left / right
        if op == '%': return left % right

    def visitAddSubExpr(self, ctx: PELEParser.AddSubExprContext):
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        op = ctx.getChild(1).getText()
        if op == '+': return left + right
        if op == '-': return left - right

    def visitRelationalExpr(self, ctx: PELEParser.RelationalExprContext):
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        op = ctx.getChild(1).getText()
        if op == '<':  return left < right
        if op == '<=': return left <= right
        if op == '>':  return left > right
        if op == '>=': return left >= right
        if op == '==': return left == right
        if op == '!=': return left != right

    def visitArrayExpr(self, ctx: PELEParser.ArrayExprContext):
        exprs = list(ctx.expr()) if ctx.expr() else []
        return [self.visit(expr) for expr in exprs]

    def visitBoolExpr(self, ctx: PELEParser.BoolExprContext):
        txt = ctx.getText()
        return txt == 'true'

    def visitStringExpr(self, ctx: PELEParser.StringExprContext):
        text = ctx.getText()
        return text[1:-1]

    def visitIntExpr(self, ctx: PELEParser.IntExprContext):
        return int(ctx.getText())

    def visitFloatExpr(self, ctx: PELEParser.FloatExprContext):
        return float(ctx.getText())

    def visitIdExpr(self, ctx: PELEParser.IdExprContext):
        var_name = ctx.getText()
        return self.get_var(var_name)

    def visitParensExpr(self, ctx: PELEParser.ParensExprContext):
        return self.visit(ctx.expr())

    # If
    def visitIfStmt(self, ctx: PELEParser.IfStmtContext):
        if_ctx = ctx.ifStatement()
        condition = self.visit(if_ctx.expr())
        if condition:
            return self.visit(if_ctx.block(0))
        else:
            blocks = list(if_ctx.block())
            if len(blocks) > 1 and blocks[1] is not None:
                return self.visit(blocks[1])
        return None

    # Return
    def visitReturnStmt(self, ctx: PELEParser.ReturnStmtContext):
        value = self.visit(ctx.expr())
        # lanzar ReturnValue para salir de la función
        raise ReturnValue(value)

    # Definición de funciones (no se ejecuta al definir, sólo registra)
    def visitFunctionDeclStmt(self, ctx: PELEParser.FunctionDeclStmtContext):
        func_ctx = ctx.functionDecl()
        name = func_ctx.ID().getText()
        params = []
        if func_ctx.params():
            params = [p.getText() for p in func_ctx.params().ID()]
        block = func_ctx.block()
        self.functions[name] = {'params': params, 'block': block}
        return None

    # Llamadas a funciones / builtins
    def visitFuncCallExpr(self, ctx: PELEParser.FuncCallExprContext):
        func_name = ctx.ID().getText()
        args = [self.visit(e) for e in ctx.expr()] if ctx.expr() else []

        # builtins first
        built = self.builtins()
        if func_name in built:
            func = built[func_name]
            try:
                return func(*args)
            except TypeError as e:
                line = ctx.start.line if hasattr(ctx, 'start') else '?'
                raise Exception(f"[Linea {line}] Error llamando a builtin '{func_name}': {e}")
            except Exception as e:
                line = ctx.start.line if hasattr(ctx, 'start') else '?'
                raise Exception(f"[Linea {line}] {e}")

        # user-defined functions
        if func_name not in self.functions:
            line = ctx.start.line if hasattr(ctx, 'start') else '?'
            raise Exception(f"[Linea {line}] Funcion '{func_name}' no definida.")

        func_info = self.functions[func_name]
        param_names = func_info['params']
        if len(args) != len(param_names):
            raise Exception(f"Funcion '{func_name}' espera {len(param_names)} argumentos, recibió {len(args)}.")

        # preparar scope local
        self.push_scope()
        try:
            # asignar parámetros en el scope local
            for pname, aval in zip(param_names, args):
                self.set_var(pname, aval)
            # ejecutar el bloque de la función y capturar ReturnValue
            try:
                self.visit(func_info['block'])
                # si no hubo 'retornar', devolvemos None
                return None
            except ReturnValue as r:
                return r.value
        finally:
            # restaurar scope (pop)
            self.pop_scope()

    # ----- Builtins (igual que antes, sin librerías externas) -----
    def builtins(self):
        return {
            # Mapas
            "crear_mapa": self._crear_mapa,
            "mapa_put": self._mapa_put,
            "mapa_get": self._mapa_get,
            "mapa_keys": self._mapa_keys,
            "mapa_values": self._mapa_values,
            # Pilas y colas
            "crear_pila": self._crear_pila,
            "pila_push": self._pila_push,
            "pila_pop": self._pila_pop,
            "crear_cola": self._crear_cola,
            "cola_enqueue": self._cola_enqueue,
            "cola_dequeue": self._cola_dequeue,
            # Conjuntos
            "crear_conjunto": self._crear_conjunto,
            "conjunto_add": self._conjunto_add,
            "conjunto_contains": self._conjunto_contains,
            # Matrices
            "crear_matriz": self._crear_matriz,
            "mat_get": self._mat_get,
            "mat_set": self._mat_set,
            # Arboles
            "crear_arbol": self._crear_arbol,
            "arbol_add_child": self._arbol_add_child,
            "arbol_preorder": self._arbol_preorder,
            # Grafos
            "crear_grafo": self._crear_grafo,
            "grafo_add_node": self._grafo_add_node,
            "grafo_add_edge": self._grafo_add_edge,
            "grafo_neighbors": self._grafo_neighbors,
            "grafo_bfs": self._grafo_bfs,
            # Helpers para arreglos (opcional)
            "arr_get": self._arr_get,
            "arr_set": self._arr_set,
            # Mini-Numpy
            "np_array": pnp.np_array,
            "np_shape": pnp.np_shape,
            "np_zeros": pnp.np_zeros,
            "np_ones": pnp.np_ones,
            "np_add": pnp.np_add,
            "np_sub": pnp.np_sub,
            "np_mul": pnp.np_mul,
            "np_div": pnp.np_div,
            "np_scalar_mul": pnp.np_scalar_mul,
            "np_transpose": pnp.np_transpose,
            "np_matmul": pnp.np_matmul,
            "np_sum": pnp.np_sum,
            "np_argmax": pnp.np_argmax,
        }

    # Implementaciones de builtins (idénticas a las previas)
    # Mapas
    def _crear_mapa(self, pairs):
        m = {}
        if pairs is None:
            return m
        for pair in pairs:
            if isinstance(pair, list) and len(pair) == 2:
                m[pair[0]] = pair[1]
            else:
                raise Exception("crear_mapa espera un arreglo de pares [clave,valor]")
        return m

    def _mapa_put(self, m, k, v):
        if not isinstance(m, dict):
            raise Exception("mapa_put: primer argumento debe ser un mapa")
        m[k] = v
        return m

    def _mapa_get(self, m, k):
        if not isinstance(m, dict):
            raise Exception("mapa_get: primer argumento debe ser un mapa")
        return m.get(k, None)

    def _mapa_keys(self, m):
        if not isinstance(m, dict):
            raise Exception("mapa_keys: argumento debe ser un mapa")
        return list(m.keys())

    def _mapa_values(self, m):
        if not isinstance(m, dict):
            raise Exception("mapa_values: argumento debe ser un mapa")
        return list(m.values())

    # Pilas
    def _crear_pila(self):
        return []

    def _pila_push(self, stack, value):
        if not isinstance(stack, list):
            raise Exception("pila_push: primer argumento debe ser una pila")
        stack.append(value)
        return None

    def _pila_pop(self, stack):
        if not isinstance(stack, list):
            raise Exception("pila_pop: primer argumento debe ser una pila")
        if not stack:
            raise Exception("Error: pila vacia")
        return stack.pop()

    # Colas
    def _crear_cola(self):
        return []

    def _cola_enqueue(self, queue, value):
        if not isinstance(queue, list):
            raise Exception("cola_enqueue: primer argumento debe ser una cola")
        queue.append(value)
        return None

    def _cola_dequeue(self, queue):
        if not isinstance(queue, list):
            raise Exception("cola_dequeue: primer argumento debe ser una cola")
        if not queue:
            raise Exception("Error: cola vacia")
        return queue.pop(0)

    # Conjuntos
    def _crear_conjunto(self, arr):
        if arr is None:
            return set()
        return set(arr)

    def _conjunto_add(self, s, v):
        if not isinstance(s, set):
            raise Exception("conjunto_add: primer argumento debe ser un conjunto")
        s.add(v)
        return None

    def _conjunto_contains(self, s, v):
        if not isinstance(s, set):
            raise Exception("conjunto_contains: primer argumento debe ser un conjunto")
        return v in s

    # Matrices
    def _crear_matriz(self, rows, cols, fill):
        rows_i = int(rows); cols_i = int(cols)
        return [[fill for _ in range(cols_i)] for _ in range(rows_i)]

    def _mat_get(self, mat, i, j):
        return mat[int(i)][int(j)]

    def _mat_set(self, mat, i, j, val):
        mat[int(i)][int(j)] = val
        return None

    # Arboles
    def _crear_arbol(self, value):
        return {'value': value, 'children': []}

    def _arbol_add_child(self, node, child):
        if not isinstance(node, dict) or 'children' not in node:
            raise Exception("arbol_add_child: primer argumento no es un nodo de arbol")
        node['children'].append(child)
        return child

    def _arbol_preorder(self, node):
        if not isinstance(node, dict) or 'children' not in node:
            raise Exception("arbol_preorder: argumento no es un nodo de arbol")
        res = []
        def _rec(n):
            res.append(n['value'])
            for c in n['children']:
                _rec(c)
        _rec(node)
        return res

    # Grafos
    def _crear_grafo(self):
        return {}

    def _grafo_add_node(self, g, node):
        if not isinstance(g, dict):
            raise Exception("grafo_add_node: primer argumento debe ser un grafo")
        if node not in g:
            g[node] = []
        return None

    def _grafo_add_edge(self, g, u, v):
        if not isinstance(g, dict):
            raise Exception("grafo_add_edge: primer argumento debe ser un grafo")
        if u not in g: g[u] = []
        if v not in g: g[v] = []
        g[u].append(v)
        return None

    def _grafo_neighbors(self, g, node):
        if not isinstance(g, dict):
            raise Exception("grafo_neighbors: primer argumento debe ser un grafo")
        return g.get(node, [])

    def _grafo_bfs(self, g, start):
        if not isinstance(g, dict):
            raise Exception("grafo_bfs: primer argumento debe ser un grafo")
        visited = set()
        queue = []
        order = []
        queue.append(start); visited.add(start)
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in g.get(u, []):
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        return order

    # Helpers arrays
    def _arr_get(self, arr, idx):
        return arr[int(idx)]

    def _arr_set(self, arr, idx, val):
        arr[int(idx)] = val
        return None

    # === Ciclos ===
    def visitCicloWhile(self, ctx: PELEParser.CicloWhileContext):
        expr_ctx = ctx.expr()
        while True:
            condition = self.visit(expr_ctx)
            if not isinstance(condition, bool):
                raise TypeError(f"La condición de 'mientras' debe ser booleana, no '{type(condition).__name__}'")
            if not condition:
                break
            self.visit(ctx.block())
        return None

    def visitCFor(self, ctx: PELEParser.CForContext):
        init_assign = ctx.assignment(0)
        var_name_init = init_assign.ID().getText()
        init_value = self.visit(init_assign.expr())
        self.set_var(var_name_init, init_value)

        cond_expr = ctx.expr()
        incr_assign = ctx.assignment(1)

        # Loop
        while True:
            cond = self.visit(cond_expr)
            if not isinstance(cond, bool):
                raise TypeError("Condición del for debe ser booleana")
            if not cond:
                break
            # ejecutar cuerpo
            self.visit(ctx.block())
            # ejecutar incremento (manualmente)
            var_name_inc = incr_assign.ID().getText()
            inc_value = self.visit(incr_assign.expr())
            self.set_var(var_name_inc, inc_value)
        return None
    def visitForEach(self, ctx: PELEParser.ForEachContext):
        # FOR '(' ID IN expr ')' '{' block '}'
        var_name = ctx.ID().getText()
        iterable = self.visit(ctx.expr())
        if not isinstance(iterable, list):
            raise TypeError(f"'for-in' requiere un arreglo, no '{type(iterable).__name__}'")
        had_prev = any(var_name in s for s in self.scopes)
        prev_val = None
        if had_prev:
            prev_val = None
            # buscar y guardar previo en el primer scope que lo contenga (desde arriba)
            for s in reversed(self.scopes):
                if var_name in s:
                    prev_val = s[var_name]
                    break
        for item in iterable:
            self.set_var(var_name, item)
            self.visit(ctx.block())
        # restaurar
        if had_prev:
            # asignar en el primer scope que lo contenía originalmente
            for s in reversed(self.scopes):
                if var_name in s:
                    s[var_name] = prev_val
                    break
        else:
            # eliminar de scope actual si existe
            if var_name in self.current_scope():
                del self.current_scope()[var_name]
        return None
