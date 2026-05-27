from PELEVisitor import PELEVisitor
from PELEParser import PELEParser

class ReturnValue(Exception):
    def __init__(self, value):
        self.value = value

class PeleLambda:
    def __init__(self, param, body_ctx, visitor):
        self.param = param
        self.body_ctx = body_ctx
        self.visitor = visitor
    def __repr__(self):
        return f"<lambda {self.param}>"

class EvalVisitor(PELEVisitor):
    def __init__(self):
        self.scopes = [{}]
        self.functions = {}
        self.stop_on_error = False

    def push_scope(self):
        self.scopes.append({})

    def pop_scope(self):
        if len(self.scopes) > 1:
            self.scopes.pop()
        else:
            self.scopes[0].clear()

    def current_scope(self):
        return self.scopes[-1]

    def set_var(self, name, value):
        self.current_scope()[name] = value

    def get_var(self, name):
        for s in reversed(self.scopes):
            if name in s:
                return s[name]
        raise Exception(f"Error: Variable '{name}' no definida.")

    def visitProgram(self, ctx: PELEParser.ProgramContext):
        return self.visit(ctx.block())

    def visitBlock(self, ctx: PELEParser.BlockContext):
        for stmt in ctx.statement():
            try:
                self.visit(stmt)
            except ReturnValue:
                raise
            except Exception as e:
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

    def visitAssignStmt(self, ctx: PELEParser.AssignStmtContext):
        assign_ctx = ctx.assignment()
        var_name = assign_ctx.ID().getText()
        value = self.visit(assign_ctx.expr())
        self.set_var(var_name, value)
        return value

    def _format_value(self, v):
        if isinstance(v, PeleLambda):
            return repr(v)
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
        if isinstance(v, bool):
            return "true" if v else "false"
        return repr(v)

    def visitMostrarStmt(self, ctx: PELEParser.MostrarStmtContext):
        value = self.visit(ctx.expr())
        print("> " + self._format_value(value))
        return None

    def visitExprStmt(self, ctx: PELEParser.ExprStmtContext):
        return self.visit(ctx.expr())

    # === Expressions ===
    def visitUnaryMinusExpr(self, ctx: PELEParser.UnaryMinusExprContext):
        val = self.visit(ctx.expr())
        if isinstance(val, list):
            return [-v for v in val]
        return -val

    def visitPowerExpr(self, ctx: PELEParser.PowerExprContext):
        return self.visit(ctx.expr(0)) ** self.visit(ctx.expr(1))

    def visitMulDivModExpr(self, ctx: PELEParser.MulDivModExprContext):
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        op = ctx.getChild(1).getText()
        if op == '*':
            if isinstance(left, (int, float)) and isinstance(right, list):
                return [left * r for r in right]
            if isinstance(left, list) and isinstance(right, (int, float)):
                return [l * right for l in left]
            return left * right
        if op == '/':
            if right == 0:
                raise Exception("Division por cero.")
            return left / right
        if op == '%': return left % right

    def visitAddSubExpr(self, ctx: PELEParser.AddSubExprContext):
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        op = ctx.getChild(1).getText()
        if isinstance(left, list) and isinstance(right, list):
            if op == '+': return [l + r for l, r in zip(left, right)]
            if op == '-': return [l - r for l, r in zip(left, right)]
        if op == '+': return left + right
        if op == '-': return left - right

    def visitEqExpr(self, ctx: PELEParser.EqExprContext):
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        op = ctx.getChild(1).getText()
        if op == '==': return left == right
        if op == '!=': return left != right

    def visitRelationalExpr(self, ctx: PELEParser.RelationalExprContext):
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        op = ctx.getChild(1).getText()
        if op == '<':  return left < right
        if op == '<=': return left <= right
        if op == '>':  return left > right
        if op == '>=': return left >= right

    # === NEW: Logical operators ===
    def visitAndExpr(self, ctx: PELEParser.AndExprContext):
        left = self.visit(ctx.expr(0))
        if not bool(left): return False
        return bool(self.visit(ctx.expr(1)))

    def visitOrExpr(self, ctx: PELEParser.OrExprContext):
        left = self.visit(ctx.expr(0))
        if bool(left): return True
        return bool(self.visit(ctx.expr(1)))

    def visitNotExpr(self, ctx: PELEParser.NotExprContext):
        return not bool(self.visit(ctx.expr()))

    # === NEW: Pipe operator ===
    def visitPipeExpr(self, ctx: PELEParser.PipeExprContext):
        value = self.visit(ctx.expr(0))
        fn = self.visit(ctx.expr(1))
        return self._apply_callable(fn, [value])

    # === NEW: Lambda ===
    def visitLambdaExpr(self, ctx: PELEParser.LambdaExprContext):
        param = ctx.ID().getText()
        return PeleLambda(param, ctx.expr(), self)

    # === NEW: Postfix delegation ===
    def visitPostfixExpr(self, ctx: PELEParser.PostfixExprContext):
        return self.visit(ctx.postfix())

    def visitAtomExpr(self, ctx: PELEParser.AtomExprContext):
        return self.visit(ctx.atom())

    # === NEW: Index access ===
    def visitIndexExpr(self, ctx: PELEParser.IndexExprContext):
        obj = self.visit(ctx.postfix())
        idx = self.visit(ctx.expr())
        if isinstance(obj, list):
            return obj[int(idx)]
        if isinstance(obj, dict):
            key = idx if isinstance(idx, str) else str(idx)
            if key not in obj:
                raise Exception(f"Clave '{key}' no existe en el mapa.")
            return obj[key]
        if isinstance(obj, str):
            return obj[int(idx)]
        raise Exception("El operador [] requiere lista, mapa o texto.")

    # === NEW: Method calls ===
    def visitMethodCallExpr(self, ctx: PELEParser.MethodCallExprContext):
        obj = self.visit(ctx.postfix())
        method = ctx.ID().getText()
        args = [self.visit(e) for e in ctx.expr()]
        if isinstance(obj, list):
            if method == 'len': return len(obj)
            if method == 'head':
                if not obj: raise Exception("head() en lista vacia.")
                return obj[0]
            if method == 'tail': return obj[1:]
            if method == 'reverse': return obj[::-1]
            if method == 'slice': return obj[int(args[0]):int(args[1])]
            if method == 'contains': return args[0] in obj
            if method == 'get': return obj[int(args[0])]
            if method == 'append': return obj + [args[0]]
            raise Exception(f"Lista no tiene metodo '{method}'.")
        if isinstance(obj, dict):
            if method == 'get':
                k = args[0] if isinstance(args[0], str) else str(args[0])
                if k not in obj: raise Exception(f"Clave '{k}' no existe.")
                return obj[k]
            if method == 'keys': return list(obj.keys())
            if method == 'values': return list(obj.values())
            if method == 'has':
                k = args[0] if isinstance(args[0], str) else str(args[0])
                return k in obj
            if method == 'set':
                k = args[0] if isinstance(args[0], str) else str(args[0])
                new_d = dict(obj); new_d[k] = args[1]; return new_d
            raise Exception(f"Mapa no tiene metodo '{method}'.")
        if isinstance(obj, str):
            if method == 'len': return len(obj)
            if method == 'contains': return args[0] in obj
            raise Exception(f"Texto no tiene metodo '{method}'.")
        raise Exception(f"Objeto no soporta metodos (tipo: {type(obj).__name__}).")

    # === NEW: Dict literals ===
    def visitEmptyDictExpr(self, ctx: PELEParser.EmptyDictExprContext):
        return {}

    def visitDictLiteralExpr(self, ctx: PELEParser.DictLiteralExprContext):
        result = {}
        for entry in ctx.dictEntry():
            key_token = entry.getChild(0).getText()
            key = key_token[1:-1] if key_token.startswith('"') else key_token
            result[key] = self.visit(entry.expr())
        return result

    def visitArrayExpr(self, ctx: PELEParser.ArrayExprContext):
        exprs = list(ctx.expr()) if ctx.expr() else []
        return [self.visit(expr) for expr in exprs]

    def visitBoolExpr(self, ctx: PELEParser.BoolExprContext):
        return ctx.getText() == 'true'

    def visitStringExpr(self, ctx: PELEParser.StringExprContext):
        text = ctx.getText()[1:-1]
        try:
            return text.encode('utf-8').decode('unicode_escape')
        except Exception:
            return text

    def visitIntExpr(self, ctx: PELEParser.IntExprContext):
        return int(ctx.getText())

    def visitFloatExpr(self, ctx: PELEParser.FloatExprContext):
        return float(ctx.getText())

    def visitIdExpr(self, ctx: PELEParser.IdExprContext):
        var_name = ctx.getText()
        try:
            return self.get_var(var_name)
        except Exception:
            if var_name in self.builtins() or var_name in self.functions:
                return var_name
            raise

    def visitParensExpr(self, ctx: PELEParser.ParensExprContext):
        return self.visit(ctx.expr())

    # === If (now supports else-if chaining) ===
    def visitIfStmt(self, ctx: PELEParser.IfStmtContext):
        return self.visit(ctx.ifStatement())

    def visitIfStatement(self, ctx: PELEParser.IfStatementContext):
        condition = self.visit(ctx.expr())
        if condition:
            return self.visit(ctx.block(0))
        # Check for else-if chain
        if ctx.ifStatement() is not None:
            return self.visit(ctx.ifStatement())
        # Check for else block
        blocks = list(ctx.block())
        if len(blocks) > 1:
            return self.visit(blocks[1])
        return None

    def visitReturnStmt(self, ctx: PELEParser.ReturnStmtContext):
        value = self.visit(ctx.expr())
        raise ReturnValue(value)

    def visitFunctionDeclStmt(self, ctx: PELEParser.FunctionDeclStmtContext):
        func_ctx = ctx.functionDecl()
        name = func_ctx.ID().getText()
        params = []
        if func_ctx.params():
            params = [p.getText() for p in func_ctx.params().ID()]
        block = func_ctx.block()
        self.functions[name] = {'params': params, 'block': block}
        return None

    # === Callable helper (for pipe and first-class functions) ===
    def _apply_callable(self, fn, args):
        if isinstance(fn, PeleLambda):
            fn.visitor.push_scope()
            try:
                fn.visitor.set_var(fn.param, args[0])
                result = fn.visitor.visit(fn.body_ctx)
                return result
            finally:
                fn.visitor.pop_scope()
        if isinstance(fn, str):
            built = self.builtins()
            if fn in built:
                return built[fn](*args)
            if fn in self.functions:
                return self._call_user_func(fn, args)
        raise Exception(f"|> requiere funcion, recibio {type(fn).__name__}.")

    def _call_user_func(self, func_name, args):
        func_info = self.functions[func_name]
        param_names = func_info['params']
        if len(args) != len(param_names):
            raise Exception(f"Funcion '{func_name}' espera {len(param_names)} argumentos, recibio {len(args)}.")
        self.push_scope()
        try:
            for pname, aval in zip(param_names, args):
                self.set_var(pname, aval)
            try:
                self.visit(func_info['block'])
                return None
            except ReturnValue as r:
                return r.value
        finally:
            self.pop_scope()

    # === Function calls ===
    def visitFuncCallExpr(self, ctx: PELEParser.FuncCallExprContext):
        func_name = ctx.ID().getText()
        args = [self.visit(e) for e in ctx.expr()]

        built = self.builtins()
        if func_name in built:
            try:
                return built[func_name](*args)
            except TypeError as e:
                line = ctx.start.line if hasattr(ctx, 'start') else '?'
                raise Exception(f"[Linea {line}] Error llamando a builtin '{func_name}': {e}")
            except Exception as e:
                line = ctx.start.line if hasattr(ctx, 'start') else '?'
                raise Exception(f"[Linea {line}] {e}")

        if func_name in self.functions:
            return self._call_user_func(func_name, args)

        # Try as variable holding a callable
        try:
            fn_val = self.get_var(func_name)
            if isinstance(fn_val, PeleLambda):
                return self._apply_callable(fn_val, args)
            if isinstance(fn_val, str):
                return self._apply_callable(fn_val, args)
        except Exception:
            pass

        line = ctx.start.line if hasattr(ctx, 'start') else '?'
        raise Exception(f"[Linea {line}] Funcion '{func_name}' no definida.")

    # === Builtins ===
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
            # Helpers arrays
            "arr_get": self._arr_get,
            "arr_set": self._arr_set,
            # Primitivas del lenguaje
            "longitud": lambda x: len(x) if isinstance(x, (list, dict, str, set)) else 0,
            "len": lambda x: len(x) if isinstance(x, (list, dict, str, set)) else 0,
            "es_numero": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
            "es_arreglo": lambda x: isinstance(x, list),
            "es_mapa": lambda x: isinstance(x, dict),
            "es_texto": lambda x: isinstance(x, str),
            "error": self._builtin_error,
            "tipo": lambda x: type(x).__name__,
            "entero": lambda x: int(x),
            "decimal": lambda x: float(x),
            "a_texto": lambda x: str(x),
            "escribir_archivo": self._builtin_escribir_archivo,
            "leer_archivo": self._builtin_leer_archivo,
            # NEW: Functional builtins
            "head": self._builtin_head,
            "tail": self._builtin_tail,
            "append": self._builtin_append,
            "range": self._builtin_range,
            "abs": lambda x: x if x >= 0 else -x,
            "concatenar": lambda a, b: a + b,
            "min": lambda a, b: a if a < b else b,
            "max": lambda a, b: a if a > b else b,
            "piso": lambda x: int(x),
        }

    def _builtin_error(self, msg):
        raise Exception(f"Error desde PELE: {msg}")

    def _builtin_escribir_archivo(self, ruta, contenido):
        with open(ruta, "w", encoding="utf-8") as f:
            f.write(contenido)
        return 0

    def _builtin_leer_archivo(self, ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            return f.read()

    def _builtin_head(self, lst):
        if not isinstance(lst, list) or not lst:
            raise Exception("head() requiere lista no vacia.")
        return lst[0]

    def _builtin_tail(self, lst):
        if not isinstance(lst, list):
            raise Exception("tail() requiere lista.")
        return lst[1:]

    def _builtin_append(self, lst, elem):
        if not isinstance(lst, list):
            raise Exception("append() requiere lista.")
        return lst + [elem]

    def _builtin_range(self, *args):
        if len(args) == 1: return list(range(int(args[0])))
        if len(args) == 2: return list(range(int(args[0]), int(args[1])))
        return list(range(int(args[0]), int(args[1]), int(args[2])))

    # Mapas
    def _crear_mapa(self, pairs=None):
        if pairs is None: return {}
        m = {}
        for k, v in pairs:
            m[k] = v
        return m

    def _mapa_put(self, m, key, val):
        if not isinstance(m, dict): raise Exception("mapa_put: primer arg no es mapa")
        m[key] = val
        return val

    def _mapa_get(self, m, key):
        if not isinstance(m, dict): raise Exception(f"mapa_get: '{m}' no es un mapa")
        if key not in m: raise Exception(f"mapa_get: llave '{key}' no existe")
        return m[key]

    def _mapa_keys(self, m):
        if not isinstance(m, dict): raise Exception("mapa_keys: argumento debe ser un mapa")
        return list(m.keys())

    def _mapa_values(self, m):
        if not isinstance(m, dict): raise Exception("mapa_values: argumento debe ser un mapa")
        return list(m.values())

    # Pilas
    def _crear_pila(self): return []
    def _pila_push(self, pila, val):
        if not isinstance(pila, list): raise Exception("pila_push: primer arg no es pila")
        pila.append(val); return val
    def _pila_pop(self, pila):
        if not isinstance(pila, list): raise Exception("pila_pop: primer arg no es pila")
        if len(pila) == 0: raise Exception("pila_pop: pila vacia")
        return pila.pop()

    # Colas
    def _crear_cola(self): return []
    def _cola_enqueue(self, queue, value):
        if not isinstance(queue, list): raise Exception("cola_enqueue: primer arg debe ser cola")
        queue.append(value); return None
    def _cola_dequeue(self, queue):
        if not isinstance(queue, list): raise Exception("cola_dequeue: primer arg debe ser cola")
        if not queue: raise Exception("Error: cola vacia")
        return queue.pop(0)

    # Conjuntos
    def _crear_conjunto(self, arr=None):
        if arr is None: return set()
        return set(arr)
    def _conjunto_add(self, s, v):
        if not isinstance(s, set): raise Exception("conjunto_add: primer arg debe ser conjunto")
        s.add(v); return None
    def _conjunto_contains(self, s, v):
        if not isinstance(s, set): raise Exception("conjunto_contains: primer arg debe ser conjunto")
        return v in s

    # Matrices
    def _crear_matriz(self, rows, cols, fill):
        return [[fill for _ in range(int(cols))] for _ in range(int(rows))]
    def _mat_get(self, mat, i, j): return mat[int(i)][int(j)]
    def _mat_set(self, mat, i, j, val): mat[int(i)][int(j)] = val; return None

    # Arboles
    def _crear_arbol(self, value): return {'value': value, 'children': []}
    def _arbol_add_child(self, node, child):
        if not isinstance(node, dict) or 'children' not in node:
            raise Exception("arbol_add_child: primer arg no es nodo de arbol")
        node['children'].append(child); return child
    def _arbol_preorder(self, node):
        res = []
        def _rec(n):
            if not isinstance(n, dict) or 'children' not in n: return
            res.append(n['value'])
            for c in n['children']: _rec(c)
        _rec(node)
        return res

    # Grafos
    def _crear_grafo(self): return {}
    def _grafo_add_node(self, g, node):
        if not isinstance(g, dict): raise Exception("grafo_add_node: primer arg debe ser grafo")
        if node not in g: g[node] = []
        return None
    def _grafo_add_edge(self, g, u, v):
        if not isinstance(g, dict): raise Exception("grafo_add_edge: primer arg debe ser grafo")
        if u not in g: g[u] = []
        if v not in g: g[v] = []
        g[u].append(v); return None
    def _grafo_neighbors(self, g, node):
        if not isinstance(g, dict): raise Exception("grafo_neighbors: primer arg debe ser grafo")
        return g.get(node, [])
    def _grafo_bfs(self, g, start):
        if not isinstance(g, dict): raise Exception("grafo_bfs: primer arg debe ser grafo")
        visited = set(); queue = []; order = []
        queue.append(start); visited.add(start)
        while queue:
            u = queue.pop(0); order.append(u)
            for v in g.get(u, []):
                if v not in visited: visited.add(v); queue.append(v)
        return order

    def _arr_get(self, arr, idx): return arr[int(idx)]
    def _arr_set(self, arr, idx, val): arr[int(idx)] = val; return None

    # === Ciclos ===
    def visitCicloWhile(self, ctx: PELEParser.CicloWhileContext):
        while True:
            condition = self.visit(ctx.expr())
            if not condition: break
            self.visit(ctx.block())
        return None

    def visitCFor(self, ctx: PELEParser.CForContext):
        init_assign = ctx.assignment(0)
        var_name_init = init_assign.ID().getText()
        init_value = self.visit(init_assign.expr())
        self.set_var(var_name_init, init_value)
        cond_expr = ctx.expr()
        incr_assign = ctx.assignment(1)
        while True:
            cond = self.visit(cond_expr)
            if not cond: break
            self.visit(ctx.block())
            var_name_inc = incr_assign.ID().getText()
            inc_value = self.visit(incr_assign.expr())
            self.set_var(var_name_inc, inc_value)
        return None

    def visitForEach(self, ctx: PELEParser.ForEachContext):
        var_name = ctx.ID().getText()
        iterable = self.visit(ctx.expr())
        if isinstance(iterable, str):
            iterable = list(iterable)
        if not isinstance(iterable, list):
            raise TypeError(f"'for-in' requiere un arreglo o texto, no '{type(iterable).__name__}'")
        had_prev = any(var_name in s for s in self.scopes)
        prev_val = None
        if had_prev:
            for s in reversed(self.scopes):
                if var_name in s:
                    prev_val = s[var_name]; break
        for item in iterable:
            self.set_var(var_name, item)
            self.visit(ctx.block())
        if had_prev:
            for s in reversed(self.scopes):
                if var_name in s:
                    s[var_name] = prev_val; break
        else:
            if var_name in self.current_scope():
                del self.current_scope()[var_name]
        return None
