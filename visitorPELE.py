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

    # ── Scope Management ──────────────────────────────────────
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

    # ── Program / Block ───────────────────────────────────────
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

    # ── Statements ────────────────────────────────────────────
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
            items = ", ".join(f"{repr(k)}: {self._format_value(val)}" for k, val in v.items())
            return "{" + items + "}"
        if isinstance(v, list):
            inner = ", ".join(self._format_value(x) for x in v)
            return "[" + inner + "]"
        if isinstance(v, bool):
            return "true" if v else "false"
        return repr(v)

    def visitMostrarStmt(self, ctx: PELEParser.MostrarStmtContext):
        value = self.visit(ctx.expr())
        print("> " + self._format_value(value))
        return None

    def visitExprStmt(self, ctx: PELEParser.ExprStmtContext):
        return self.visit(ctx.expr())

    # ── Arithmetic Operators ──────────────────────────────────
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
        if op == '%':
            return left % right

    def visitAddSubExpr(self, ctx: PELEParser.AddSubExprContext):
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        op = ctx.getChild(1).getText()
        if isinstance(left, list) and isinstance(right, list):
            if op == '+': return [l + r for l, r in zip(left, right)]
            if op == '-': return [l - r for l, r in zip(left, right)]
        if op == '+': return left + right
        if op == '-': return left - right

    # ── Comparison Operators ──────────────────────────────────
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

    # ── Logical Operators ─────────────────────────────────────
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

    # ── Pipe Operator ─────────────────────────────────────────
    def visitPipeExpr(self, ctx: PELEParser.PipeExprContext):
        value = self.visit(ctx.expr(0))
        fn = self.visit(ctx.expr(1))
        return self._apply_callable(fn, [value])

    # ── Lambda ────────────────────────────────────────────────
    def visitLambdaExpr(self, ctx: PELEParser.LambdaExprContext):
        param = ctx.ID().getText()
        return PeleLambda(param, ctx.expr(), self)

    # ── Postfix (index, method) ───────────────────────────────
    def visitPostfixExpr(self, ctx: PELEParser.PostfixExprContext):
        return self.visit(ctx.postfix())

    def visitAtomExpr(self, ctx: PELEParser.AtomExprContext):
        return self.visit(ctx.atom())

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

    def visitMethodCallExpr(self, ctx: PELEParser.MethodCallExprContext):
        obj = self.visit(ctx.postfix())
        method = ctx.ID().getText()
        args = [self.visit(e) for e in ctx.expr()]
        # Listas
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
        # Diccionarios
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
        # Strings
        if isinstance(obj, str):
            if method == 'len': return len(obj)
            if method == 'contains': return args[0] in obj
            raise Exception(f"Texto no tiene metodo '{method}'.")
        raise Exception(f"Objeto no soporta metodos (tipo: {type(obj).__name__}).")

    # ── Literals ──────────────────────────────────────────────
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

    # ── Control Flow ──────────────────────────────────────────
    def visitIfStmt(self, ctx: PELEParser.IfStmtContext):
        return self.visit(ctx.ifStatement())

    def visitIfStatement(self, ctx: PELEParser.IfStatementContext):
        condition = self.visit(ctx.expr())
        if condition:
            return self.visit(ctx.block(0))
        if ctx.ifStatement() is not None:
            return self.visit(ctx.ifStatement())
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

    # ── Callable (pipe, first-class functions) ────────────────
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

    # ── Function Calls ────────────────────────────────────────
    def visitFuncCallExpr(self, ctx: PELEParser.FuncCallExprContext):
        func_name = ctx.ID().getText()
        args = [self.visit(e) for e in ctx.expr()]

        built = self.builtins()
        if func_name in built:
            try:
                return built[func_name](*args)
            except TypeError as e:
                line = ctx.start.line if hasattr(ctx, 'start') else '?'
                raise Exception(f"[Linea {line}] Error en builtin '{func_name}': {e}")
            except Exception as e:
                line = ctx.start.line if hasattr(ctx, 'start') else '?'
                raise Exception(f"[Linea {line}] {e}")

        if func_name in self.functions:
            return self._call_user_func(func_name, args)

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

    # ── Loops ─────────────────────────────────────────────────
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
            raise TypeError(f"'for-in' requiere arreglo o texto, no '{type(iterable).__name__}'")
        for item in iterable:
            self.set_var(var_name, item)
            self.visit(ctx.block())
        return None

    # ══════════════════════════════════════════════════════════
    # BUILTINS MINIMOS — Solo primitivas imposibles en PELE
    # (acceso a memoria, tipos, I/O, conversión)
    # ══════════════════════════════════════════════════════════
    def builtins(self):
        return {
            # Acceso a memoria de listas (imposible en PELE puro)
            "arr_get": lambda arr, idx: arr[int(idx)],
            "arr_set": self._arr_set,
            # Longitud (requiere len() de Python)
            "longitud": lambda x: len(x) if isinstance(x, (list, dict, str)) else 0,
            "len": lambda x: len(x) if isinstance(x, (list, dict, str)) else 0,
            # Introspección de tipos (requiere isinstance de Python)
            "es_numero": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
            "es_arreglo": lambda x: isinstance(x, list),
            "es_mapa": lambda x: isinstance(x, dict),
            "es_texto": lambda x: isinstance(x, str),
            "tipo": lambda x: type(x).__name__,
            # Conversión de tipos (requiere int/float/str de Python)
            "entero": lambda x: int(x),
            "decimal": lambda x: float(x),
            "a_texto": lambda x: str(x),
            # I/O (requiere open() de Python)
            "escribir_archivo": self._escribir_archivo,
            "leer_archivo": self._leer_archivo,
            # Control de flujo
            "error": self._error,
            # Primitivas funcionales atómicas
            "head": self._head,
            "tail": self._tail,
            "append": self._append,
            "range": self._range,
            "abs": lambda x: x if x >= 0 else -x,
            "min": lambda a, b: a if a < b else b,
            "max": lambda a, b: a if a > b else b,
            "piso": lambda x: int(x),
            "concatenar": lambda a, b: a + b,
            # Mapas — primitivas (Python dict access)
            "mapa_get": self._mapa_get,
            "mapa_put": self._mapa_put,
            "mapa_keys": lambda m: list(m.keys()),
            "mapa_values": lambda m: list(m.values()),
            "crear_mapa": self._crear_mapa,
            # Construcción mutable de listas (primitiva del runtime,
            # imposible en PELE puro ya que el lenguaje es funcional)
            "crear_pila": lambda: [],
            "pila_push": self._pila_push,
            "pila_pop": self._pila_pop,
        }

    def _arr_set(self, arr, idx, val):
        arr[int(idx)] = val
        return None

    def _escribir_archivo(self, ruta, contenido):
        with open(ruta, "w", encoding="utf-8") as f:
            f.write(contenido)
        return 0

    def _leer_archivo(self, ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            return f.read()

    def _error(self, msg):
        raise Exception(f"Error desde PELE: {msg}")

    def _head(self, lst):
        if not isinstance(lst, list) or not lst:
            raise Exception("head() requiere lista no vacia.")
        return lst[0]

    def _tail(self, lst):
        if not isinstance(lst, list):
            raise Exception("tail() requiere lista.")
        return lst[1:]

    def _append(self, lst, elem):
        if not isinstance(lst, list):
            raise Exception("append() requiere lista.")
        return lst + [elem]

    def _range(self, *args):
        if len(args) == 1: return list(range(int(args[0])))
        if len(args) == 2: return list(range(int(args[0]), int(args[1])))
        return list(range(int(args[0]), int(args[1]), int(args[2])))

    def _mapa_get(self, m, key):
        if not isinstance(m, dict):
            raise Exception(f"mapa_get: '{m}' no es un mapa")
        if key not in m:
            raise Exception(f"mapa_get: llave '{key}' no existe")
        return m[key]

    def _mapa_put(self, m, key, val):
        if not isinstance(m, dict):
            raise Exception("mapa_put: primer arg no es mapa")
        m[key] = val
        return val

    def _crear_mapa(self, pairs=None):
        if pairs is None:
            return {}
        m = {}
        for p in pairs:
            m[p[0]] = p[1]
        return m

    def _pila_push(self, pila, val):
        if not isinstance(pila, list):
            raise Exception("pila_push: primer arg no es pila")
        pila.append(val)
        return val

    def _pila_pop(self, pila):
        if not isinstance(pila, list):
            raise Exception("pila_pop: primer arg no es pila")
        if len(pila) == 0:
            raise Exception("pila_pop: pila vacia")
        return pila.pop()

