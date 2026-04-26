"""Minimal Lisp-like DSL interpreter for evolvable brain DNA."""
import operator as op
import math

class LispError(Exception): pass

class Env(dict):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
    def find(self, var):
        if var in self:
            return self
        elif self.parent:
            return self.parent.find(var)
        else:
            raise LispError(f"Unbound symbol: {var}")

# Built-in functions
BUILTINS = {
    '+': op.add, '-': op.sub, '*': op.mul, '/': op.truediv,
    '>': op.gt, '<': op.lt, '>=': op.ge, '<=': op.le, '=': op.eq,
    'abs': abs, 'max': max, 'min': min, 'pow': pow, 'sqrt': math.sqrt,
    'and': lambda *a: all(a), 'or': lambda *a: any(a), 'not': op.not_,
    'print': print,
    'append': lambda a, b: (a or []) + (b or []),
    'take': lambda seq, n: (seq or [])[:int(n)],
    'list': lambda *args: list(args),
    'get': lambda obj, key, default=None: (obj.get(key[1:], default) if isinstance(key, str) and key.startswith("'") else (obj.get(key, default) if hasattr(obj, 'get') else default)),
    # DNA helpers
    'projection': lambda src, tgt, type_, nt, cond: ('projection', src, tgt, type_, nt, cond),
    'nt_prod': lambda nt, amount_fn, ctx_fn: ('nt_prod', nt, amount_fn, ctx_fn),
    'score-candidate': lambda c, novelty, alpha: 0.0,
    'score-action': lambda a, th: 0.0,
    'query-memory': lambda q, k: [],
}

# Parser
def parse(src):
    """Parse Lisp source into nested Python lists."""
    # Remove comments starting with ';' to the end of line so they don't become tokens
    lines = src.splitlines()
    cleaned_lines = [line.partition(';')[0] for line in lines]
    src = '\n'.join(cleaned_lines)
    # Treat square-bracket vectors as literal lists: map [ ... ] -> (vector ...)
    src = src.replace('[', ' (vector ').replace(']', ' ) ')
    # Tokenize: keep parentheses as separate tokens
    src = src.replace('(', ' ( ').replace(')', ' ) ')
    tokens = src.split()

    def atom(token):
        # Try int, then float, otherwise return raw token
        if token == "true":
            return True
        if token == "false":
            return False
        if token == "nil":
            return None
        if token.startswith("'"):
            # keep quoted symbols as-is (e.g. 'PFC)
            return token
        if token.lstrip('-').isdigit():
            return int(token)
        try:
            return float(token)
        except ValueError:
            return token

    def read_from(tokens_list):
        if not tokens_list:
            raise LispError('Unexpected EOF')
        token = tokens_list.pop(0)
        if token == '(':
            L = []
            while tokens_list:
                if tokens_list[0] == ')':
                    tokens_list.pop(0)  # consume ')'
                    return L
                L.append(read_from(tokens_list))
            # If we exit the loop without finding a ')', it's an EOF
            raise LispError('Unexpected EOF')
        elif token == ')':
            raise LispError("Unexpected )")
        else:
            return atom(token)

    ast = []
    while tokens:
        ast.append(read_from(tokens))
    return ast

# Evaluator
def eval(x, env):
    if isinstance(x, str):
        # Quoted symbol like 'PFC should be returned as string literal
        if x.startswith("'"):
            return x
        try:
            return env.find(x)[x]
        except LispError:
            # Return unbound symbol as literal string to be used as tags/identifiers
            return x
    elif not isinstance(x, list):
        return x
    if not x:
        return None
    # vector special form: literal list
    if x[0] == 'vector':
        return [eval(el, env) for el in x[1:]]
    # Special form: (region NAME body...)
    if x[0] == 'region':
        # ignore the region name when evaluating per-region files; evaluate body forms into env
        _, name, *forms = x
        env['__region__'] = name
        res = None
        for f in forms:
            res = eval(f, env)
        return res
    # lambda: (lambda (params) body)
    if x[0] == 'lambda':
        (_, params, body) = x
        def _anon(*args):
            local = Env(dict(zip(params, args)), parent=env)
            return eval(body, local)
        return _anon
    # let: (let ((a v) (b v2)) body)
    if x[0] == 'let':
        (_, bindings, body) = x
        local_bindings = {}
        for pair in bindings:
            if isinstance(pair, list) and len(pair) == 2:
                var = pair[0]
                val = eval(pair[1], env)
                local_bindings[var] = val
        return eval(body, Env(local_bindings, parent=env))
    # foreach: (foreach var collection body)
    if x[0] == 'foreach':
        (_, var, collection_expr, body) = x
        coll = eval(collection_expr, env) or []
        res = None
        for item in coll:
            local = Env({var: item}, parent=env)
            res = eval(body, local)
        return res
    if x[0] == 'def':
        (_, var, exp) = x
        env[var] = eval(exp, env)
    elif x[0] == 'set!':
        (_, var, exp) = x
        env.find(var)[var] = eval(exp, env)
    elif x[0] == 'if':
        (_, test, conseq, alt) = x
        return eval((conseq if eval(test, env) else alt), env)
    elif x[0] == 'while':
        (_, test, body) = x
        while eval(test, env):
            eval(body, env)
    elif x[0] == 'defun':
        (_, name, params, body) = x
        env[name] = lambda *args: eval(body, Env(dict(zip(params, args), parent=env)))
    else:
        proc = eval(x[0], env)
        if not callable(proc):
            raise LispError(f"Attempted to call non-callable object: {proc}")
        args = [eval(arg, env) for arg in x[1:]]
        return proc(*args)

# Top-level runner
class LispVM:
    def __init__(self, extra_builtins=None):
        self.env = Env()
        self.env.update(BUILTINS)
        if extra_builtins:
            self.env.update(extra_builtins)
    def run(self, src):
        ast = parse(src)
        res = None
        for expr in ast:
            try:
                res = eval(expr, self.env)
            except Exception as e:
                # Attach the failing expression to the exception for easier debugging
                raise LispError(f"Error evaluating expression {expr!r}: {e}") from e
        # After running, extract neural architecture and projections/NT logic if present
        for key in ["layers", "connections", "learning_rule", "projections", "nt_production"]:
            if key in self.env:
                self.env[key] = self.env[key]
        return res
