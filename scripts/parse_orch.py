import ast, traceback
p='neuroslm/intelligence/orchestrator.py'
try:
    s=open(p,'r',encoding='utf-8').read()
    ast.parse(s)
    print('OK')
except Exception as e:
    traceback.print_exc()
    import sys
    print('ExceptionType:', type(e))
    sys.exit(1)
