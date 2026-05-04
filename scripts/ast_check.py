import ast
p='neuroslm/intelligence/flow.py'
with open(p,'r',encoding='utf-8') as f:
    s=f.read()
try:
    ast.parse(s)
    print('AST parse ok')
except SyntaxError as e:
    print('SyntaxError:', e)
    print('offset:', e.offset, 'lineno:', e.lineno)
    lines=s.splitlines()
    for i in range(max(0,e.lineno-3), min(len(lines), e.lineno+2)):
        print(i+1, lines[i])
