with open('neuroslm/intelligence/flow.py','rb') as f:
    lines=f.readlines()
for i in range(35,50):
    print(i+1, lines[i])
