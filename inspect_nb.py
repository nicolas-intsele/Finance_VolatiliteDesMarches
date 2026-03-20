import json
nb_path = r'c:\Users\nicol\OneDrive\Bureau\Finance\notebooks\05_deep_learning.ipynb'
with open(nb_path,'r',encoding='utf-8') as f:
    nb = json.load(f)
for i,cell in enumerate(nb['cells']):
    if cell['cell_type']=='code':
        src = ''.join(cell['source'])
        if 'Sequential' in src or 'model_lstm' in src or 'X_test' in src:
            print(f'--- CELL {i} ---')
            print(src)
            print()
