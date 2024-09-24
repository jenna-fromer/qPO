dataset_array=['wong_antibiotics']
objective_array=['Mean_50uM']
n_iter = [10]
batch_size = [50]

methods = ['UCB', 'Greedy', 'Ours', 'pTS', 'qEI']

call_strs = []
for dataset, obj, n_i, b_s in zip(dataset_array, objective_array, n_iter, batch_size): 
    for method in methods: 
        call_str = f'python run.py --c=-1 --dataset={dataset} --objective={obj} --method={method} --n_iter={n_i} --batch_size={b_s} --gpu'
        call_strs.append(call_str)

with open('scripts/experiments_antibiotics.txt', 'w') as outfile:
    outfile.write("\n".join(call_strs))