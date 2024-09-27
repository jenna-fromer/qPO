from pathlib import Path 
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import torch
import numpy as np 
import pandas as pd 
import random 
import json 
import copy
import argparse 

from gp import TanimotoGP, fit_gp_hyperparameters
from acquisition_functions import acquire

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='qEI', choices=['Ours', 'pTS', 'Greedy', 'UCB', 'qEI', 'random', 'random_10k'])
    parser.add_argument('--dataset', type=str, default='Lipophilicity')
    parser.add_argument('--objective', type=str, default='exp')
    parser.add_argument('--c', type=int, default=1)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--initial_batch_size', type=int, default=None)
    parser.add_argument('--res_dir', type=str, default='results')
    parser.add_argument('--res_file', type=str, default=None)

    args = parser.parse_args()
    return args 

def smiles_to_fingerprint_arr(
    smiles_list: list[str],
    radius: int = 3, 
    fpSize: int = 2048,
) -> np.array:
    """ Converts a list of SMILES to a numpy array of fingerprints """

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [mfpgen.GetCountFingerprintAsNumPy(m) for m in mols]
    return np.asarray(fps, dtype=float)

def fp_featurizer(smiles_list): 
    """ Generates a dictionary mapping SMILES in the design space to count Morgan fingerprints """
    fps = smiles_to_fingerprint_arr(smiles_list)
    featurizer = {
        smi: fps[i,:]
        for i, smi in enumerate(smiles_list)
    }
    return featurizer

def update_acquired(acquired_data, unacquired_smiles: set, selected_smiles, test_data): 
    for smi in selected_smiles: 
        if smi not in unacquired_smiles:
            print(smi) 
        unacquired_smiles.remove(smi)
        acquired_data[smi] = test_data[smi]
    
    return acquired_data, unacquired_smiles

def train_model(acquired_data, featurizer, gpu: bool = True): 
    """ Train surrogate model on acquired data """
    X_train = np.array([featurizer[smi] for smi in acquired_data])
    y_train = np.array(list(acquired_data.values()))
    if gpu: 
        model = TanimotoGP(
            train_x=torch.as_tensor(X_train).cuda(), train_y=torch.as_tensor(y_train).cuda()
        ).cuda()
    else: 
        model = TanimotoGP(
            train_x=torch.as_tensor(X_train), train_y=torch.as_tensor(y_train)
        )        
    fit_gp_hyperparameters(model)
    # model.train(xs=X_train, ys=y_train)
    return model

def run(
    dataset, objective, 
    c: int = 1, gpu: bool = True, 
    n_iter: int = 10, random_seeds: list = None, 
    batch_size: int = 100, initial_batch_size: int = None, 
    res_dir: str = 'results', method: str = 'ours', 
    res_file: str = None): 

    """ Performs Bayesian optimization loop """

    print('starting run')

    # files 
    datafile = Path('data') / f'{dataset}.csv'
    res_dir = Path(res_dir)
    res_dir.mkdir(exist_ok=True)
    if res_file is None: 
        res_file = f'{dataset}_{objective}_batch{batch_size}_initbatch{initial_batch_size}_method{method}.json'

    if random_seeds is None: 
        random_seeds = range(5)
    
    # load data, initialize featurizer and other objects 
    all_data = pd.read_csv(datafile).sample(frac=1, random_state=0) # shuffle order 
    all_smiles = list(all_data['smiles'])
    featurizer = fp_featurizer(all_smiles)
    storage = []
    test_data = {smi: score for smi, score in zip(all_smiles, all_data[objective])}
    initial_batch_size = initial_batch_size or batch_size

    # run BO 
    for rs in random_seeds: 
        # initialize 
        acquired_data = {}
        unacquired_smiles = list(set(all_smiles))

        random.seed(rs)

        # get initial batch 
        selected_smiles = random.sample(sorted(unacquired_smiles), initial_batch_size)
        acquired_data, unacquired_smiles = update_acquired(acquired_data, unacquired_smiles, selected_smiles, test_data)

        # update storage 
        acq_vals = sorted(acquired_data.values(), key = lambda s: c*s)
        top_aves = {f'Top {k} ave': np.mean(acq_vals[-1*k:]) for k in [1, 10, 50, 100]}
        storage.append({**{
            'Method': method, 
            'Objective': objective, 
            'Dataset': dataset,
            'Iteration': 0,
            'All acquired points': copy.deepcopy(acquired_data),
            'New acquired points': {smi: acquired_data[smi] for smi in selected_smiles},
            'Random seed': rs
        }, **top_aves})

        # train model
        model = train_model(acquired_data, featurizer, gpu=gpu)

        # print
        print(f'METHOD: {method}, SEED: {rs}')
        print(f'\t Iter 0 -- top 1: {np.mean(acq_vals[-1:]):0.2f}, top 10 ave: {np.mean(acq_vals[-10:]):0.2f}, top 50 ave: {np.mean(acq_vals[-50:]):0.2f}')
        
        for iter in range(1, n_iter+1): 
            # acquire 
            selected_smiles = acquire(
                method=method, smiles=unacquired_smiles, 
                model=model, featurizer=featurizer, 
                batch_size=batch_size, gpu=gpu, 
                best_f=max(acq_vals) if c == 1 else min(acq_vals), c=c
            ) 

            # get actual scores and update list of acquired data and unacquired smiles 
            acquired_data, unacquired_smiles = update_acquired(acquired_data, unacquired_smiles, selected_smiles, test_data)

            # update storage 
            acq_vals = sorted(acquired_data.values(), key = lambda s: c*s)
            top_aves = {f'Top {k} ave': np.mean(acq_vals[-1*k:]) for k in [1, 10, 50, 100]}
            storage.append({**{
                'Method': method, 
                'Objective': objective, 
                'Dataset': dataset,
                'Iteration': iter,
                'All acquired points': copy.deepcopy(acquired_data),
                'New acquired points': {smi: acquired_data[smi] for smi in selected_smiles},
                'Random seed': rs
            }, **top_aves})

            # train model 
            model = train_model(acquired_data, featurizer, gpu=gpu)

            # print an update
            print(f'\t Iter {iter} -- top 1: {np.mean(acq_vals[-1:]):0.2f}, top 10 ave: {np.mean(acq_vals[-10:]):0.2f}, top 50 ave: {np.mean(acq_vals[-50:]):0.2f}')

        with open(res_dir / res_file, 'w') as f: 
            json.dump(storage, f, indent='\t')

if __name__=='__main__':
    args = parse_args()
    print(args)
    run(
        dataset=args.dataset,
        objective=args.objective,
        c=args.c, gpu=args.gpu, 
        n_iter=args.n_iter, random_seeds=range(10),
        batch_size=args.batch_size, initial_batch_size=args.initial_batch_size,
        res_dir=args.res_dir, res_file=args.res_file, 
        method=args.method,
    )

