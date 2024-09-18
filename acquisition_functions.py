from scipy.stats import multivariate_normal
from gp import TanimotoGP
import torch
import botorch 
import numpy as np 

def acquire(method, smiles, model, featurizer, gpu: bool = True, c: int = 1, batch_size: int = 100, best_f: float = None, **kwargs):
    acq_functions = {
        'Greedy': acquire_mean, 
        'UCB': acquire_ucb, 
        'Ours': acquire_ours, 
        'pTS': acquire_ts,
        'qEI': acquire_sequential_qei,
    }
    if ( method == 'Ours' or method == 'pTS' or method == 'qEI') and len(smiles) > 5000: 
        smiles_filtered = acq_functions['Greedy'](
            smiles=smiles, 
            model=model, 
            featurizer=featurizer, 
            gpu=gpu, c=c, batch_size=5000, **kwargs
        )
        return acq_functions[method](
            smiles=smiles_filtered, 
            model=model, 
            featurizer=featurizer, 
            gpu=gpu, c=c, batch_size=batch_size, 
            best_f=best_f, **kwargs
        )
    
    return acq_functions[method](
        smiles=smiles, 
        model=model, 
        featurizer=featurizer, 
        gpu=gpu, c=c, batch_size=batch_size, 
        best_f=best_f, **kwargs
    )

def mean_cov_from_gp(model: TanimotoGP, smiles: list, featurizer: dict, full_cov: bool = True, gpu: bool = True): 
    model.eval()
    model.likelihood.eval()
    X_test = np.array([featurizer[smi] for smi in smiles])
    if gpu: 
        f_preds = model.likelihood(model(torch.as_tensor(X_test).cuda()))
        if full_cov: 
            return f_preds.mean.cpu().detach().numpy(), f_preds.covariance_matrix.cpu().detach().numpy()
        return f_preds.mean.cpu().detach().numpy(), f_preds.variance.cpu().detach().numpy()

    f_preds = model.likelihood(model(torch.as_tensor(X_test)))
    if full_cov: 
        # mean, cov = model.model.posterior().predict_f(X_test, full_cov=True)
        return f_preds.mean.detach().numpy(), f_preds.covariance_matrix.detach().numpy()
    # mean, var = model.model.posterior().predict_f(X_test, full_cov=False)
    return f_preds.mean.detach().numpy(), f_preds.variance.detach().numpy() # np.squeeze(mean.numpy()), np.squeeze(var.numpy())

def acquire_mean(smiles, model, featurizer, gpu, c: int = 1, batch_size: int = 100, **kwargs): 
    mean, _ = mean_cov_from_gp(smiles=smiles, model=model, featurizer=featurizer, full_cov=False, gpu=gpu)
    acquisition_scores = {smi: score for smi, score in zip(smiles, c*mean)}
    sorted_smis = sorted(smiles, key=lambda smi: -1*acquisition_scores[smi])
    return sorted_smis[:batch_size]

def acquire_ucb(smiles, model, featurizer, gpu, c: int = 1, batch_size: int = 100, beta: float = 1, **kwargs): 
    mean, var = mean_cov_from_gp(smiles=smiles, model=model, featurizer=featurizer, full_cov=False, gpu=gpu)
    acquisition_scores = {smi: score for smi, score in zip(smiles, c*mean + beta*np.sqrt(var))}
    sorted_smis = sorted(smiles, key=lambda smi: -1*acquisition_scores[smi])
    return sorted_smis[:batch_size]

def acquire_ours(smiles, model, featurizer, gpu, c: int = 1, batch_size: int = 100, Ns: int = 10000, seed: int = None, **kwargs): 
    mean, cov = mean_cov_from_gp(smiles=smiles, model=model, featurizer=featurizer, full_cov=True, gpu=gpu)
    p_yx = multivariate_normal(mean=mean, cov=cov, allow_singular=True, seed=seed)
    try: 
        samples = p_yx.rvs(size=Ns, random_state=seed)
    except: 
        print('Error sampling from multivariate, adding noise to diagonal')
        cov = cov + np.identity(len(mean))*1e-5
        p_yx = multivariate_normal(mean=mean, cov=cov, allow_singular=True, seed=seed)
        samples = p_yx.rvs(size=Ns, random_state=seed)
    
    top_samples = np.array([np.argmax(sample) if c==1 else np.argmin(samples) for sample in samples])
    probs = np.bincount(top_samples, minlength=len(mean))/Ns # [np.sum(top_k_samples==i)/N_samples for i in range(samples.shape[1])]
    acquisition_scores = {smi: (-1*prob, -1*c*mean) for smi, prob, mean in zip(smiles, probs, mean)} # for equal probs, use mean for sorting 
    sorted_smis = sorted(smiles, key=lambda smi: acquisition_scores[smi] )
    return sorted_smis[:batch_size]

def acquire_ts(smiles, model, featurizer, gpu, c: int = 1, batch_size: int = 100, seed: int = None, **kwargs): 
    mean, cov = mean_cov_from_gp(smiles=smiles, model=model, featurizer=featurizer, full_cov=True, gpu=gpu)
    p_yx = multivariate_normal(mean=mean, cov=cov, allow_singular=True, seed=seed)
    try: 
        samples = p_yx.rvs(size=batch_size, random_state=seed)
    except: 
        print('Error sampling from multivariate, adding noise to diagonal')
        cov = cov + np.identity(len(mean))*1e-5
        p_yx = multivariate_normal(mean=mean, cov=cov, allow_singular=True, seed=seed)
        samples = p_yx.rvs(size=batch_size, random_state=seed)

    selected_inds = []

    for sample in samples:
        for ind in np.argsort(c*sample)[::-1]:            
            if ind not in selected_inds: 
                selected_inds.append(ind)
                break 

    selected_smis = [smiles[i] for i in selected_inds]

    return selected_smis

def acquire_sequential_qei(smiles, model, featurizer, gpu, best_f, c: int = 1, batch_size: int = 100, seed: int = None, **kwargs):
    X_test = np.array([featurizer[smi] for smi in smiles])
    X_test = torch.as_tensor(X_test).cuda() if gpu else torch.as_tensor(X_test)    
    sampler = botorch.sampling.normal.SobolQMCNormalSampler(sample_shape=X_test[0].shape, seed=seed)
    if c == -1: 
        weights = torch.as_tensor([-1]).cuda() if gpu else torch.as_tensor([-1])
        objective = botorch.acquisition.objective.LinearMCObjective(weights)
        acq_function = botorch.acquisition.logei.qLogExpectedImprovement(model=model, best_f=c*best_f, sampler=sampler, objective=objective)
    else: 
        acq_function = botorch.acquisition.logei.qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler)

    selections, _ = botorch.optim.optimize.optimize_acqf_discrete(acq_function, q=batch_size, choices=X_test, max_batch_size=batch_size, unique=True)
    idx = np.where( (X_test.cpu()==selections.cpu()[:,None]).all(-1) )[1]
    idx = list(set(idx))[:batch_size]
    return [smiles[i] for i in idx] 