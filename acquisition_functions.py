from scipy.stats import multivariate_normal
from scipy.special import ndtri, ndtr

from gp import TanimotoGP
import torch
import botorch 
import numpy as np 
import random
import copy 

from utils import simple_parallel, tanimoto_matrix

def acquire(method, smiles, model, featurizer, gpu: bool = True, c: int = 1, batch_size: int = 100, best_f: float = None, N_samples: int = 10000, **kwargs):
    """ Calls appropriate acquisition function """
    
    acq_functions = {
        'Greedy': acquire_mean, 
        'UCB': acquire_ucb, 
        'qPO': acquire_qPO, # qPO
        'pTS': acquire_ts,
        'qEI': acquire_sequential_qei,
        'qPI': acquire_sequential_qpi,
        'random_10k': acquire_random, 
        'random': acquire_random,
        'GIBBON': acquire_GIBBON,
        'qPO_orthant': acquire_qPO_orthant,
        'TS_RSR': acquire_TSRSR,
        'DPPTS': acquire_DPPTS,
        'BUCB': acquire_batch_ucb,
    }
    if method in {'qPO', 'pTS', 'qEI', 'random_10k', 'GIBBON', 'qPO_orthant', 'TS_RSR', 'DPPTS', 'qPI', 'BUCB'} and len(smiles) > 10000: 
        # get top 10k by mean 
        smiles_filtered = acq_functions['UCB'](
            smiles=smiles, 
            model=model, 
            featurizer=featurizer, 
            gpu=gpu, c=c, batch_size=10000, **kwargs
        )
        # apply acquisition strategy to remaining 10k candidates 
        return acq_functions[method](
            smiles=smiles_filtered, 
            model=model, N_samples=N_samples,
            featurizer=featurizer, 
            gpu=gpu, c=c, batch_size=batch_size, 
            best_f=best_f, **kwargs
        )
    
    return acq_functions[method](
        smiles=smiles, 
        model=model, N_samples=N_samples,
        featurizer=featurizer, 
        gpu=gpu, c=c, batch_size=batch_size, 
        best_f=best_f, **kwargs
    )

def mean_cov_from_gp(model: TanimotoGP, smiles: list, featurizer: dict, full_cov: bool = True, gpu: bool = True): 
    """ Returns the mean and covariance (or variance) of the surrogate model posterior """
    
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
    """ Greedy acquisition function """

    mean, _ = mean_cov_from_gp(smiles=smiles, model=model, featurizer=featurizer, full_cov=False, gpu=gpu)
    acquisition_scores = {smi: score for smi, score in zip(smiles, c*mean)}
    sorted_smis = sorted(smiles, key=lambda smi: -1*acquisition_scores[smi])
    return sorted_smis[:batch_size]

def acquire_ucb(smiles, model, featurizer, gpu, c: int = 1, batch_size: int = 100, beta: float = 1, **kwargs): 
    """ Upper confidence bound acquisition function """

    mean, var = mean_cov_from_gp(smiles=smiles, model=model, featurizer=featurizer, full_cov=False, gpu=gpu)
    acquisition_scores = {smi: score for smi, score in zip(smiles, c*mean + beta*np.sqrt(var))}
    sorted_smis = sorted(smiles, key=lambda smi: -1*acquisition_scores[smi])
    return sorted_smis[:batch_size]

def acquire_qPO(smiles, model, featurizer, gpu, c: int = 1, batch_size: int = 100, N_samples: int = 10000, seed: int = None, **kwargs): 
    """ The proposed acquisition function -- qPO (multipoint probability of optimality) """
    
    mean, cov = mean_cov_from_gp(smiles=smiles, model=model, featurizer=featurizer, full_cov=True, gpu=gpu)
    p_yx = multivariate_normal(mean=mean, cov=cov, allow_singular=True, seed=seed)
    try: 
        samples = p_yx.rvs(size=N_samples, random_state=seed)
    except: 
        count = 0
        sampled = False 
        while count < 10 and not sampled: 
            print('Error sampling from multivariate, adding noise to diagonal')
            try: 
                cov = cov + np.identity(len(mean))*1e-8
                p_yx = multivariate_normal(mean=mean, cov=cov, allow_singular=True, seed=seed)
                samples = p_yx.rvs(size=N_samples, random_state=seed)
                sampled = True 
            except: 
                continue
    
    top_samples = np.array([np.argmax(c*sample) for sample in samples])
    probs = np.bincount(top_samples, minlength=len(mean))/N_samples # [np.sum(top_k_samples==i)/N_samples for i in range(samples.shape[1])]
    acquisition_scores = {smi: (-1*prob, -1*c*mean) for smi, prob, mean in zip(smiles, probs, mean)} # for equal probs, use mean for sorting 
    sorted_smis = sorted(smiles, key=lambda smi: acquisition_scores[smi] )
    return sorted_smis[:batch_size]

def acquire_ts(smiles, model, featurizer, gpu, c: int = 1, batch_size: int = 100, seed: int = None, **kwargs): 
    """ Acquisition with parallel Thomspon sampling """

    mean, cov = mean_cov_from_gp(smiles=smiles, model=model, featurizer=featurizer, full_cov=True, gpu=gpu)
    p_yx = multivariate_normal(mean=mean, cov=cov, allow_singular=True, seed=seed)
    try: 
        samples = p_yx.rvs(size=batch_size, random_state=seed)
    except: 
        count = 0
        sampled = False 
        while count < 10 and not sampled:
            try:
                print('Error sampling from multivariate, adding noise to diagonal')
                cov = cov + np.identity(len(mean))*1e-8
                p_yx = multivariate_normal(mean=mean, cov=cov, allow_singular=True, seed=seed)
                samples = p_yx.rvs(size=batch_size, random_state=seed)
                sampled = True 
            except: 
                continue

    selected_inds = []

    for sample in samples:
        for ind in np.argsort(c*sample)[::-1]:            
            if ind not in selected_inds: 
                selected_inds.append(ind)
                break 

    selected_smis = [smiles[i] for i in selected_inds]

    return selected_smis

def acquire_sequential_qei(smiles, model, featurizer, gpu, best_f, c: int = 1, batch_size: int = 100, seed: int = None, **kwargs):
    """Acquisition with multipoint expected improvement"""

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

def acquire_GIBBON(smiles, model, featurizer, gpu, c: int = 1, batch_size: int = 100, **kwargs): 
    X_test = np.array([featurizer[smi] for smi in smiles])
    dim = X_test.shape[1]
    bounds = torch.as_tensor(
        np.array([np.zeros(shape=(dim,)), np.max(X_test, axis=0)]), device = 'cuda' if gpu else 'cpu' 
    )

    X_test = torch.as_tensor(X_test).cuda() if gpu else torch.as_tensor(X_test)  
    qGIBBON = botorch.acquisition.max_value_entropy_search.qLowerBoundMaxValueEntropy(
        model=model, candidate_set=X_test,
        maximize=False if c==-1 else True
    )
    
    selections, _ = botorch.optim.optimize_acqf_discrete(
        acq_function=qGIBBON,
        q=batch_size,
        choices=X_test,
        max_batch_size=batch_size,
        unique=True,
    )

    idx = np.where( (X_test.cpu()==selections.cpu()[:,None]).all(-1) )[1]
    idx = list(set(idx))[:batch_size]
    return [smiles[i] for i in idx] 

def qPO_acqscore_orthant(mean: torch.Tensor, cov: torch.Tensor, device, i, N_samples, c): 
    # construct A 
    k = len(mean)
    A = np.zeros(shape=(k-1,k))
    for j in range(k-1): 
        A[j,i] = 1
    for p in range(i): 
        A[p,p] = -1
    for p in range(i+1, k): 
        A[p-1,p] = -1
    
    # A = torch.as_tensor(A, device=device).float()
    A = np.array(A)
    mean_diff = A.dot(mean) # torch.matmul(A, mean)
    cov_diff = A.dot(cov).dot(A.T) # torch.matmul(torch.matmul(A, cov), A.T)

    m = len(mean_diff)

    if device == 'cuda': 
        C = np.linalg.cholesky(cov_diff) # .cpu().detach().numpy()
        a = -1*c*mean_diff # .cpu().detach().numpy()
    else: 
        C = torch.linalg.cholesky(cov_diff).detach().numpy()
        a = -1*c*mean_diff.detach().numpy()

    # initialize 
    intsum = 0 
    varsum = 0
    d = [ndtr(a[0]/C[0,0])]
    e = [1] # Normal CDF at inf 
    f = [e[0] - d[0]] 

    for _ in range(N_samples): 
        w = np.random.uniform(low=0.0, high=1.0, size=(m-1,))
        y = []
        d = [ndtr(a[0]/C[0,0])]
        e = [1] # Normal CDF at inf 
        f = [e[0] - d[0]] 
        for i in range(1, m): 
            y.append( ndtri( d[i-1] + w[i-1]*(e[i-1] - d[i-1]) ) )
            d.append( ndtr( (a[i]-sum([Cij*yj for Cij, yj in zip(C[i,:i],y)]))/C[i,i] ) )
            e.append(1)
            f.append((e[i] - d[i])*f[i-1])
        intsum += f[-1]
        varsum += f[-1]**2 
    
    return intsum/N_samples    

def acquire_qPO_orthant(smiles, model, featurizer, gpu, c: int = 1, batch_size: int = 100, N_samples: int = 10000, seed: int = None, **kwargs): 
    mean, cov = mean_cov_from_gp(smiles=smiles, model=model, featurizer=featurizer, full_cov=True, gpu=gpu)
    device = 'cuda' if gpu else 'cpu'
    # mean = torch.as_tensor(mean, device=device).float()
    # cov = torch.as_tensor(cov, device=device).float()
    fn = lambda i: qPO_acqscore_orthant(mean=mean, cov=cov, i=i, device=device, N_samples=N_samples, c=c)
    probs = simple_parallel(input_list=list(range(len(smiles))), function=fn, max_cpu=64)
    acquisition_scores = {smi: (-1*prob, -1*c*mean) for smi, prob, mean in zip(smiles, probs, mean)} # for equal probs, use mean for sorting 
    sorted_smis = sorted(smiles, key=lambda smi: acquisition_scores[smi] )
    return sorted_smis[:batch_size]

def acquire_TSRSR(smiles, model, featurizer, gpu, c: int = 1, batch_size: int = 100, seed: int = None, **kwargs): 
    """ Acquisition with parallel Thomspon sampling """

    mean, cov = mean_cov_from_gp(smiles=smiles, model=model, featurizer=featurizer, full_cov=True, gpu=gpu)
    std = np.sqrt(np.diag(cov))
    p_yx = multivariate_normal(mean=mean, cov=cov, allow_singular=True, seed=seed)
    try: 
        samples = p_yx.rvs(size=batch_size, random_state=seed)
    except: 
        count = 0
        sampled = False 
        while count < 10 and not sampled:
            try:
                print('Error sampling from multivariate, adding noise to diagonal')
                cov = cov + np.identity(len(mean))*1e-8
                p_yx = multivariate_normal(mean=mean, cov=cov, allow_singular=True, seed=seed)
                samples = p_yx.rvs(size=batch_size, random_state=seed)
                sampled = True 
            except: 
                continue

    selected_inds = []

    for sample in samples:
        f_star = np.max(sample) if c==1 else np.min(sample)
        rsr = c*(f_star - mean)/std
        for ind in np.argsort(rsr):            
            if ind not in selected_inds: 
                selected_inds.append(ind)
                break 

    selected_smis = [smiles[i] for i in selected_inds]

    return selected_smis

def acquire_DPPTS(smiles, model, featurizer, gpu, c: int = 1, batch_size: int = 100, N_samples: int = 10000, seed: int = None, N_iter: int = 1000, **kwargs): 

    selections = random.sample(range(len(smiles)), batch_size)
    mean, cov = mean_cov_from_gp(smiles=smiles, model=model, featurizer=featurizer, full_cov=True, gpu=gpu)
    p_yx = multivariate_normal(mean=mean, cov=cov, allow_singular=True, seed=seed)

    # obtain N_samples posterior samples 
    samples = p_yx.rvs(random_state=seed, size=N_samples) 

    X = np.array([featurizer[smiles[i]] for i in selections])
    for _ in range(N_iter): 
        # uniformly pick point to replace 
        replace_ind = random.sample(selections, 1)

        # sample from posterior until optimum is not in acquired set with max_samples = 100
        ns = 0
        sample_obtained = False
        while ns < 1000: 
            ns += 1
            sample = samples[random.sample(range(samples.shape[0]), 1),:]
            if np.argmax(sample) not in selections: 
                sample_obtained = True
                new_ind = np.argmax(sample)
                break 
        
        if not sample_obtained: 
            continue 
    
        new_selections = [ind for ind in selections if ind != replace_ind]
        new_selections.append(new_ind)

        new_X = np.array([featurizer[smiles[i]] for i in new_selections])
        p_accept = np.linalg.det(tanimoto_matrix(new_X))/np.linalg.det(tanimoto_matrix(X))
        if p_accept > 1: 
            accept = True 
        else: 
            r = np.random.uniform()
            accept = True if r < p_accept else False 
        
        if accept: 
            X = copy.deepcopy(new_X)
            selections = copy.deepcopy(new_selections) 
 
    return [smiles[i] for i in selections]

def acquire_sequential_qpi(smiles, model, featurizer, gpu, best_f, c: int = 1, batch_size: int = 100, seed: int = None, **kwargs):
    """Acquisition with multipoint probability of improvement"""

    X_test = np.array([featurizer[smi] for smi in smiles])
    X_test = torch.as_tensor(X_test).cuda() if gpu else torch.as_tensor(X_test)    
    sampler = botorch.sampling.normal.SobolQMCNormalSampler(sample_shape=X_test[0].shape, seed=seed)
    if c == -1: 
        weights = torch.as_tensor([-1]).cuda() if gpu else torch.as_tensor([-1])
        objective = botorch.acquisition.objective.LinearMCObjective(weights)
        acq_function = botorch.acquisition.monte_carlo.qProbabilityOfImprovement(model=model, best_f=c*best_f, sampler=sampler, objective=objective)
    else: 
        acq_function = botorch.acquisition.monte_carlo.qProbabilityOfImprovement(model=model, best_f=best_f, sampler=sampler)

    selections, _ = botorch.optim.optimize.optimize_acqf_discrete(acq_function, q=batch_size, choices=X_test, max_batch_size=batch_size, unique=True)
    idx = np.where( (X_test.cpu()==selections.cpu()[:,None]).all(-1) )[1]
    idx = list(set(idx))[:batch_size]
    return [smiles[i] for i in idx] 

def acquire_batch_ucb(smiles, model, featurizer, gpu, best_f, c: int = 1, batch_size: int = 100, seed: int = None, **kwargs):
    """Acquisition with multipoint upper confidence bound, and uses beta = sqrt(3) following Wilson et al 2017. """

    X_test = np.array([featurizer[smi] for smi in smiles])
    X_test = torch.as_tensor(X_test).cuda() if gpu else torch.as_tensor(X_test)    
    sampler = botorch.sampling.normal.SobolQMCNormalSampler(sample_shape=X_test[0].shape, seed=seed)
    if c == -1: 
        weights = torch.as_tensor([-1]).cuda() if gpu else torch.as_tensor([-1])
        objective = botorch.acquisition.objective.LinearMCObjective(weights)
        acq_function = botorch.acquisition.monte_carlo.qUpperConfidenceBound(model=model, beta=np.sqrt(3), sampler=sampler, objective=objective)
    else: 
        acq_function = botorch.acquisition.monte_carlo.qUpperConfidenceBound(model=model, beta=np.sqrt(3), sampler=sampler)

    selections, _ = botorch.optim.optimize.optimize_acqf_discrete(acq_function, q=batch_size, choices=X_test, max_batch_size=batch_size, unique=True)
    idx = np.where( (X_test.cpu()==selections.cpu()[:,None]).all(-1) )[1]
    idx = list(set(idx))[:batch_size]
    return [smiles[i] for i in idx] 

def acquire_random(smiles, batch_size: int = 100, **kwargs): 
    """ Random acquistion """
    return random.sample(smiles, batch_size)