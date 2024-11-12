
import torch
import numpy as np 
from scipy.special import ndtri, ndtr

def genz_orthant_probability(mean: torch.Tensor, cov: torch.Tensor, N_samples: int = 10000):
    """ 
    Estimates orthant probability using MC approach defined by Genz (1992) 
    All a_i = 0
    All b_i = inf 
    """

    m = len(mean)

    C = torch.linalg.cholesky(cov)

    # initialize 
    intsum = 0 
    N = 0
    varsum = 0
    d = [0.5] # Normal CDF at 0 
    e = [1] # Normal CDF at inf 
    f = [e[0] - d[0]] 

    for _ in range(N_samples): 
        w = np.random.uniform(low=0.0, high=1.0, size=(m-1,))
        y = []
        for i in range(1, m): 
            y.append( ndtri( d[i-1] + w[i-1]*(e[i-1] - d[i-1]) ) )
            d.append( ndtr( (0-sum([C[i,j]*y[j] for j in range(i-1)]))/C[i,i] ) )
            e.append(1)
            f.append((e[i] - d[i])*f[i-1])
        intsum += f[-1]
        varsum += f[-1]**2 
        N += 1
    
    return intsum/N



