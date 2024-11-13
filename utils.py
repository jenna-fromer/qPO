
import torch
import numpy as np 
from scipy.special import ndtri, ndtr
from scipy.stats import multivariate_normal
from tqdm import tqdm
from pathos import multiprocessing as mp

import multiprocess.context as ctx
ctx._force_start_method('spawn')

def simple_parallel(input_list, function, max_cpu=16,
                    timeout=4000, max_retries=3, desc=None):
    """ Simple parallelization.

    Use map async and retries in case we get odd stalling behavior.

    input_list: Input list to op on
    function: Fn to apply
    max_cpu: Num cpus
    timeout: Length of timeout
    max_retries: Num times to retry this

    """
    from multiprocess.context import TimeoutError

    cpus = min(mp.cpu_count(), max_cpu)
    pool = mp.Pool(processes=cpus)
    async_results = [pool.apply_async(function, args=(i, ))
                     for i in input_list]
    pool.close()

    retries = 0
    while True:
        try:
            list_outputs = []
            for async_result in tqdm(async_results, total=len(input_list), desc=desc):
                result = async_result.get(timeout)
                list_outputs.append(result)

            break
        except TimeoutError:
            retries += 1
            print(f"Timeout Error (s > {timeout})")
            if retries <= max_retries:
                pool = mp.Pool(processes=cpus)
                async_results = [pool.apply_async(function, args=(i, ))
                                 for i in input_list]
                pool.close()
                print(f"Retry attempt: {retries}")
            else:
                raise ValueError()

    return list_outputs


def construct_A(k: int, i: int, device: str = None) -> torch.Tensor: 
    """ constructs k-1 x k matrix A that defines difference vector for candidate i """
    A = np.zeros(shape=(k-1,k))
    for j in range(k-1): 
        A[j,i] = 1
    for p in range(i): 
        A[p,p] = -1
    for p in range(i+1, k): 
        A[p-1,p] = -1
    return torch.as_tensor(A, device=device)

def genz_orthant_probability(cov: torch.Tensor, a: torch.Tensor, N_samples: int = 10000):
    """ 
    Estimates orthant probability using MC approach defined by Genz (1992) 
    All b_i = inf 
    """

    m = len(a)

    C = torch.linalg.cholesky(cov).detach().numpy()
    a = a.detach().numpy()

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

def tanimoto_similarity(x1, x2):
    """
    Calculates the Tanimoto similarity between two numpy arrays.
    """
    dot_prod = x1.dot(x2)
    x1_sum = np.sum(x1 ** 2)
    x2_sum = np.sum(x2 ** 2)
    return (dot_prod) / (x1_sum + x2_sum - dot_prod)

def tanimoto_matrix(X):
    """
    Calculates the Tanimoto similarity matrix for a given numpy array.
    """
    n = X.shape[0]
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            similarity_matrix[i, j] = tanimoto_similarity(X[i], X[j])
            similarity_matrix[j, i] = similarity_matrix[i, j]

    return similarity_matrix

def test():
    device = 'cpu'
    mean = torch.as_tensor([10, 5, 0], device=device).float()
    cov = torch.as_tensor([[101, 100, 0], [100, 101, 0], [0, 0, 1]], device=device).float()

    print('----------- DIRECT SAMPLING ------------')
    p_yx = multivariate_normal(mean=mean, cov=cov)
    M = 1e5
    samples = p_yx.rvs(size=int(M))
    top_samples = np.array([np.argmax(sample) for sample in samples])
    probs = np.bincount(top_samples, minlength=len(mean))/M
    print('\n'.join([f'P{i}: {p}' for i, p in enumerate(probs)]))
    print('----------------------------------------\n')


    print('----------- Genz 1992 for orthants ------------')
    probs = []
    for i in range(3):
        A_diff = construct_A(k=len(mean), i=i, device=device).float()
        mean_diff = torch.matmul(A_diff, mean)
        cov_diff = torch.matmul(torch.matmul(A_diff, cov), A_diff.T) #dot(cov).dot(A_diff.T)
        probs.append(genz_orthant_probability(cov=cov_diff, a=-1*mean_diff))

    print('\n'.join([f'P{i}: {p}' for i, p in enumerate(probs)]))
    print('----------------------------------------------------------------\n')