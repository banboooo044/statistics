import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
import scipy.integrate

def convergence(f,p = uniform()):
    N = 100000
    X = f(p.rvs(size=N))
    estinct = np.cumsum(X) / np.arrange(1,N+1) #近似値(サンプル数(1 ~ N)についてそれぞれ)
    esterr = np.sqrt(np.cumsum((X-estinct)**2)) / np.arange(1, N + 1)    
    return estinct,esterr