import numpy as np

def ackley(x, a=20, b=0.2, c=2*np.pi):
    x = np.asarray(x)
    n = x.size
    
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    
    return (
        -a * np.exp(-b * np.sqrt(sum_sq / n))
        - np.exp(sum_cos / n)
        + a + np.e
    )
