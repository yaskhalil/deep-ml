import numpy as np
def exp_weighted_average(Q1, rewards, alpha):
    """
    Q1: float, initial estimate
    rewards: list or array of rewards, R_1 to R_k
    alpha: float, step size (0 < alpha <= 1)
    Returns: float, exponentially weighted average after k rewards
    """
    one_a = (1-alpha)
    rewards = np.array(rewards)
    k = len(rewards)
    i = np.arange(len(rewards)) 
    return ((one_a**k)*Q1) + np.sum(alpha*(one_a**(k-i-1)) * rewards[i])
    
