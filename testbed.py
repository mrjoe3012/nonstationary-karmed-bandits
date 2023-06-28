import random

class RewardValue:
    def __init__(self, mu, var):
        self._mean = mu
        self._variance = var
    
    def __call__(self):
        return random.normalvariate(self._mean, self._variance**0.5)

def generate_testbed(mu=0.0, var1=1.0, var2=1.0, k=10):
    """
    Generates a set of reward distributions for k actions.
    
    :param mu1: The mean with which to select true values.
    :param var1: The variance with which to select true values.
    :param var2: The variance with which to sample reward values.
    :param k: The number of reward distributions to return.
    :returns: A list of reward distributions which can be sampled from with operator ()
    """
    testbed = []
    for i in range(k):
        testbed.append(
            RewardValue(random.normalvariate(mu, var1**0.5), var2)
        )
    return testbed

def vary_reward(reward: RewardValue, mu=0.0, var=1e-2):
    """
    Shifts the mean of a testbed by a value sampled from
    the normal distribution N(mu, sqrt(var))
    
    :param reward: The reward distribution to vary.
    :param mu: Mean.
    :param var: Variance.
    """
    reward._mean += random.normalvariate(mu, var**0.5)