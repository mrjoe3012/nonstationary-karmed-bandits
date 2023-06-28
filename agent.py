import random

class IncrementalAverage:
    def __init__(self):
        """
        Implements the incremental algorithm with a custom step-size.
        """
        self._value = 0.0
        self._n = 0

    def update(self, new_value: float, step_size=None):
        """
        Implements the incremental algorithm with a custom step-size.
        
        :param new_value: The new value to update the average with.
        :param step_size: The step size to use. Defaults to 1/n (sample mean)
        """
        self._n += 1
        if step_size is None:
            step_size = 1 / self._n
        self._value += step_size * (new_value - self._value)

    def get(self):
        return self._value

class Agent:
    def __init__(self, k=10, epsilon=0.0, step_size=None):
        """
        Implements an epsilon-greedy action-value estimate
        RL agent.
        
        :param k: The number of possible actions to choose from.
        :param epsilon: The probability for random exploration.
        :param step_size: The step size to use in incremental average calculations. Defaults to sample mean.
        """
        self._k = k
        self._reward_estimates = [IncrementalAverage() for i in range(k)]
        self._step_size = step_size
        self._epsilon = epsilon

    def next_action(self):
        """
        Decides upon an action using the epsilon-greedy approach
        and action-value estimates.
        
        :returns: The index representing the action chosen [0, k)
        """
        rewards = [x.get() for x in self._reward_estimates]
        exploration = self._determine_exploration(self._epsilon)
        if exploration: action = random.randint(0, self._k - 1)
        else: action = rewards.index(max(rewards))
        # print(f"Exploration: {exploration}, Action: {action}")
        return action

    def give_reward(self, action, reward):
        """
        Give the agent a reward value for a specific action.
        
        :param action: The action index [0, k)
        :param reward: The reward value. 
        """
        avg = self._reward_estimates[action]
        avg.update(reward, self._step_size)
        # print(f"Action {action} now has value {avg.get()}")

    def _determine_exploration(self, epsilon):
        x = random.uniform(0.0, 1.0)
        return x <= epsilon
