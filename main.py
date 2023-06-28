from testbed import generate_testbed, vary_reward
from agent import Agent, IncrementalAverage
import matplotlib.pyplot as plt
import random

iterations = 1000000
k = 10
eps = 0.1
alpha = 0.1

testbed = generate_testbed()

avg_reward1, avg_reward2 = IncrementalAverage(), IncrementalAverage() # avg rewards
avg_optimal1, avg_optimal2 = IncrementalAverage(), IncrementalAverage() # percentage of average optimal decisions
max_avg_reward = IncrementalAverage() # keep track of optimal average reward
sample_mean = Agent(k=k, epsilon=eps, step_size=None) # uses the sample mean as its action-value estimator
fixed_step = Agent(k=k, epsilon=eps, step_size=alpha) # uses a weighted average as its action-value estimator
random.seed(0)

# data to plot at end
data = {
    "sample_mean" : {
        "rewards" : [],
        "optimal" : [],
    },
    "fixed_step" : {
        "rewards" : [],
        "optimal" : [],
    },
    "max_avg_reward" : []
}

for i in range(iterations):
    # determine the optimal action at this timestep
    true_values = [x._mean for x in testbed]
    optimal_action = true_values.index(max(true_values))
    max_avg_reward.update(testbed[optimal_action]._mean)
    data["max_avg_reward"].append(max_avg_reward.get())

    # run the first agent
    action = sample_mean.next_action()
    reward = testbed[action]()
    sample_mean.give_reward(action, reward)
    avg_reward1.update(reward)
    data["sample_mean"]["rewards"].append(avg_reward1.get())
    avg_optimal1.update(1.0 if action == optimal_action else 0.0)
    data["sample_mean"]["optimal"].append(avg_optimal1.get()*100)

    # run the second agent
    action = fixed_step.next_action()
    reward = testbed[action]()
    fixed_step.give_reward(action, reward)
    avg_reward2.update(reward)
    data["fixed_step"]["rewards"].append(avg_reward2.get())
    avg_optimal2.update(1.0 if action == optimal_action else 0.0)
    data["fixed_step"]["optimal"].append(avg_optimal2.get()*100)

    # vary testbed
    for r in testbed: vary_reward(r)

# plots
plt.plot(list(range(iterations)), data["sample_mean"]["rewards"], "r-", label="sample mean")
plt.plot(list(range(iterations)), data["fixed_step"]["rewards"], "b-", label="fixed step (0.1)")
plt.plot(list(range(iterations)), data["max_avg_reward"], "-", color="black", label="optimal average reward")
plt.title("Average Reward")
plt.legend()
# plt.show()
plt.savefig(fname="average_reward")
plt.cla()

plt.title("Average Optimal Action")
plt.plot(list(range(iterations)), data["sample_mean"]["optimal"], "r-", label="sample mean")
plt.plot(list(range(iterations)), data["fixed_step"]["optimal"], "b-", label="fixed step (0.1)")
plt.legend()
# plt.show()
plt.savefig(fname="average_optimal_action")
plt.cla()
