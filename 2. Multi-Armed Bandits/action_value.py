
def sample_average(Q, last_action, last_reward, N, t):
    Q[last_action] += (last_reward - Q[last_action]) / N[last_action]

# Make sure to pass store_rewards=True to agent.
# TODO: Maybe use classes for action values instead of functions.


def exponential_regency_weighted_average(Q, last_action, last_reward, Q_init, R, alpha, **kwargs):
    assert R, "Empty|Null Rewards Array."
    n = len(R[last_action])
    Q[last_action] = pow(1 - alpha, n) * Q_init[last_action]
    for i in range(1, n):
        Q[last_action] += alpha * pow(1 - alpha, n - i) * R[last_action][i]
