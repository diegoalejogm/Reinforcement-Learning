from IPython.display import clear_output as co
import time

def extract_actions(env):
    return list(range(env.nA))

def run_environment_greedy(env, Q):
    '''
    Executes environment using a greedy policy on the input action value.
    Environment is ran consecutively and interactively.
    
    Parameters
    ----------
    env : gym.core.Env
         OpenAI Gym Environment instance.
    Q : ActionValue
        Action Value Instance.
    '''
    state = env.reset()
    while(True):
        co()
        print('Current State: {}'.format(state))
        action = Q.argmax(state)
        env.render()
        print('Greedy action: {}'.format(action))
        state, reward, done, _ = env.step(action)
        try:
            input("Press Enter to continue...")
        except Exception as e:
            raw_input("Press Enter to continue...")
        if done:
            break
    clear_output()
    print('Current State: {}'.format(state))
    env.render()
    print('Finished!')

def display_episode_log(episode, num_episodes):
    '''
    Displays the current execution status, given a total number of episodes
    and the episode number.
    
    Parameters
    ----------
    episode : int
        Current episode
    num_episodes : int
        Total number of episodes
    '''
    clear_output()
    print('Episode: {}/{}'.format(episode, num_episodes))

def clear_output():
    co(wait=True)
