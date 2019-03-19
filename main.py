import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent
import torch


def dqn(
    env,
    n_episodes=10000,
    max_t=1000,
    eps_start=1.0,
    eps_end=0.005,
    eps_decay=0.995,
    train_mode=True,
):
    """Deep Q-Learning.
    
    Params
    ======
        agent: 
        env: 
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        train_mode (bool): set environment into training mode if True. 
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    env_info = env.reset(train_mode=train_mode)[brain_name]
    state_size = len(env_info.vector_observations[0])

    agent = Agent(state_size=state_size, action_size=action_size, seed=1)

    for i_episode in range(1, n_episodes + 1):
        state = env_info.vector_observations[0]
        score = 0
        for _ in range(max_t):
            action = np.int32(agent.act(state, eps))
            env_info = env.step(action)[
                brain_name
            ]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                env.reset(train_mode=train_mode)[brain_name]
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print(
            "\rEpisode {}\tAverage Score: {:.2f}".format(
                i_episode, np.mean(scores_window)
            ),
            end="",
        )
        if i_episode % 100 == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_window)
                )
            )
        if np.mean(scores_window) >= 13.0:
            print(
                "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                    i_episode - 100, np.mean(scores_window)
                )
            )
            torch.save(agent.qnetwork_local.state_dict(), "checkpoint_vanilla.pth")
            break
    return scores


if __name__ == "__main__":
    os.environ["NO_PROXY"] = "localhost,127.0.0.*"

    from datetime import datetime

    start = datetime.now()
    scores = dqn()
    end = datetime.now()
    print("Duration: {}".format(end - start))

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.show()

