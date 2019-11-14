import logging

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ai.ppo import PPO
from config import *
from data.memory import Memory

summary_writer = SummaryWriter(log_dir=TB_LOGS_PATH)
logging.info('Tensorboard path: {}'.format(TB_LOGS_PATH))

# if gpu is to be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# creating environment
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

if random_seed:
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)

memory = Memory()
ppo = PPO(device, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
print(lr, betas)

# logging variables
running_reward = 0
avg_length = 0
time_step = 0

# training loop
for i_episode in tqdm(range(1, max_episodes + 1), desc='Training'):
    state = env.reset()

    for t in tqdm(range(max_timesteps), desc='Running environment'):
        time_step += 1
        # Running policy_old:
        action = ppo.select_action(state, memory)
        state, reward, done, _ = env.step(action)

        # Saving reward and is_terminals:
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        # update if its time
        if time_step % update_timestep == 0:
            ppo.update(memory)
            memory.clear_memory()
            time_step = 0
        running_reward += reward
        if render:
            env.render()
        if done:
            break

    avg_length += t

    # stop training if avg_reward > solved_reward
    if running_reward > (log_interval * solved_reward):
        print("########## Solved! ##########")
        torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
        break

    # save every 500 episodes
    if i_episode % 50 == 0:
        torch.save(ppo.policy.state_dict(), '{}PPO_{}.pth'.format(MODEL_PATH, env_name))

    # logging
    summary_writer.add_scalar('avg episode length', avg_length, i_episode)
    summary_writer.add_scalar('reward', running_reward, i_episode)
    summary_writer.close()

    if i_episode % log_interval == 0:
        avg_length = int(avg_length / log_interval)
        running_reward = int((running_reward / log_interval))

        print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
        running_reward = 0
        avg_length = 0
