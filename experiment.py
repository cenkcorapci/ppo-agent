import argparse
import logging
import pathlib

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ai.ppo import PPO
from config import *
from data.memory import Memory

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", help="gym environment to be used", type=str, default='BipedalWalker-v2')
parser.add_argument("--render", help="render gym environment", type=bool, default=False)
parser.add_argument("--solved_reward", help="stop training if avg_reward > solved_reward", type=int, default=300)
parser.add_argument("--log_interval", help="print avg reward in the interval", type=int, default=20)
parser.add_argument("--max_episodes", help="max training episodes", type=int, default=10000)
parser.add_argument("--max_timesteps", help="max timesteps in one episode", type=int, default=1500)
parser.add_argument("--update_timestep", help="max timesteps in one episode", type=int, default=4000)
parser.add_argument("--action_std", help="constant std for action distribution (Multivariate Normal)", type=float,
                    default=.5)
parser.add_argument("--K_epochs", help="update policy for K epochs", type=int, default=80)
parser.add_argument("--eps_clip", help="clip parameter for PPO", type=float, default=.2)
parser.add_argument("--gamma", help="discount factor", type=float, default=.99)
parser.add_argument("--lr", help="learning rate", type=float, default=.0003)
parser.add_argument("--betas", type=tuple, default=(.9, .999))
parser.add_argument("--log_path", help="Tensorboard log path", type=str, default='tb_logs')
parser.add_argument("--model_path", help="Path for model persistence", type=str, default='models')

args = parser.parse_args()

parser.log_path = "{parser.log_path}/{EXPERIMENT_NAME}"

pathlib.Path(args.log_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(args.model_path).mkdir(parents=True, exist_ok=True)

summary_writer = SummaryWriter(log_dir=args.log_path)
logging.info(f"Tensorboard path: {args.log_path}")

# if gpu is to be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# creating environment
env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

if random_seed:
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)

memory = Memory()
ppo = PPO(device, state_dim, action_dim, args.action_std, args.lr, betas, args.gamma, args.K_epochs, args.eps_clip)
print(args.lr, betas)

# logging variables
running_reward = 0
avg_length = 0
time_step = 0

# training loop
for i_episode in tqdm(range(1, args.max_episodes + 1), desc='Training'):
    state = env.reset()

    for t in tqdm(range(args.max_timesteps), desc='Running environment'):
        time_step += 1
        # Running policy_old:
        action = ppo.select_action(state, memory)
        state, reward, done, _ = env.step(action)

        # Saving reward and is_terminals:
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        # update if its time
        if time_step % args.update_timestep == 0:
            ppo.update(memory)
            memory.clear_memory()
            time_step = 0
        running_reward += reward
        if args.render:
            env.render()
        if done:
            break

    avg_length += t

    # stop training if avg_reward > solved_reward
    if running_reward > (args.log_interval * args.solved_reward):
        print("########## Solved! ##########")
        torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(args.env_name))
        break

    # save every 500 episodes
    if i_episode % 50 == 0:
        torch.save(ppo.policy.state_dict(), '{}PPO_{}.pth'.format(args.MODEL_PATH, args.env_name))

    # logging
    summary_writer.add_scalar('avg episode length', avg_length, i_episode)
    summary_writer.add_scalar('reward', running_reward, i_episode)
    summary_writer.close()

    if i_episode % args.log_interval == 0:
        avg_length = int(avg_length / args.log_interval)
        running_reward = int((running_reward / args.log_interval))

        print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
        running_reward = 0
        avg_length = 0
