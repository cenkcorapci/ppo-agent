import pathlib
import time

############## Hyperparameters ##############
env_name = "BipedalWalker-v2"
render = False
solved_reward = 300  # stop training if avg_reward > solved_reward
log_interval = 20  # print avg reward in the interval
max_episodes = 10000  # max training episodes
max_timesteps = 1500  # max timesteps in one episode

update_timestep = 4000  # update policy every n timesteps
action_std = 0.5  # constant std for action distribution (Multivariate Normal)
K_epochs = 80  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor

lr = 0.0003  # parameters for Adam optimizer
betas = (0.9, 0.999)

random_seed = None
################## Other ######################

EXPERIMENT_NAME = f'ppo_{time.time()}'
TB_LOGS_PATH = f'/tmp/tb_logs/{EXPERIMENT_NAME}'
MODEL_PATH = '/home/cenk/research/drl/workout/saved_models/'

pathlib.Path(TB_LOGS_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
