# PPO-Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![GitHub last commit](https://img.shields.io/github/last-commit/cenkcorapci/ppo-agent)
[![Requirements Status](https://requires.io/github/cenkcorapci/ppo-agent/requirements.svg?branch=master)](https://requires.io/github/cenkcorapci/ppo-agent/requirements/?branch=master)

A [Proximal Policy Optimization ](https://arxiv.org/abs/1707.06347)
 implementation with [PyTorch](https://pytorch.org).
 
## Usage
To train on [BipedalWalker-v2](https://gym.openai.com/envs/BipedalWalker-v2/) with default parameters use;
```bash
python experiment.py
```
Here are the optional parameters;

|parameter name| description|type|default|
|----|---|---|----|
|--env_name|gym environment to be used|str|BipedalWalker-v2|
|--render|render gym environment|bool|False|
|--solved_reward|stop training if avg_reward > solved_reward|int|300|
|--log_interval|print avg reward in the interval|int|20|
|--max_episodes|max training episodes|int|10000|
|--max_timesteps|max timesteps in one episode|int|1500|
|--update_timestep|max timesteps in one episode|int|4000|
|--action_std|constant std for action distribution (Multivariate Normal)|float|default=0.5|
|--K_epochs|update policy for K epochs|int|80|
|--eps_clip|clip parameter for PPO|float|0.2|
|--gamma|discount factor|float|0.99|
|--lr|learning rate|float|0003|
|--log_path|Tensorboard log path|str|tb_logs|
|--model_path|Path for model persistence | str| models|