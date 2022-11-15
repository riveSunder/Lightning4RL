import argparse
import time
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

import lightning as pl

# rl imports
import gym
import procgen

def enjoy(agent, env, total_steps=100, render=True):

    agent.to("cpu")
    agent.eval()

    done = True
    rewards = []
    for step in range(total_steps): 
        if done:
            # reset environment after episode
            obs = env.reset()

            done = False

        if len(agent.input_dim) == 3:
            #height, width, channels
            h, w, c = agent.input_dim
            obs = torch.Tensor(obs.reshape(1, c, h, w))
        else:
            obs = torch.Tensor(obs.reshape(1, -1))

        try:
            action = agent.get_action(obs)
        except:
            import pdb; pdb.set_trace()

        obs, reward, done, info = env.step(action[0])

        if render:
            env.render()

        rewards.append(reward)

    sum_rewards = np.sum(rewards)
    print(f"total reward: {sum_rewards:.3f} in {step+1} steps:  "\
        f"{sum_rewards / (step+1)} reward per step")

