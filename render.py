import gymnasium as gym
import numpy as np
from collections import deque
import torch
import wandb
import argparse
from buffer import ReplayBuffer
import glob
import random

import yaml
import os
from matplotlib import animation
import matplotlib.pyplot as plt

from utils import *
from networks import *

def render(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ckpt = config["model_ckpt"]
    assert model_ckpt != ""
    model_name = model_ckpt.split(".")[1].split("/")[-2]

    if config["env"] == "Pendulum-v1":
        action_range = [-2, 2]
        action_scale = 2.
    else:
        action_scale = 1.

    env = gym.make(config["env"], render_mode="rgb_array")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    q1, q2 = QNet(n_observations, n_actions, config), QNet(n_observations, n_actions, config)
    q1_target, q2_target = QNet(n_observations, n_actions, config), QNet(n_observations, n_actions, config)
    pi = PolicyNet(n_observations, n_actions, config)

    checkpoint = torch.load(model_ckpt, map_location=lambda storage, loc: storage)
    pi.load_state_dict(checkpoint["pi"])
    q1.load_state_dict(checkpoint["q1"])
    q2.load_state_dict(checkpoint["q2"])
    q1_target.load_state_dict(checkpoint["q1_target"])
    q2_target.load_state_dict(checkpoint["q2_target"])
    print("Model: {} Loaded!".format(model_ckpt))

    env = gym.wrappers.RecordVideo(env, "animation/{}/{}".format(config["env"],model_name),\
                                     episode_trigger=lambda episode_id: episode_id%1==0)
    with torch.no_grad():
        for x in range(10):
            observation, info = env.reset()
            truncated = False
            terminated = False
            while not (truncated or terminated):
                # 일정 Step 이상 되었을 떄 done이 되지 않으면, gif render가 되지 않는다.
                action, _ = pi.forward(torch.tensor(observation).float())
                action = action.detach().cpu().numpy()
                observation, reward, terminated, truncated, info = env.step(action_scale*action)
                env.render()

        env.close()


if __name__ == "__main__":
    config = get_config()
    render(config)
