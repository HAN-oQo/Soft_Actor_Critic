import gymnasium as gym
import numpy as np
import collections, random
from datetime import datetime
import os
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from buffer import ReplayBuffer
from networks import *
from utils import *

#Hyperparameters
# lr_pi           = 0.001 #05
# lr_q            = 0.001
# init_alpha      = 0.01 #0.01
# gamma           = 0.98 #0.99
# batch_size      = 32 #256 #32 #256 #32
# buffer_limit    = 50000 #100000
# tau             = 0.005 # for target network soft update
# target_entropy  = -1.0 # for automated alpha update
# lr_alpha        = 0.005 #0.001#0.001 #0.001  # for automated alpha update


def calc_target(pi, q1, q2, mini_batch, gamma):
    s, a, r, s_prime, done = mini_batch
    with torch.no_grad():
        a_prime, log_prob= pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob
        entropy = entropy.sum(dim=-1, keepdim = True)
        q1_val, q2_val = q1(s_prime,a_prime), q2(s_prime,a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = r + gamma * done * (min_q + entropy)

    return target
    
def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the current date and time
    now = datetime.now()

    # Format the date and time as an integer in the format of 20230753
    formatted = now.strftime("%Y%m%d%H%M%S")
    model_save_dir = os.path.join(config["model_basedir"], config["env"], config["run_name"], formatted)
    os.makedirs(model_save_dir, exist_ok = True)

    if config["env"] == "Pendulum-v1":
        action_range = [-2, 2]
        action_scale = 2.
    else:
        action_scale = 1.

    env = gym.make(config["env"])
    env = gym.wrappers.TimeLimit(env, max_episode_steps = config["max_episode_steps"])

    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    memory = ReplayBuffer(max_size = config["buffer_size"], input_shape= n_observations, n_actions=n_actions)

    q1, q2 = QNet(n_observations, n_actions, config), QNet(n_observations, n_actions, config)
    q1_target, q2_target = QNet(n_observations, n_actions, config), QNet(n_observations, n_actions, config)
    pi = PolicyNet(n_observations, n_actions, config)

    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    q1 = q1.to(device)
    q2 = q2.to(device)
    q1_target = q1_target.to(device)
    q2_target = q2_target.to(device)
    pi = pi.to(device)
    
    score_history = []
    update_steps = 0
    best_score = env.reward_range[0]
    pi_loss, alpha_loss, q1_loss, q2_loss, average10 = 0., 0., 0., 0., 0.
    with wandb.init(project="SAC_{}".format(config["env"]), name="{}_{}".format(now, config["run_name"]), config=config):
        for n_epi in range(config["n_episodes"]):
            s, _ = env.reset()
            done = False
            trunc = False
            score = 0.0
            # print(f"episode{n_epi} start")
            while not (done or trunc):
                # print("step ++")
                a, log_prob= pi(torch.tensor(s).float().to(device))
                a = a.detach().cpu().numpy()
                s_prime, r, done, trunc, info = env.step(action_scale*a)
                memory.put(s, a, r, s_prime, done) #r/10.0
                score +=r
                update_steps += 1
                s = s_prime
                    
                if memory.size()>config["start_size"]:
                    # for i in range(20):
                    mini_batch = memory.sample(config["batch_size"])
                    
                    mini_batch = ReplayBuffer.batch_to_device(mini_batch, device)
                    
                    td_target = calc_target(pi, q1_target, q2_target, mini_batch, config["discount"])
                    q1_loss = q1.train_net(td_target, mini_batch)
                    q2_loss = q2.train_net(td_target, mini_batch)
                    pi_loss, alpha_loss = pi.train_net(q1, q2, mini_batch)
                    q1.soft_update(q1_target)
                    q2.soft_update(q2_target)
            
            score_history.append(score)
            if n_epi > 10:
                average10 = np.mean(score_history[-10:])

            if n_epi > 100:
                if  average10 > best_score:
                    best_score = average10
                    torch.save({
                        "pi": pi.state_dict(),
                        "q1": q1.state_dict(),
                        "q2": q2.state_dict(),
                        "q1_target": q1_target.state_dict(),
                        "q2_target": q2_target.state_dict(),
                    }, os.path.join(model_save_dir, "best_score.ckpt"))
                    wandb.save(os.path.join(model_save_dir, "best_score.ckpt"))
                    
            if n_epi%config["log_every"]==0 and n_epi > 0:
                print("# of episode :{}, score1: {:.1f}, score10 : {:.1f}, alpha:{:.4f}, buffer_size: {}".format(n_epi, score, average10, pi.log_alpha.exp(), memory.size()))
                wandb.log({"Score_1": score,
                        "Score_10": average10,
                        "Policy Loss": pi_loss,
                        "Alpha Loss": alpha_loss,
                        "Critic Loss 1": q1_loss,
                        "Critic Loss 2": q2_loss,
                        "Alpha": pi.log_alpha.exp(),
                        "Update Steps": update_steps,
                        "Episode": n_epi ,
                        "Buffer size": memory.size()})
                score = 0.0

            
            if n_epi%config["save_every"]==0:
                torch.save({
                        "pi": pi.state_dict(),
                        "q1": q1.state_dict(),
                        "q2": q2.state_dict(),
                        "q1_target": q1_target.state_dict(),
                        "q2_target": q2_target.state_dict(),
                    }, os.path.join(model_save_dir, f"{n_epi}.ckpt"))
                wandb.save(os.path.join(model_save_dir, f"{n_epi}.ckpt"))

        env.close()

if __name__ == '__main__':
    config = get_config()
    print(config)
    main(config)