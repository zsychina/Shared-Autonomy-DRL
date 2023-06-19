import numpy
import torch
import torch.nn as nn
import gym
import pygame

from network import Qnet
from helper import Transition, ReplayMemory, human_action

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("LunarLander-v2", render_mode='human')

action_n = env.action_space.n
state_n = len(env.observation_space.sample()) + 1

policy_net = Qnet(state_n, action_n).to(device)
target_net = Qnet(state_n, action_n).to(device)

policy_net.load_state_dict(torch.load('./out/policy.pt'))
target_net.load_state_dict(torch.load('./out/target.pt'))


def user_policy(state, user_action): 
    alpha = 0.8    
    state_cat = torch.cat((state, user_action), dim=1)
    # Q'
    state_values = policy_net(state_cat) - policy_net(state_cat).min(1)[0]
    # mask can never be all false, because there must exist a biggest value
    mask = torch.gt(state_values, (1 - alpha) * state_values.max(1)[0]) 
    a_candidates = torch.nonzero(mask)[:, 1].view(-1, 1)
    # dist is one-hot
    dist = torch.where(a_candidates == user_action, torch.tensor(1), torch.tensor(0))
    a_t = a_candidates[dist.argmax()].view(1, 1)
    return a_t

state, _ = env.reset()
state = torch.tensor(state, device=device).unsqueeze(0)
while True:
    keys = pygame.key.get_pressed()
    if any(keys):
        user_action = human_action(keys)
    else:
        user_action = env.action_space.sample()
        user_action = torch.tensor(user_action, device=device).view(1, 1)
        
    action = user_policy(state, user_action)
    
    state_next, reward, terminated, truncated, _ = env.step(action.item())
    state_next = torch.tensor(state_next, device=device).unsqueeze(0)    
    
    state = state_next
    
    if terminated or truncated:
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    


env.close()    


    


