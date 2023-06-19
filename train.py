import numpy
import torch
import torch.optim as optim
import torch.nn as nn
import gym
import pygame

from network import Qnet
from helper import Transition, ReplayMemory, human_action

BATCH_SIZE = 16
GAMMA = 0.99
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("LunarLander-v2")

action_n = env.action_space.n
state_n = len(env.observation_space.sample()) + 1 # add human_action
# print(state_n, action_n)    # 9 4

policy_net = Qnet(state_n, action_n).to(device)
target_net = Qnet(state_n, action_n).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

def optimize_model():
    if len(memory) < 10 * BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    # state_batch and state_next_batch should be considered state_[next]_cat_batch
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward) 
    state_next_batch = torch.cat(batch.next_state) 
    
    # y for predictions
    y = reward_batch + GAMMA * target_net(state_next_batch).gather(1, policy_net(state_next_batch).max(1)[1].unsqueeze(1))
    
    criterion = nn.SmoothL1Loss()
    loss = criterion(y, policy_net(state_batch).gather(1, action_batch))
    
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    

def user_policy(state, user_action): 
    alpha = 0.8    
    state_cat = torch.cat((state, user_action), dim=1)
    # Q'
    state_values = policy_net(state_cat) - policy_net(state_cat).min(1)[0]
    # mask can never be all false, because there must exist a biggest value
    mask = torch.gt(state_values, (1 - alpha) * state_values.max(1)[0]) 
    a_suboptimal = torch.nonzero(mask)[:, 1].view(-1, 1)
    # dist is one-hot
    dist = torch.where(a_suboptimal == user_action, torch.tensor(1), torch.tensor(0))
    a_t = a_suboptimal[dist.argmax()].view(1, 1)
    return a_t


# train without human input
step_cnt = 0
episode_num = 50
for episode_idx in range(episode_num):
    episode_reward = 0
    terminated = False
    state, _ = env.reset()
    state = torch.tensor(state, device=device).unsqueeze(0)  
    while not terminated:
        step_cnt += 1
        # 1. Sample action At
        # real user action, or random value, or last value
        user_action = env.action_space.sample()
        user_action = torch.tensor(user_action, device=device).view(1, 1)
        
        action = user_policy(state, user_action)
        
        state_cat = torch.cat((state, user_action), dim=1)
        
        # 2. Execute action and observe (St+1, At+1, Rt)
        state_next, reward, terminated, truncated, _ = env.step(action.item())
        state_next = torch.tensor(state_next, device=device).unsqueeze(0)
        
        # TODO: get user_action_next, inferred?
        user_action_next = target_net(state_cat).max(1)[1].view(1, 1)
        
        state_next_cat = torch.cat((state_next, user_action_next), dim=1)
                
        episode_reward += reward
        reward = torch.tensor([reward], device=device)
        
        # 3. Store transition (St, At, Rt, St+1)
        memory.push(state_cat, action, state_next_cat, reward)
        
        # Update state
        state = state_next # not state_cat
        
        # 4. Optimize Model
        if terminated or truncated:
            for k in range(100):
                optimize_model()
        
        # 5. every C steps reset \hat{Q}=Q
        if step_cnt % 1000 == 0:
            step_cnt = 0
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in target_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]
            target_net.load_state_dict(target_net_state_dict)
        
    print("Episode {}, reward {}".format(episode_idx, episode_reward))

env.close()

# train with human input
env = gym.make("LunarLander-v2", render_mode='human')
step_cnt = 0
episode_rewards = []
episode_num = 100
for episode_idx in range(episode_num):
    episode_reward = 0
    terminated = False
    state, _ = env.reset()
    state = torch.tensor(state, device=device).unsqueeze(0)  
    while not terminated:
        step_cnt += 1
        # 1. Sample action At
        # real user action, or random value, or last value
        keys = pygame.key.get_pressed()
        if any(keys):
            user_action = human_action(keys)
        else:
            user_action = env.action_space.sample()
            user_action = torch.tensor(user_action, device=device).view(1, 1)
        
        action = user_policy(state, user_action)
        
        state_cat = torch.cat((state, user_action), dim=1)
        
        # 2. Execute action and observe (St+1, At+1, Rt)
        state_next, reward, terminated, truncated, _ = env.step(action.item())
        state_next = torch.tensor(state_next, device=device).unsqueeze(0)
        
        # TODO: get user_action_next, inferred?
        # target_network(state_cat) -> possible actions
        user_action_next = target_net(state_cat).max(1)[1].view(1, 1)
        
        state_next_cat = torch.cat((state_next, user_action_next), dim=1)
                
        episode_reward += reward
        reward = torch.tensor([reward], device=device)
        
        # 3. Store transition (St, At, Rt, St+1)
        memory.push(state_cat, action, state_next_cat, reward)
        
        # Update state
        state = state_next # not state_cat
        
        # 4. Optimize Model
        if terminated or truncated:
            for k in range(100):
                optimize_model()
        
        # 5. every C steps reset \hat{Q}=Q
        if step_cnt % 1000 == 0:
            step_cnt = 0
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in target_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]
            target_net.load_state_dict(target_net_state_dict)
    
    episode_rewards.append(episode_reward)
    print("Episode {}, reward {}".format(episode_idx, episode_reward))

env.close()

# plot
import matplotlib.pyplot as plt
plt.plot(episode_rewards)
plt.show()
plt.savefig('./out/lander_rewards.png')

# save
torch.save(policy_net.state_dict(), './out/policy.pt')
torch.save(target_net.state_dict(), './out/target.pt')

