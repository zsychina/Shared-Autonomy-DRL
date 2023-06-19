from collections import namedtuple, deque
import random

import pygame
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

def human_action(keys):
    action = 0
    human_input = 'ELSE'
    if keys[pygame.K_LEFT]:
        human_input = 'LEFT'
        action = 1
    if keys[pygame.K_RIGHT]:
        human_input = 'RIGHT'
        action = 3
    if keys[pygame.K_UP]:
        human_input = 'UP'
        action = 2
    if keys[pygame.K_DOWN]:
        human_input = 'DOWN'
        action = 0
    action = torch.tensor(action, device=device).view(1, 1)
    print("Human input: {}".format(human_input))
    return action


if __name__ == '__main__':
    memory = ReplayMemory(1000)
    memory.push([1, 2, 3, 4, 5, 6, 7, 8], 0, [1, 2, 3, 4, 5, 6, 7, 8], 10)
    memory.push([1, 2, 3, 4, 5, 6, 7, 8], 0, [1, 2, 3, 4, 5, 6, 7, 8], 10)
    memory.push([1, 2, 3, 4, 5, 6, 7, 8], 0, [1, 2, 3, 4, 5, 6, 7, 8], 10)
    memory.push([1, 2, 3, 4, 5, 6, 7, 8], 0, [1, 2, 3, 4, 5, 6, 7, 8], 10)
    memory.push([1, 2, 3, 4, 5, 6, 7, 8], 0, [1, 2, 3, 4, 5, 6, 7, 8], 10)
    transitions = memory.sample(3)
    print(transitions)
    batch = Transition(*zip(*transitions))
    print(batch)
    
    
    
