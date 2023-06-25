import torch
import torch.nn as nn
import torch.nn.functional as F

class Qnet(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Qnet, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, n_actions)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


if __name__ == '__main__':
    import numpy as np
    qnet = Qnet(5, 2)
    state = [0.1, 2.1, -0.7, 0.4, 1]
    state = torch.Tensor(state)
    action = qnet(state).max(0)[1]
    print(qnet(state))
    print(action.item())
    
