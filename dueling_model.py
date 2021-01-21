import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2_value = nn.Linear(fc1_units, fc2_units)
        self.fc3_value = nn.Linear(fc2_units, action_size)
        self.fc2_adv = nn.Linear(fc1_units, fc2_units)
        self.fc3_adv = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        
        value = F.relu(self.fc2_value(x))
        value = F.relu(self.fc3_value(value))
        
        adv = F.relu(self.fc2_value(x))
        adv = F.relu(self.fc3_value(adv))
        
        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage
        
        return Q

   
