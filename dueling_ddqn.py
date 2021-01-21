import numpy as np
import random
from collections import namedtuple, deque

from dueling_model import DuelingQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
ALPHA = 0.5             # experience replay sampling exponent
BETA = 0.1              # experience replay weigthed importance sampling exponent
EPSILON = 0.01          # epsilon parameter added to priorities for experience replay 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, ALPHA, BETA)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.beta = BETA
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(self.t_step,state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1)
        if self.t_step % UPDATE_EVERY == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(self.t_step)
                self.learn(experiences, GAMMA, self.beta)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, beta):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        sampled_times, states, actions, rewards, next_states, dones, priorities, sampling_weight = experiences

        # Get max predicted Q values (for next states) from target model
        Q_max_action = torch.argmax(self.qnetwork_local(next_states), dim=1)
        Q_targets_next = self.qnetwork_target(next_states).gather(1,Q_max_action.unsqueeze(1)) 
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        TD_Error= torch.abs(Q_targets-Q_expected).cpu().data.numpy() + EPSILON
        
        self.memory.update_priorities(sampled_times ,TD_Error)
        
        beta = beta**(1-beta)
        
        is_weight = ALPHA*(1/BATCH_SIZE*1/sampling_weight)**beta
        is_weight = torch.from_numpy(is_weight).float().to(device)

        # Compute loss
        loss = F.mse_loss(is_weight * Q_expected, is_weight * Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha, beta):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (float): sampling exponent (between 0 and 1)
            beta(float): importance sampling exponent
        """ 
        self.action_size = action_size
        self.memory = {} #deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done" ,"priority"])
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.beta = beta
    
    def add(self, t_step, state, action, reward, next_state, done, priority=1):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, priority)
        self.memory[t_step] = e
        if t_step >=BUFFER_SIZE:
            self.memory.pop(t_step-BUFFER_SIZE)
    
    def sample(self, t_step):
        """Randomly sample a batch of experiences from memory."""
        sampling_weight = [e.priority**self.alpha for e in list(self.memory.values()) if e is not None]
        sampling_weight = sampling_weight/np.sum(sampling_weight)
        sampled_times = np.random.choice(list(self.memory.keys()), replace=False, p=sampling_weight, size=self.batch_size)
        experiences = [self.memory[t_step] for t_step in sampled_times] 

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        priorities = torch.from_numpy(np.vstack([e.priority for e in experiences if e is not None])).float().to(device)

        time_shift = max(t_step-BUFFER_SIZE,0)
        sampling_weight= sampling_weight[sampled_times-time_shift] #time steps are integers and start from 0, so can be used for indexing 
        
        return (sampled_times, states, actions, rewards, next_states, dones, priorities, sampling_weight)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def update_priorities(self, t_steps, priorities):
        """Update the priority of experiences"""
        
        for i, t_step in enumerate(t_steps):
            temp_list = self.memory[t_step]._replace(priority=priorities[i])