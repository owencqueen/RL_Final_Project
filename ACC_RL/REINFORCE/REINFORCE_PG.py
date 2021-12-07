import torch
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transform_state(state_vec):

    state_vec = state_vec.T
    my_vec = state_vec[:13][torch.tensor([1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1], dtype = bool)]
    
    return my_vec.T

class REINFORCE_trainer:
    
    def __init__(self, state_dim = 11, hidden_dims = 32, action_dim = 3, lr_pi = 3e-4,\
                 gamma = 0.99, train_v_iters = 1):

        self.gamma = gamma
        self.action_dim = action_dim
        self.policy = Gaussian_pi(state_dim, hidden_dims, action_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr = lr_pi)
        self.train_v_iters = train_v_iters 

    def select_action(self, state):
        state = torch.from_numpy(transform_state(state)).float().unsqueeze(0) #make tensor object
        mean, stdev = self.policy(state)

        normaldist = Normal(mean, stdev)
        action = normaldist.sample()
        ln_prob = normaldist.log_prob(action)
        ln_prob = ln_prob.sum()
        action = torch.tanh(action)
        action = action.numpy()

        return action[0], ln_prob

    def train(self, trajectory):
        '''
        Trajectory: list of the form [(state, action, lnP(a_t|s_t),reward),...]
        Using the "rewards-to-go" forulation of policy gradient as it is a bit easier
        to implement in my opinion
        '''
        states = [item[0] for item in trajectory]
        actions = [item[1] for item in trajectory]
        log_probs = [item[2] for item in trajectory]
        rewards = [item[3] for item in trajectory]

        #Rewards-to-go formulation
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append( - log_prob * R)

        policy_loss = torch.stack(policy_loss).sum()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return policy_loss



class Gaussian_pi(nn.Module):
    '''
    Consists of a nn with 1 hidden layer. Outputs mean and log standard deviation (parameters)
    of a gaussian policy
    '''

    def __init__(self, state_dim, hidden_dims, action_dim):

        super(Gaussian_pi, self).__init__()
        self.action_dim = action_dim
        num_outputs = action_dim

        self.linear = nn.Linear(state_dim, hidden_dims)
        self.mean = nn.Linear(hidden_dims, num_outputs)
        self.log_stdev = nn.Linear(hidden_dims, num_outputs)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear(x))
        mean = self.mean(x)
        log_stdev = self.log_stdev(x)
        stdev = log_stdev.exp()
        
        return mean, stdev

class Softmax_pi(nn.Module):
    '''
    Softmax action selection
    '''

    def __init__(self, state_dim, hidden_dims, action_dim):

        super(Softmax_pi, self).__init__()
        num_outputs = action_dim
        self.linear1 = nn.Linear(state_dim, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, num_outputs)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        action_scores = self.linear2(x)

        return F.softmax(action_scores)
