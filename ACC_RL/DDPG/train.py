import torch
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Union, Callable

from .DDPG_model import Actor_DDPG as Actor, Critic_DDPG as Critic

# Modeled after: https://github.com/vy007vikas/PyTorch-ActorCriticRL/blob/master/train.py

TAU = 0.001

def soft_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        # Gradual updating of target network "stabilizes learning"
        target_param.data.copy_(
            target_param.data * (1.0 - TAU) + param.data * TAU
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class DDPGTrainer:
    '''
    Training class for the Deep Deterministic Policy Gradient

    Args:
        state_dim (int): Dimension of state input vector
        action_dim (int): Dimension of action (output) vector
        batch_size (int, optional): Size of batch to sample during each update
        gamma (float, optional): Weighting factor on the Q_prime in the TD
            error (check DDPG paper)
        exploration_noise (float or tensor, optional): Standard deviation of 
            Gaussian noise added to each output action. If tensor is given, 
            it must be the same length as action, and noise is defined on a per-
            action basis.
        actor_lr (float, optional): Learning rate for actor network
            (:default: `0.1`)
        critic_lr (float, optional): Learning rate for critic network 
            (:default: `0.1`)
        state_transform (callable, optional): Transforms the state each time it 
            is fed into one of the networks.
        actor_layers (list, optional): List of sizes of hidden layers for the 
            actor network. Used for experimentation with different network 
            structures. (:default: `[32, 64, 32]`)
    '''
    
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            batch_size = 128,
            gamma: float = 1,
            exploration_noise: Union[float, torch.Tensor] = 100,
            actor_lr: float = 0.1,
            critic_lr: float = 0.1,
            state_transform: Callable[[torch.Tensor], torch.Tensor] = torch.nan_to_num,
            actor_layers = [32, 64, 32],
        ):

        self.gamma = gamma
        self.batch_size = batch_size
        self.explore_noise = exploration_noise
        self.state_transform = state_transform
        
        self.actor = Actor(state_dim, action_dim, actor_layers)
        self.copy_actor = Actor(state_dim, action_dim, actor_layers)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), actor_lr)

        self.critic = Critic(state_dim, action_dim)
        self.copy_critic = Critic(state_dim, action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), critic_lr)

        # Copy parameters of critic and actor
        hard_update(self.copy_actor, self.actor); hard_update(self.copy_critic, self.critic)

    def explore_action(self, state, explore_noise_weight = 1):
        #print('state', state)
        # if torch.isnan(state).any():
        #     print('nan state', state)
        #     exit()
        action = self.actor(self.state_transform(state))
        #print('action', action)
        #exit()

        # Note: GitHub repo uses Ornstein-Uhlenbeck noise here
        # Use normal for now:
        #print('action', action)
        return (action + torch.randn_like(action) * self.explore_noise * explore_noise_weight) #* (torch.tensor([0, 1, 1]))
        #return torch.abs(torch.nan_to_num(action) + torch.randn_like(action) * self.explore_noise) + torch.tensor([100000, 100000, 0])

    def exploit_action(self, state):

        return self.actor(Variable(state))

    def optimize(self, replay_buffer):

        s1, a1, r1, s2 = replay_buffer.TD_sample(batch_size = self.batch_size)

        # Optimize critic -----------------------:

        a2 = self.copy_actor.forward(self.state_transform(s2)).detach() # Action 2
        Q_prime = torch.squeeze(self.copy_critic.forward(self.state_transform(s2), a2).detach())

        # Loss: L = 1/N * \sum\limits_{i} (yi - Q_pred)^2
        yi = r1 + self.gamma * Q_prime
        Q_pred = torch.squeeze(self.critic.forward(self.state_transform(s1), a1))

        # Smooth L1 loss suggested by GitHub repo, used here
        critic_loss = F.smooth_l1_loss(Q_pred, yi)

        # Run critic optimizer::
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Optimize actor  -----------------------:
        a1_pred = self.actor.forward(self.state_transform(s1))
        actor_loss = -1 * torch.sum(self.critic.forward(self.state_transform(s1), a1_pred))

        # Run actor optimizer:
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update models:
        soft_update(self.copy_critic, self.critic)
        soft_update(self.copy_actor, self.actor)


