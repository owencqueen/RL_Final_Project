import torch
import torch.nn.functional as F
from torch.autograd import Variable

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
    
    def __init__(
            self,
            state_dim,
            action_dim,
            batch_size = 128,
            gamma: float = 1,
            exploration_noise: float = 100,
            actor_lr = 0.001,
            critic_lr = 0.001,
            actor_layers = [32, 64, 32],
        ):

        self.gamma = gamma
        self.batch_size = batch_size
        self.explore_noise = exploration_noise
        
        self.actor = Actor(state_dim, action_dim, actor_layers)
        self.copy_actor = Actor(state_dim, action_dim, actor_layers)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), actor_lr)

        self.critic = Critic(state_dim, action_dim)
        self.copy_critic = Critic(state_dim, action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), critic_lr)

        # Copy parameters of critic and actor
        hard_update(self.copy_actor, self.actor); hard_update(self.copy_critic, self.critic)

    def explore_action(self, state):
        action = self.actor(Variable(torch.from_numpy(state))).detach().data.numpy()

        # Note: GitHub repo uses Ornstein-Uhlenbeck noise here
        # Use normal for now:
        return (action + torch.randn_like(action) * self.explore_noise)

    def exploit_action(self, state):

        return self.actor(Variable(torch.from_numpy(state))).detach().data.numpy()

    def optimize(self, replay_buffer):

        s1, a1, r1, s2 = replay_buffer.TD_sample(batch_size = self.batch_size)

        # Optimize critic -----------------------:

        a2 = self.copy_actor.forward(s2).detach() # Action 2
        Q_prime = torch.squeeze(self.copy_critic.forward(s2, a2).detach())

        # Loss: L = 1/N * \sum\limits_{i} (yi - Q_pred)^2
        yi = r1 + self.gamma * Q_prime
        Q_pred = torch.squeeze(self.critic.forward(s1, a1))

        # Smooth L1 loss suggested by GitHub repo, used here
        critic_loss = F.smooth_l1_loss(Q_pred, yi)

        # Run critic optimizer::
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Optimize actor  -----------------------:
        a1_pred = self.actor.forward(s1)
        actor_loss = -1 * torch.sum(self.critic.forward(s1, a1_pred))

        # Run actor optimizer:
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update models:
        soft_update(self.copy_critic, self.critic)
        soft_update(self.copy_actor, self.actor)


