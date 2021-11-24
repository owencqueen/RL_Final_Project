import torch
import torch.nn.functional as F

class Actor(torch.nn.Module):
    '''
    Actor network (policy network) for actor-critic method
        - Uses Gaussian sampling for action

    Based loosely on VAE architecture: 
        https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

    Args:
        input_len (int): length of input
    '''

    def __init__(self, 
            input_len, 
            action_size,
            hidden_dims = [32, 64, 32] 
        ):
        super().__init__()

        modules = []

        # Build the state embedder model:
        for i in range(len(hidden_dims)):
            input_dim = input_len if i == 0 else hidden_dims[i-1]

            # Each unit is linear -> leaky relu (parameterized ReLU)
            modules.append(torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dims[i]),
                torch.nn.LeakyReLU()))

        self.state_embedder = torch.nn.Sequential(*modules)

        self.fc_mu = torch.nn.Linear(hidden_dims[-1], action_size)
        self.fc_var = torch.nn.Linear(hidden_dims[-1], action_size)

    def encode(self, x):
        x = self.state_embedder(x) # State embedding
        mu = self.fc_mu(x) # Mu of Gaussian
        log_var = self.fc_var(x) # Log of variance (still need to exp())

        return [mu, log_var]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # Gets standard deviation
        eps = torch.randn_like(std) # Gets N(0,1) vector
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encode(x)
        action = self.reparameterize(mu, log_var)
        return [action, mu, log_var]


    def loss_function(self):
        pass

class Critic(torch.nn.Module):
    def __init__(self):
        pass