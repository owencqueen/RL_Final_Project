import torch
import torch.nn.functional as F

# Networks for DDPG

class Actor_DDPG(torch.nn.Module):
    '''
    Actor/policy network for DDPG

    Args:
        state_dim (int): Dimension of input state
        action_dim (int): Dimension of output action
        hidden_dims (list): In order, dimension of hidden layers
    '''

    def __init__(
            self,
            state_dim = 17,
            action_dim = 3,
            hidden_dims = [32, 64, 32],
        ):
        super().__init__()

        modules = []

        # Build the state embedder model:
        for i in range(len(hidden_dims)):
            input_dim = state_dim if i == 0 else hidden_dims[i-1]

            # Each unit is linear -> leaky relu (parameterized ReLU)
            modules.append(torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dims[i]),
                torch.nn.LeakyReLU()))

        self.state_embedder = torch.nn.Sequential(*modules)

        self.fc_action = torch.nn.Linear(hidden_dims[-1], action_dim)
        #self.init_weights()

    def forward(self, x):
        x = self.state_embedder(x)
        action = self.fc_action(x)
        return F.relu(action)


    def init_weights(self):
        ''' Init all weights in model '''

        torch.nn.init.normal_(self.fc_action.bias.data[:-1], mean = 1e6, std = 1000)
        torch.nn.init.constant_(self.fc_action.bias.data[-1], 0)

class Critic_DDPG(torch.nn.Module):
    '''
    Critic/value network for DDPG 
    
    Args:
        state_dim (int): Dimension of input state
        action_dim (int): Dimension of input action
    '''
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.state_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU()
        )

        self.action_net = torch.nn.Sequential(
            torch.nn.Linear(action_dim, 32),
            torch.nn.LeakyReLU()
        )

        self.final_score = torch.nn.Linear(32 * 2, 1)

    def forward(self, state, action):
        # Embed both state and action:
        state_embed = self.state_net(state)
        action_embed =  self.action_net(action)

        # Concatenate and score:
        x = torch.squeeze(torch.cat((state_embed, action_embed), dim = 1))
        x = self.final_score(x)
        return x
