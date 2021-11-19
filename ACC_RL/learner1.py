import torch
import torch.nn.functional as F
from torch.nn import Linear

from tqdm import trange
import matplotlib.pyplot as plt

from base_env import Environment

class PolicyNet(torch.nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.fc1 = Linear(16, 32)
        self.fc2 = Linear(32, 32)
        self.fc3 = Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x

if __name__ == '__main__':

    env = Environment()

    agent = PolicyNet()

    episodes_training = 10

    optimizer = torch.optim.Adam(params = agent.parameters(), lr = 1)

    rewards = []

    for i in range(episodes_training):

        # Run an episode:
        r = env.run_episode(agent = agent, cutoff = 1e4)
        rewards.append(r)

        optimizer.zero_grad()

        # Train on replay buffer:
        for i in range(len(env.replay_buffer_y)):
            #print(i)

            reward = env.replay_buffer_y[i]

            loss = torch.autograd.Variable(torch.FloatTensor([reward]), requires_grad = True)

            loss.backward()
        
        optimizer.step()


    # Plot:
    plt.plot(rewards)
    plt.show()
    



    