import torch
from actor_critic_PG import Actor

def REINFORCE_loss():

    

    pass

def train_REINFORCE(
    agent, 
    env, 
    epochs,
    batch_size = 32,
    cutoff = None,
    ):
    '''
    Trains REINFORCE algorithm on model

    Args:
        batch_size (int): Unlike other batch sizes, defines 
            how large of a batch to extract from the replay buffer
    '''

    optimizer = torch.optim.SGD(lr = 0.001)

    # Run episodes for epochs number:
    for i in range(epochs):
        rewards = env.run_episode(agent, 
            gather_buffer = True,
            cutoff = cutoff)

        xbatch, ybatch, action_batch = env.sample_buffer(batch_size = batch_size)

        #torch.log()



    pass

