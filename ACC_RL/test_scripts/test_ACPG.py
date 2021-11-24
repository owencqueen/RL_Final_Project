import torch
import sys; sys.path.append('..')

from actor_critic_PG import Actor

def test_output_Actor():
    '''Only tests output (no training)'''
    test_input = torch.randn(20)

    model = Actor(
        input_len = 20,
        action_size = 3
        )

    print(model(test_input))

if __name__ == '__main__':
    test_output_Actor()