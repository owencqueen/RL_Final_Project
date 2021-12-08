# Adaptive Cruise Control for a Hybrid Vehicle with Deep Policy Gradients
Final project for ECE 517/414 Reinforcement Learning.

## Running Code
Example code is provided to train the REINFORCE and DDPG algorithms and view training curves over a small number of steps. This requires the confidential Blazer Model, and it cannot be ran without it. Each test script is ran with the raw speed reward function.

### REINFORCE
Run the following.
```
python3 ACC_RL/test_scripts/test_REINFORCE.py
```

### DDPG

Run the following. Be cautioned that it takes ~2 minutes and will output a plot.
```
python3 ACC_RL/test_scripts/test_DDPG.py
```
