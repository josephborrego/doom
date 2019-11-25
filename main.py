
import numpy as np           # Handle matrices
from vizdoom import *        # Doom Environment
import random                # Handling random number generation
import time                  # Handling time calculation
from skimage import transform # Help us to preprocess the frames
from collections import deque # Ordered collection with ends

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage

warnings.filterwarnings('ignore')
#tf.enable_eager_execution()


"""
Here we create our environment
"""
def create_environment():
    # Load the correct configuration
    game = DoomGame()
    game.load_config("../scenarios/basic.cfg")
    # Load the correct scenario (in our case basic scenario)
    game.set_doom_scenario_path("basic.wad")
    game.init()
    # Here our possible actions
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]
    return game, possible_actions

game, possible_actions = create_environment()
stack_size = 4 # We stack 4 frames
# Initialize deque with zero-images one array for each image
stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)

### MODEL HYPERPARAMETERS
state_size = [84,84,4]      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels) 
action_size = game.get_available_buttons_size()              # 3 possible actions: left, right, shoot
learning_rate =  0.0002      # Alpha (aka learning rate)
### TRAINING HYPERPARAMETERS
total_episodes = 500        # Total episodes for training
max_steps = 100              # Max possible steps in an episode
batch_size = 64
# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.0001            # exponential decay rate for exploration prob
# Q learning hyperparameters
gamma = 0.95               # Discounting rate
### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep
### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True
## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False

#tf.compat.v1.reset_default_graph()
DQNetwork = DQN(state_size, action_size, learning_rate)
