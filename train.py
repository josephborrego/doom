from vizdoom import *
from frames import *
import dqn
from collections import deque
import numpy as np
import random

num_episodes = 50
game = DoomGame()
game.set_doom_scenario_path("borrego.wad")
game.init()
stack_size = 4

stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)

#actions
shoot = [0, 0, 1]
left = [1, 0, 0]
right = [0, 1, 0]
actions = [shoot, left, right]

for i in range(num_episodes):
    game.new_episode()
    state = game.get_state().screen_buffer
    state, stacked_frames = stack_frames(stacked_frames, state, True)
    
    while not game.is_episode_finished():
        #preprocess part
        img = state.screen_buffer
        misc = state.game_variables
        action = random.choice(actions)
        #DQN part, chooses actions ^^
        print(action)
        reward = game.make_action(action)

game.close()

