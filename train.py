from vizdoom import *
from frames import *
import dqn
from collections import deque
import numpy as np
import random
import skimage
import itertools as it


game = DoomGame()
game.load_config("borrego.cfg")
game.set_doom_scenario_path("borrego.wad")
game.set_window_visible(False)
game.set_mode(Mode.PLAYER)
game.set_screen_format(ScreenFormat.GRAY8)
game.set_screen_resolution(ScreenResolution.RES_640X480)
game.init()

model_savefile = "./model-doom.pth"
stack_size = 4
num_episodes = 50
stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]

for i in range(num_episodes):
    game.new_episode()
    #state = game.get_state().screen_buffer
    state = game.get_state()
    img = state.screen_buffer
    state, stacked_frames = stack_frames(stacked_frames, img, True)

    while not game.is_episode_finished():
        #preprocess part
        #misc = state.game_variables
        action = random.choice(actions)
        #DQN part, chooses actions ^^
        print(action)
        reward = game.make_action(action)

game.close()

