from vizdoom import *
from frames import *
from dqn import *
from collections import deque
import numpy as np
import random
import skimage
import itertools as it
from memory import *
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
game = DoomGame()
game.load_config("borrego.cfg")
game.set_doom_scenario_path("borrego.wad")
game.set_window_visible(True)
game.set_mode(Mode.PLAYER)
game.set_screen_format(ScreenFormat.BGR24)
game.set_screen_resolution(ScreenResolution.RES_640X480)
game.init()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

model_savefile = "./model-doom.pth"
save_model = True
load_model = False

stack_size = 4
num_episodes = 50
memory_size = 1000
memory = Memory(memory_size)
stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]

if load_model:
    model = torch.load(model_savefile)
else:
    model = DQN(len(actions))

#print(actions)

for i in range(num_episodes):
    game.new_episode()
    state = preprocess_frame(game.get_state().screen_buffer)
    cv2.imshow('yo', state)
    #img, stacked_frames = stack_frames(stacked_frames, state, True)

    while not game.is_episode_finished():
        action = random.choice(actions)
        reward = game.make_action(action)

game.close()

