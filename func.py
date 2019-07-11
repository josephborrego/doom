import cv2
import numpy as np
from skimage import transform
from vizdoom import *

# box the threats
def box(buffer, x, y, wt, ht, color):
 for i in range(wt):
  buffer[y, x+ i, :] = color
  buffer[y + ht, x + i, :] = color
 for i in range(ht):
  buffer[y + i, x, :] = color
  buffer[y + i, x + wt, :] = color

# inverses color to red and blue in other cv2 screen
def color_labels(labels):
 tmp = np.stack([labels] * 3, -1)
 tmp[labels == 0] = [255, 0, 0]
 tmp[labels == 1] = [0, 0, 255]
 return tmp

# preprocess the frame to reduce memory and increase speed
def prep_frame(frame):
 cropped = frame[30:-10,30:30]
 normal = cropped/255.0
 prep = transform.resize(normal,[84,84])
 return prep

def create_env():
 game = DoomGame()
 game.set_doom_scenario_path('../scenarios/basic.wad')
 game.set_doom_map("map02")

 # actions
 actions = np.identity(21,dtype=int).tolist()
 #rendering
 game.set_screen_resolution(ScreenResolution.RES_640X480)
 game.set_screen_format(ScreenFormat.BGR24)
 game.set_render_hud(True)
 game.set_render_crosshair(True)
 game.set_render_weapon(True)
 game.set_render_decals(False)
 game.set_render_particles(False)
 game.set_labels_buffer_enabled(True)
 #movement
 game.add_available_button(Button.MOVE_LEFT)
 game.add_available_button(Button.MOVE_RIGHT)
 #interaction
 game.add_available_button(Button.ATTACK)
 game.add_available_button(Button.USE)
 #misc
 game.add_available_button(Button.CROUCH)
 game.add_available_button(Button.RELOAD)
 game.add_available_button(Button.ZOOM)
 #weapons
 game.add_available_button(Button.SELECT_WEAPON1)
 game.add_available_button(Button.SELECT_WEAPON2)
 game.add_available_button(Button.SELECT_WEAPON3)
 game.add_available_button(Button.SELECT_WEAPON4)
 game.add_available_button(Button.SELECT_WEAPON5)
 game.add_available_button(Button.SELECT_WEAPON6)
 game.add_available_button(Button.SELECT_WEAPON7)
 game.add_available_button(Button.SELECT_WEAPON8)
 game.add_available_button(Button.SELECT_WEAPON9)
 game.add_available_button(Button.SELECT_WEAPON0)

 # non binary - pass in as degrees
 game.add_available_button(Button.MOVE_FORWARD_BACKWARD_DELTA, 10)
 game.add_available_button(Button.MOVE_LEFT_RIGHT_DELTA, 5)
 game.add_available_button(Button.TURN_LEFT_RIGHT_DELTA,5)
 game.add_available_button(Button.LOOK_UP_DOWN_DELTA)
 #variables
 game.add_available_game_variable(GameVariable.AMMO2)
 game.set_episode_timeout(200)
 game.set_episode_start_time(10)
 game.set_window_visible(True)
 game.set_living_reward(-1)
 game.init()
 return game, actions

# image, state, actions
 
