from vizdoom import *
import numpy as np
from skimage import transform
from func import *
import random
import time
import cv2
import numpy as np

game, actions = create_env()
episodes = 10
font = cv2.FONT_HERSHEY_SIMPLEX

for i in range(episodes):
 game.new_episode()
 while not game.is_episode_finished():
  state = game.get_state()
  img = state.screen_buffer
  misc = state.game_variables
  labels = state.labels_buffer

  #red and blue screen
  cv2.imshow('doom label buff', color_labels(labels))

  #boxing images
  for l in state.labels:
   box(img, l.x, l.y, l.width, l.height, [0,0,203])
   color = [l.value] * 3
   cv2.putText(img,l.object_name,(l.x,l.y-3),font,0.3,[int(c) for c in color],1,cv2.LINE_AA)
  cv2.imshow('Vizdoom screen buffer', img)
  cv2.waitKey(1)

  ch = random.choice(actions)
  game.make_action(ch, 5)
  time.sleep(0.01)

game.close()
