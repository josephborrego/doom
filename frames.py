# I was inspired to emabrk on this journey with the help from Thomas Simonini
# # https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Deep%20Q%20Learning/Doom/Deep%20Q%20learning%20with%20Doom.ipynb

import numpy as np
from skimage import transform
import skimage.transform
from collections import deque
import cv2
from PIL import Image
stack_size = 4

# https://towardsdatascience.com/image-pre-processing-c1aec0be3edf
# Preprocessing is an important step, because we want to reduce the complexity of our states to reduce the computation time
# needed for training.

def preprocess_frame(frame):
    # converts to gray scale
    src = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('uint8')
    #cropped_frame = frame[30:-10,30:-30] 
    #normalized_frame = cropped_frame/255.0 
    cropped_frame = src[30:-10, 30:-30]
    normalized_frame = cropped_frame/255.0
    #preprocessed_frame = skimage.transform.resize(normalized_frame, (84,84)) 
    width = int(normalized_frame.shape[1] * 60 / 100)
    height = int(normalized_frame.shape[0] * 60 / 100)
    dim = (width, height)
    # resize the frame
    preprocessed_frame = cv2.resize(normalized_frame, dim, interpolation=cv2.INTER_LINEAR)
    preprocessed_frame = preprocessed_frame.astype(np.float32)
    # remove gaussian noise
    #x = cv2.blur(preprocessed_frame, (5,5))
    x = cv2.GaussianBlur(preprocessed_frame, (5, 5), 0).astype('float32')
    blur = cv2.GaussianBlur(preprocessed_frame, (1, 9), 0).astype('uint8')
    # segmentation & morphology
    # segment - separating background from foreground objects &  more noise removal
    # otsus binarization
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # apply another blur to improve the looks
    # further noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # seperate different objects in the image with markers
    ret, markers = cv2.connectedComponents(sure_fg)
    #markers = markers + 1
    #markers[unknown == 255] == 0
    #markers = cv2.watershed(frame, markers)
    #cv2.imshow('yo', sure_fg)
    return x
    #return markers
    
# https://pdfs.semanticscholar.org/74c3/5bb13e71cdd8b5a553a7e65d9ed125ce958e.pdf
# stack frames is used for the experience replay buffer
# Stacking frames is really important because it helps us to give have a sense of motion to our NN
# For the first frame, we feed 4 frames
# At each timestep, we add the new frame to deque and then we stack them to form a new stacked frame
#And so onstack
# If we're done, we create a new stack with 4 new frames (because we are in a new episode).
def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)
        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames
