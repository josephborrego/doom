import numpy as np
from skimage import transform
import skimage.transform
from collections import deque
stack_size = 4

def preprocess_frame(frame):
    # Greyscale frame already done in our vizdoom config
    #x = np.mean(frame,-1)
    cropped_frame = frame[30:-10,30:-30] # Crop the screen (remove the roof b/c no info)
    normalized_frame = cropped_frame/255.0 # Normalize Pixel Values
    preprocessed_frame = skimage.transform.resize(normalized_frame, (84,84)) # Resize
    preprocessed_frame = preprocessed_frame.astype(np.float32)
    return preprocessed_frame

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
