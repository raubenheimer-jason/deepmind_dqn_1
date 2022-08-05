
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


class State:
    def __init__(self):
        """
        needs to take in every "observation", after every "step"
        - the action passed into step might be a repeat of the previous action, that is already taken care of.

        - this class needs to construct the "current state"
        -> only "store" every 4th frame (but also store every 3rd to get the max pixel values...)

        -> method "get_state" returns eg: s1 = {x1, x2, x3, x4}
            where x1 has already taken the max between two frames
            and x1 to x2 has already taken into account the skipping

        https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        """

        n_frames = 4
        frame_h = 84
        frame_w = 84

        self.x_buffer = deque(maxlen=n_frames)  # stores x1, x2, x3, x4
        for _ in range(n_frames):
            # initialise with black frames (0)
            n = np.zeros(shape=(frame_h, frame_w))
            self.x_buffer.append(n)

        self.prev_frame = np.zeros(shape=(frame_h, frame_w))

        self.frame_count = 0
        self.frame_skip = 4

    def add_frame(self, frame):
        """ add a frame (74, 84) to the deque ONLY if we have skipped enough...

        """
        # increment frame count
        self.frame_count += 1

        # if frame count = 4 and frame skip = 4 then this is the frame we need to store
        if self.frame_count % self.frame_skip == 0:
            # but we need to look at the prev frame to get the max pixel values...
            curr_frame = np.maximum(frame, self.prev_frame)
            # now we store this frame in the deque
            self.x_buffer.append(curr_frame)

        elif (self.frame_count+1) % self.frame_skip == 0:
            # only need to store the frame just before we're about the sample the 4th frame...
            # set frame to prev_frame
            self.prev_frame = frame

    def plot_state(self):
        """ useful for visualising the state """
        fig = plt.figure(figsize=(15, 5), tight_layout=True)
        for i, f in enumerate(self.x_buffer):
            fig.add_subplot(1, 4, i+1)
            plt.imshow(f)
        plt.show()

    def get_state(self):
        """ returns s1 = {x1, x2, x3, x4}
            in the shape: (4, 84, 84)
            This gets called each iteration...
            as numpy array?
        """
        return np.stack(self.x_buffer)
