import cv2
import numpy as np
from collections import deque


class CvFps(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv2.getTickCount()
        self._freq = 1000.0 / cv2.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv2.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded


def draw_fps(image, fps):
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def streamer_view(main, sub):
    # Resize image2 to take up around 30% of the screen
    sub = cv2.resize(sub, (int(main.shape[1]*0.3), int(main.shape[0]*0.3)))
    # Create a black canvas to overlay image2 on top of image1
    canvas = np.zeros_like(main)

    # Copy image1 onto the canvas
    canvas[:main.shape[0], :main.shape[1]] = main

    # Specify the offset of image2 from the bottom left corner of the canvas
    offset = 10

    # Copy image2 onto the bottom left corner of the canvas
    canvas[canvas.shape[0]-sub.shape[0]-offset:canvas.shape[0] -
           offset, offset:sub.shape[1]+offset] = sub

    # Draw a black border around image2 using cv2.copyMakeBorder
    border_size = 5  # Specify the size of the border
    constant = cv2.copyMakeBorder(sub, border_size, border_size, border_size,
                                  border_size, cv2.BORDER_CONSTANT, None, value=(0, 0, 0))

    # Calculate the new border size based on the shape of the constant image
    new_border_size = int((constant.shape[1] - sub.shape[1]) / 2)
    canvas[canvas.shape[0]-sub.shape[0]-offset-border_size:canvas.shape[0]-offset +
           border_size, offset-new_border_size:offset+sub.shape[1]+new_border_size] = constant
    return canvas
