import numpy as np
import tty, termios, sys


def sample(p):
    return np.random.multinomial(1, p).astype(np.bool)


def getch():
    # Returns a single character from standard input
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def get_frame(x, c_in):
    frame = x.screen_buffer / 255.
    frame = np.transpose(frame, [2, 0, 1])
    if c_in == 1:
        frame = np.expand_dims(frame, 0)
    return frame


class Buffer(list):
    def __init__(self, length, x=None):
        self.length = length
        if x is not None:
            for _ in range(self.length):
                self.append(x)

    def append(self, item):
        list.append(self, item)
        if len(self) > self.length:
            self[:1] = []


def init_list(x, length=None):
    l = list()
    if length is not None:
        for _ in range(length):
            l.append(x)
    return l


def plot_sequence(x):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    axes = fig.add_subplot(111)
    x = 255 * x
    x = np.transpose(x, [0, 2, 3, 1]).astype(np.uint8)
    im_obj = axes.imshow(x[0])
    plt.show(block=False)
    i=0
    for xt in x:
        i+=1
        print("Frame:%d" % i)
        xt = np.flip(xt, axis=2)
        im_obj.set_data(xt)
        im_obj.set_clim(vmin=xt.min(), vmax=xt.max())
        plt.pause(.1)
