import numpy as np
import tty, termios, sys
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.misc


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def sample(p):
    x = np.random.multinomial(1, softmax(p)).astype(np.bool)
    return x # x[:-1]


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


def get_frame(x, c_in, h, w, scale=True):
    frame = x.screen_buffer
    frame = scipy.misc.imresize(frame, [h, w])
    if scale:
        frame = frame / 255.
    if c_in == 3:
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
    """"
    x: 4D tensor, (TIME x CHANNELS x HEIGHT x WIDTH)
    x_p: permuted tensor: "channels last" for display (TIME x HEIGHT x WIDTH x CHANNELS)
    """
    fig = plt.figure()
    axes = fig.add_subplot(111)
    cin = x.shape[1]
    # if cin == 1:
    #     x = x/255.
    x = np.transpose(x, [0, 2, 3, 1])
    im_obj = axes.imshow(np.squeeze(x[0]), cmap='gray')
    plt.show(block=False)
    i=0
    for xt in x:
        i+=1
        print("Frame:%d" % i)
        xt = np.flip(xt, axis=2)
        im_obj.set_data(np.squeeze(xt))
        im_obj.set_clim(vmin=xt.min(), vmax=xt.max())
        plt.pause(.25)


def compute_mean_abs_norm(grads_and_vars):
    tot_grad = 0
    tot_w = 0
    N = len(grads_and_vars)

    for g, w in grads_and_vars:
        if g is not None:
            tot_grad += tf.reduce_mean(tf.abs(g))
        tot_w += tf.reduce_mean(tf.abs(w))

    return tot_grad/N, tot_w/N