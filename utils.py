import numpy as np
import tty, termios, sys
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.misc


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def sample(p, tau=100):
    # v1
    # p = p * tau
    # x = np.random.multinomial(1, softmax(p)).astype(np.bool)
    # return np.argmax(x)
    # v2
    # p = np.log(p) / temperature
    # dist = np.exp(p) / np.sum(np.exp(p))
    # choices = range(len(p))
    # choice_idx = np.random.choice(choices, p=dist)
    # x = np.asarray([False] * len(p))
    # x[choice_idx] = True
    # return x
    # v3
    logits= p * tau
    noise = np.random.uniform(size=len(p))
    return np.argmax(logits - np.log(-np.log(noise)))


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


def one_hot(x, n):
    x = np.asarray(x)
    b = np.zeros((x.size, n))
    b[np.arange(x.size), x] = 1
    return b


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def get_frame(x, c_in, h, w, scale=True, resize=True):
    if resize:
        x = scipy.misc.imresize(x, [h, w])
    if x.shape[-1] == 3:
        x = rgb2gray(x).astype(np.uint8)
    x = np.expand_dims(x, 0)
    return x


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


def gradient_norm(loss, variables):

    grad = tf.gradients(loss, variables)

    g_norm = 0

    for g in grad:

        if g is not None:
            g_norm += tf.reduce_sum(tf.square(g))

    return g_norm


def compute_mean_abs_norm(grads_and_vars):
    tot_grad = 0
    tot_w = 0
    N = len(grads_and_vars)

    for g, w in grads_and_vars:
        if g is not None:
            tot_grad += tf.reduce_mean(tf.abs(g))
        tot_w += tf.reduce_mean(tf.abs(w))

    return tot_grad/N, tot_w/N


def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


def acc(logits, label_idx):
    correct = tf.greater_equal(tf.slice(logits, [0, label_idx], [-1, 0]),
                               tf.slice(logits, [0, 1 - label_idx], [-1, 0]))

    correct = tf.cast(correct, tf.float32)
    wrong = 1 - correct
    return tf.reduce_sum(correct) / (tf.reduce_sum(correct)+tf.reduce_sum(wrong))
