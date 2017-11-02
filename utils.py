import numpy as np
import tty, termios, sys


def sample(n, p, mode='discrete'):
    if mode == 'discrete':
        return np. random.multinomial(n, p)
    else:  # mode = 'continuous
        return p + np.random.normal(p, 1, n)


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
