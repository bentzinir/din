import gym
import time
import matplotlib.pyplot as plt

env = gym.make("Breakout-v0")


obs = env.reset()
env.step(1)

for i in range(10):
    env.step(2)
    env.render()
    s = env.env.clone_full_state()
    s_ref = env.env._get_obs()

a=1
for i in range(10):
    env.step(2)
    env.render()
    # time.sleep(0.2)

env.env.restore_full_state(s)
env.step(0)
s_copy = env.env._get_obs()

fig, axarr = plt.subplots(2)

axarr[0].imshow(s_ref, interpolation='none')
axarr[1].imshow(s_copy, interpolation='none')

env.step(0)
env.render()



a = 1
# parent()



