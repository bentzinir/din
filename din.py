from nn import NN
from params import *
import utils


expert_buffer = ER()

din = NN()


# 2. sample a starting point: s = [s_t, s_tm1, s_tm2, ...s_tmk]
s = expert_buffer.sample()


# 3. set the environment at state s_t
env.initialize(s[0])


# 4. train the policy
d_actions, d_logits_ = din(s)
for i in range(N_STEPS):
    # 4.1 play action
    a = utils.sample(BS, d_actions)

    # 4.2 step
    s_t = env.step(a)

    # 4.3 update state buffer
    s = [s_t, s[:end - 1]]

    # 4.4 compute action / discrimination predictions
    d_actions, d_logits = din(s)

    # 4.5 compute policy gradient
    din.accum_gradient(d_logits_, d_logits)

    # 4.6 update d_logits
    d_logits_ = d_logits

# 4.7 apply the gradient
din.apply_gradient()


# 5. train the discriminator
