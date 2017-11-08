import tensorflow as tf
import numpy as np
from din import DIN
from params import *
import utils
from env.doom_game import DoomClass
# from experience_replay.rank_based import Experience
from experience_replay.replay_buffer import ReplayBuffer
from experience_replay.ER import ER


class Trainer:
    def __init__(self):
        self.agent_buffer = ER(memory_size=100,
                               state_channels=C_IN,
                               state_height=HEIGHT,
                               state_width=WIDTH,
                               action_dim=NUM_ACTIONS,
                               batch_size=BS,
                               history_length=N_FRAMES
                               )

        self.env = DoomClass(scenario='env/take_cover', timeout=1000, width=320, height=240, render=True, c_in = C_IN)

        self.create_train_graph()

        self.reset = True

    def create_train_graph(self):

        self.model = DIN(num_actions=2)

        opt = tf.train.AdamOptimizer(learning_rate=POLICY_LR)

        # Actor training graph
        self.advantage = tf.placeholder(shape=None, dtype=tf.float32)

        self.actions = tf.placeholder(shape=None, dtype=tf.float32)

        if IS_TRAINING:
            in_shape = [None, N_FRAMES * C_IN, HEIGHT, WIDTH]
        else:
            in_shape = [1, N_FRAMES * C_IN, HEIGHT, WIDTH]

        self.x_fake = tf.placeholder(shape=in_shape, dtype=tf.float32, name="input")

        self.y_fake = tf.placeholder(shape=(None, 2), dtype=tf.float32)

        self.a_logits, self.d_fake = self.model.forward(self.x_fake, reuse=False)

        pi = tf.nn.softmax(self.a_logits)

        uniform_logits = tf.ones_like(self.a_logits)

        causal_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=uniform_logits, logits=self.a_logits)

        actor_utility = self.advantage * tf.reduce_sum(tf.log(pi) * self.actions, 1) + BETA * causal_entropy

        grads_n_vars = opt.compute_gradients(loss=-actor_utility)

        trainable_vars = tf.trainable_variables()

        grad_vars = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in trainable_vars]

        self.zero_grad = [grad_var.assign(tf.zeros_like(grad_var)) for grad_var in grad_vars]

        # accum_grad = [grad_vars[i].assign_add(gv[0]) for i, gv in enumerate(grads_n_vars)]
        self.accum_grad = []
        for i, gv in enumerate(grads_n_vars):
            if gv[0] is not None:
                self.accum_grad.append(grad_vars[i].assign_add(gv[0]))

        self.apply_grad = opt.apply_gradients([(grad_vars[i], grad_n_var[1]) for i, grad_n_var in enumerate(grads_n_vars)])

        # Discriminator training graph
        # expert label = [1, 0], fake label = [0, 1]

        self.expert_sequence = self.read_data()

        self.x_expert = tf.reshape(self.expert_sequence, [-1, C_IN*N_FRAMES, HEIGHT, WIDTH])

        _, d_expert = self.model.forward(self.x_expert)

        ones_vec = tf.ones(shape=(self.x_expert.get_shape().as_list()[0], 1))

        y_expert = tf.concat(values=[ones_vec, 1 - ones_vec], axis=1)

        d_expert_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_expert, logits=d_expert))

        self.d_expert_grad_op = opt.minimize(d_expert_loss)

        d_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_fake, logits=self.d_fake))

        self.d_fake_grad_op = opt.minimize(d_fake_loss)

        self.init_op = tf.global_variables_initializer()

    def read_data(self):
        filename_queue = tf.train.string_input_producer(
            [("env/take_cover/ep_%d_frame.bin" % i) for i in range(1)], num_epochs=None)

        reader = tf.FixedLengthRecordReader(WIDTH*HEIGHT*C_IN*N_FRAMES*SKIP_FRAME)

        _, serialized_example = reader.read(filename_queue)

        x = tf.decode_raw(serialized_example, tf.uint8)

        x = tf.reshape(x, [N_FRAMES*SKIP_FRAME, C_IN, HEIGHT, WIDTH])

        x = tf.strided_slice(x, [0, 0, 0, 0], [-1, C_IN, HEIGHT, WIDTH], [SKIP_FRAME, 1, 1, 1])

        x = tf.cast(x, tf.float32)/255

        x = tf.train.shuffle_batch([x], batch_size=BS, capacity=200, min_after_dequeue=100)

        return x

    def init_agent(self):

        self.env.game.new_episode()

        state = self.env.game.get_state()

        self.traj_states = utils.init_list(utils.get_frame(state, C_IN), N_FRAMES)

        self.traj_actions = utils.init_list([False] * NUM_ACTIONS, N_FRAMES)

        self.traj_d_logits = utils.init_list(np.zeros((1, 2), dtype=np.float32), N_FRAMES)

    def interact(self, action, skip_frame):

        for _ in range(skip_frame):

            self.env.game.make_action(action.tolist())

            if self.env.game.is_episode_finished():

                return None

        return self.env.game.get_state()

    def train(self):

        with tf.Session() as sess:

            sess.run(self.init_op)

            tf.train.start_queue_runners(sess)

            sess.run(self.model.set_training_mode)

            for ts in range(N_TRAIN_STEPS):

                n = 0

                if self.reset:

                    self.init_agent()

                    self.reset = False

                while True:

                    x = np.expand_dims(np.concatenate(self.traj_states[-N_FRAMES:], 0), 0)

                    a_logits, d_t = sess.run([self.a_logits, self.d_fake], {self.x_fake: x})

                    a_t = utils.sample(a_logits[0])

                    state = self.interact(a_t, SKIP_FRAME)

                    n += 1

                    if state is None:

                        self.reset = True

                        break

                    if n == N_STEPS:

                        break

                    self.traj_states.append(utils.get_frame(state, C_IN))

                    self.traj_actions.append(a_t)

                    self.traj_d_logits.append(d_t[0])

                # second pass: accumulate gradient, penalize low entropy

                sess.run(self.zero_grad)

                for t in range(n-2):

                    self.agent_buffer.add(action=self.traj_actions[t],
                                          reward=None,
                                          next_state=self.traj_states[t + 1],
                                          terminal=False)

                    adv_t = self.traj_d_logits[t + 1][0] - self.traj_d_logits[t][0]

                    x_t = self.traj_states[t:t + N_FRAMES]

                    action_t = self.traj_actions[t]

                    sess.run(self.accum_grad, {self.advantage: adv_t, self.actions: action_t, self.x_fake: np.expand_dims(np.concatenate(x_t, 0), 0)})

                self.agent_buffer.add(action=self.traj_actions[n - 1],
                                      reward=None,
                                      next_state=self.traj_states[n - 1],
                                      terminal=True)

                sess.run(self.apply_grad)

                # DISCRIMINATOR
                x_fake = self.agent_buffer.sample()[0]

                x_fake = np.reshape(x_fake, [-1, N_FRAMES*C_IN, HEIGHT, WIDTH])

                y_fake = np.zeros((BS, 2), dtype=np.float32)

                y_fake[:, 1] = 1

                sess.run(self.d_fake_grad_op, {self.x_fake: x_fake, self.y_fake: y_fake})

                # expert_seq, x_exp = sess.run([self.expert_sequence, self.x_expert])

                sess.run(self.d_expert_grad_op)

    def collect_experience(self):
        if self.env.game.is_episode_finished():
            self.env.game.new_episode()
            state = self.env.game.get_state()
            x = utils.Buffer(N_FRAMES, state)
        for j in range(1000):

            a_logits, d_fake = self.model.forward(self.x_fake, reuse=False)

            r = self.env.game.make_action(a)

            a_logits, d_t, _ = sess.run([a_logits, d_fake], {self.x_fake: traj_states[-N_FRAMES:]})

            a_t = utils.sample(a_logits)

            self.env.game.make_action(a_t)

            traj_states.append(self.env.game.get_state())


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
