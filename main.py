import tensorflow as tf
import numpy as np
from din import DIN
from params import *
import utils
import time
import os
import sys
from env.doom_game import DoomClass
# from experience_replay.rank_based import Experience
from experience_replay.replay_buffer import ReplayBuffer
from experience_replay.ER import ER
import pygame
from pygame.locals import *


class Trainer:
    def __init__(self):
        self.agent_buffer = ER(memory_size=20000,
                               state_channels=C_IN,
                               state_height=HEIGHT,
                               state_width=WIDTH,
                               action_dim=NUM_ACTIONS,
                               batch_size=BS,
                               history_length=N_FRAMES
                               )

        self.env = DoomClass(scenario='env/take_cover', timeout=1000, width=WIDTH, height=HEIGHT,
                             render=RENDER, labels_buffer=False, c_in=C_IN)

        self.create_train_graph()

        self.saver = tf.train.Saver()

        self.reset = True

        self.expert_acc = 0.

        self.fake_acc = 0.

        self.train_expert = True

        self.saved_model = MODEL

        self.MOVE_LEFT = [True, False]

        self.MOVE_RIGHT = [False, True]

        self.NULL_ACTION = [False, False]

    def create_test_graph(self):

        in_shape = [1, N_FRAMES * C_IN, HEIGHT, WIDTH]

        self.x = tf.placeholder(shape=in_shape, dtype=tf.float32, name="x")

        self.model = DIN(num_actions=NUM_ACTIONS, is_training=False)

        self.a_logits, self.d_logits = self.model.forward(self.x, reuse=False)

    def create_train_graph(self):

        opt = tf.train.AdamOptimizer(learning_rate=POLICY_LR)

        if IS_TRAINING:
            in_shape = [None, N_FRAMES * C_IN, HEIGHT, WIDTH]
        else:
            in_shape = [1, N_FRAMES * C_IN, HEIGHT, WIDTH]

        self.x = tf.placeholder(shape=in_shape, dtype=tf.float32, name="x")

        self.y = tf.placeholder(shape=(None, 2), dtype=tf.float32, name='y')

        self.model = DIN(num_actions=NUM_ACTIONS, is_training=True)

        # Discriminator training graph: expert label = [1, 0], fake label = [0, 1]

        self.expert_sequence = self.read_data(N_EPISODES)

        self.x_expert = tf.reshape(self.expert_sequence, [-1, C_IN * N_FRAMES, HEIGHT, WIDTH])

        self.a_logits, self.d_logits = self.model.forward(self.x, reuse=False)

        d_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.d_logits))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):

            grads_and_vars = opt.compute_gradients(d_loss)

            self.g_norm, self.w_norm = utils.compute_mean_abs_norm(grads_and_vars)

            self.d_grad_op = opt.apply_gradients(grads_and_vars)

        # Actor training graph
        self.advantage = tf.placeholder(shape=None, dtype=tf.float32)

        # self.actions = tf.placeholder(shape=None, dtype=tf.float32)

        # self.a_fake, self.d_fake = self.model.forward(self.x_fake, reuse=False)

        pi = tf.nn.softmax(self.a_logits)

        uniform_logits = tf.ones_like(self.a_logits)

        causal_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=uniform_logits, logits=self.a_logits)

        actor_utility = self.advantage * tf.reduce_sum(tf.log(pi) * self.actions, 1) + BETA * causal_entropy

        # grads_n_vars = opt.compute_gradients(loss=-actor_utility)

        # trainable_vars = tf.trainable_variables()

        # grad_vars = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in trainable_vars]

        # self.zero_grad = [grad_var.assign(tf.zeros_like(grad_var)) for grad_var in grad_vars]

        # accum_grad = [grad_vars[i].assign_add(gv[0]) for i, gv in enumerate(grads_n_vars)]
        # self.accum_grad = []
        # for i, gv in enumerate(grads_n_vars):
        #    if gv[0] is not None:
        #        self.accum_grad.append(grad_vars[i].assign_add(gv[0]))

        # self.apply_grad = opt.apply_gradients([(grad_vars[i], grad_n_var[1]) for i, grad_n_var in enumerate(grads_n_vars)])

        self.init_op = tf.global_variables_initializer()

    def read_data(self, n_episodes):
        filename_queue = tf.train.string_input_producer(
            [("env/take_cover/ep_%d_frame.bin" % i) for i in range(n_episodes)], num_epochs=None)

        reader = tf.FixedLengthRecordReader(WIDTH*HEIGHT*C_IN*N_FRAMES*SKIP_FRAME)

        _, serialized_example = reader.read(filename_queue)

        x = tf.decode_raw(serialized_example, tf.uint8)

        x = tf.reshape(x, [N_FRAMES*SKIP_FRAME, C_IN, HEIGHT, WIDTH])

        # x = tf.strided_slice(x, [0, 0, 0, 0], [-1, C_IN, HEIGHT, WIDTH], [SKIP_FRAME, 1, 1, 1])

        x = tf.cast(x, tf.float32) / 255.

        x = tf.train.shuffle_batch([x], batch_size=BS, capacity=2000, min_after_dequeue=1000)

        return x

    def init_agent(self):

        self.env.game.new_episode()

        state = self.env.game.get_state()

        self.traj_states = utils.init_list(utils.get_frame(state, C_IN, HEIGHT, WIDTH), N_FRAMES)

        self.traj_actions = utils.init_list([False] * (NUM_ACTIONS), N_FRAMES)

        self.traj_d_logits = utils.init_list(np.zeros((2,), dtype=np.float32), N_FRAMES)

    def save(self, sess, step):
        if self.saved_model is None:
            self.saved_model = CHECKPOINTS_DIR + time.strftime("%Y-%m-%d-%H-%M-%S")
            os.makedirs(self.saved_model)
            print('Created model: ' + self.saved_model)
        self.saver.save(sess, self.saved_model + '/model', global_step=step)

    def load(self, sess):
        ckpt = tf.train.get_checkpoint_state(MODEL)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(MODEL, ckpt_name))
            print('loaded model: %s-%s' % (MODEL, ckpt_name))
        else:
            print('unable to load model: %s' % MODEL)
            sys.exit(0)

    def interact(self, action, skip_frame):

        for _ in range(skip_frame):

            self.env.game.make_action(action.tolist())

            if self.env.game.is_episode_finished():

                return None

        return self.env.game.get_state()

    def read_expert(self, sess):

        x, x_seq = sess.run([self.x_expert, self.expert_sequence])

        x_sliced = []

        for example in x_seq:

            offset = np.random.randint(0, SKIP_FRAME, 1)[0]

            x_sliced.append(example[offset::SKIP_FRAME])

        x_array = np.asarray(x_sliced)

        return np.reshape(x_array, [BS, N_FRAMES*C_IN, HEIGHT, WIDTH])

    def human_input(self):
        keys = pygame.key.get_pressed()
        a = self.NULL_ACTION
        if keys[K_LEFT]:
            a = self.MOVE_LEFT
        elif keys[K_RIGHT]:
            a = self.MOVE_RIGHT
        event = pygame.event.poll()
        if (event.type == pygame.KEYDOWN) and (event.key == pygame.K_LEFT):
            a = self.MOVE_LEFT
        elif (event.type == pygame.KEYDOWN) and (event.key == pygame.K_RIGHT):
            a = self.MOVE_RIGHT
        time.sleep(0.75)
        return np.asarray(a)

    def test(self):

        with tf.Session() as sess:

            self.load(sess)

            sess.run(self.model.set_validation_mode)

            tf.train.start_queue_runners(sess)

            while True:

                if self.reset:

                    self.init_agent()

                    self.reset = False

                d_fake_ = 0.

                while True:

                    x = np.expand_dims(np.concatenate(self.traj_states[-N_FRAMES:], 0), 0)

                    # TODO: debugging - feeding expert examples
                    # x = self.read_expert(sess)
                    # x = np.expand_dims(x[10], 0)

                    a_fake, d_fake = sess.run([self.a_logits, self.d_logits], {self.x: x})

                    a_t = utils.sample(a_fake[0])

                    if HUMAN_MODE:

                        a_t = self.human_input()

                    # # TODO: debug
                    # a_t = np.asarray([False, False])

                    state = self.interact(a_t, SKIP_FRAME)

                    if state is None:

                        self.reset = True

                        break

                    self.traj_states.append(utils.get_frame(state, C_IN, HEIGHT, WIDTH))

                    self.traj_actions.append(a_t)

                    self.traj_d_logits.append(d_fake[0])



                    y = np.zeros((BS, 2), dtype=np.float32)

                    y[:, 1] = 1

                    acc = self.calc_accuracy(d_fake, y)

                    self.expert_acc = 0.9 * self.expert_acc + 0.1 * acc

                    adv = d_fake[0][0] - d_fake_

                    d_fake_ = d_fake[0][0]

                    print("acc: %f, advantage: %f, d_expert: %f" % (self.expert_acc, adv, d_fake_))


    def train(self):

        with tf.Session() as sess:

            if MODEL is None:

                sess.run(self.init_op)

            else:
                self.load(sess)

            tf.train.start_queue_runners(sess)

            sess.run(self.model.set_training_mode)

            for ts in range(N_TRAIN_STEPS):

                n = 0

                if self.reset:

                    self.init_agent()

                    self.reset = False

                while True:

                    x = np.expand_dims(np.concatenate(self.traj_states[-N_FRAMES:], 0), 0)

                    a_fake, d_fake = sess.run([self.a_logits, self.d_logits], {self.x: x})

                    a_t = utils.sample(a_fake[0])

                    state = self.interact(a_t, SKIP_FRAME)

                    if state is None:

                        self.reset = True

                        break

                    if n == N_STEPS:

                        break

                    n += 1

                    self.traj_states.append(utils.get_frame(state, C_IN, HEIGHT, WIDTH))

                    self.traj_actions.append(a_t)

                    self.traj_d_logits.append(d_fake[0])

                # print("d_t=%s, adv_t=%f" % (d_fake[0], self.traj_d_logits[-1][0] - self.traj_d_logits[-2][0]))

                # second pass: accumulate gradient, penalize low entropy

                # TODO: debug - disabled actor
                # sess.run(self.zero_grad)

                l = len(self.traj_states)

                for t in range(l-n, l-1):

                    self.agent_buffer.add(action=self.traj_actions[t],
                                          reward=None,
                                          next_state=self.traj_states[t],
                                          terminal=False)

                    # TODO: Debug - disabled actor
                    # adv_t = self.traj_d_logits[t + 1][0] - self.traj_d_logits[t][0]
                    #
                    # x_t = self.traj_states[t-N_FRAMES:t]
                    #
                    # action_t = self.traj_actions[t]
                    #
                    # sess.run(self.accum_grad, {self.advantage: adv_t, self.actions: action_t, self.x_fake: np.expand_dims(np.concatenate(x_t, 0), 0)})

                self.agent_buffer.add(action=self.traj_actions[-1],
                                      reward=None,
                                      next_state=self.traj_states[-1],
                                      terminal=True)

                # TODO: disabled actor
                # sess.run(self.apply_grad)

                # DISCRIMINATOR

                y = np.zeros((BS, 2), dtype=np.float32)

                if self.train_expert:

                    x = self.read_expert(sess)

                    # utils.plot_sequence(np.expand_dims(x[0], 1))

                    y[:, 0] = 1

                else:

                    x = self.agent_buffer.sample()[0]

                    # utils.plot_sequence(x[0])

                    x = np.reshape(x, [-1, N_FRAMES * C_IN, HEIGHT, WIDTH])

                    y[:, 1] = 1

                d, g_norm, w_norm = sess.run([self.d_grad_op, self.d_logits, self.g_norm, self.w_norm], {self.x: x, self.y: y})[1:]

                acc = self.calc_accuracy(d, y)

                if self.train_expert:

                    self.expert_acc = 0.9 * self.expert_acc + 0.1 * acc

                else:

                    self.fake_acc = 0.9 * self.fake_acc + 0.1 * acc

                if ts % SAVE_INTRVL == 0 and MODE == "train":
                        self.save(sess, ts)

                if ts % PRINT_INTRVL == 0:

                    print("iter: %d, fake acc: %f, expert acc: %f, g_norm: %f, w_norm: %f, agent_count: %d" %
                          (ts, self.fake_acc, self.expert_acc, g_norm, w_norm, self.agent_buffer.count))

                self.train_expert = not self.train_expert

    def calc_accuracy(self, logits, labels):

        correct = np.sum(logits * labels, 1) > np.sum(logits * (1-labels), 1)

        acc = np.sum(correct) / correct.shape[0]

        return acc

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
    if MODE == "train":
        trainer.train()
    elif MODE == "test":
        pygame.init()
        pygame.display.set_mode((10, 10))
        trainer.test()
