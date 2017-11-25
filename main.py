import os
import sys
import time

import numpy as np
import pygame
import tensorflow as tf
from pygame.locals import *

import utils
from din import DIN
# from envs.vizdoom.doom_game import DoomClass
import gym
from experience_replay.ER import ER
from params import *


class Trainer:
    def __init__(self):
        self.agent_buffer = ER(memory_size=1000,
                               state_channels=C_IN,
                               state_height=HEIGHT,
                               state_width=WIDTH,
                               action_dim=NUM_ACTIONS,
                               batch_size=BS,
                               history_length=1
                               )

        # self.env = DoomClass(scenario='envs/take_cover', timeout=1000, width=WIDTH, height=HEIGHT,
        #                      render=RENDER, labels_buffer=False, c_in=C_IN)

        self.env = gym.make("Breakout-v0")

        self.create_train_graph()

        self.saver = tf.train.Saver()

        self.expert_acc = 0.

        self.fake_acc = 0.

        self.reset = True

        self.saved_model = MODEL

        np.set_printoptions(precision=4, linewidth=160)

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

        self.x = tf.placeholder(tf.float32, in_shape, "x_fake")

        self.adv = tf.placeholder(tf.float32, [None, ], "advantage")

        self.y = tf.placeholder(tf.int32, [None, ])

        self.a_idx = tf.placeholder(tf.int32, [None, ], "action_index")

        self.model = DIN(num_actions=NUM_ACTIONS, is_training=True)

        # Discriminator training graph: expert label = [1, 0], fake label = [0, 1]

        self.expert_sequence = self.read_data(N_EPISODES)

        self.x_expert = tf.reshape(self.expert_sequence, [N_STEPS, N_FRAMES*C_IN, HEIGHT, WIDTH])

        self.a_logits, self.d_logits = self.model.forward(self.x, reuse=False)

        _, self.d_logits_ex = self.model.forward(self.x_expert, reuse=True)

        d_expert_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.zeros(N_STEPS, tf.int32), logits=self.d_logits_ex))

        d_fake_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.d_logits))

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.a_logits, labels=self.a_idx)

        self.pg_loss = BETA_PG * tf.reduce_mean(self.adv * neglogpac)

        self.entropy = BETA_ENT * tf.reduce_mean(utils.cat_entropy(self.a_logits))

        self.d_loss = BETA_DISC * (d_fake_loss + d_expert_loss)

        loss = self.pg_loss - self.entropy + self.d_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):

            gvs = opt.compute_gradients(loss)

            capped_gvs = [(tf.clip_by_value(grad, -MAX_GRAD, MAX_GRAD), var) for grad, var in gvs]

            self.g_norm, self.w_norm = utils.compute_mean_abs_norm(capped_gvs)

            self.grad_op = opt.apply_gradients(capped_gvs)

    def read_data(self, n_episodes):

        filename_queue = tf.train.string_input_producer(

            [("envs/breakout/data/ep_%d_frame.bin" % i) for i in range(n_episodes)], num_epochs=None)

        reader = tf.FixedLengthRecordReader(WIDTH*HEIGHT*C_IN*N_FRAMES*SKIP_FRAME)

        _, serialized_example = reader.read(filename_queue)

        x = tf.decode_raw(serialized_example, tf.uint8)

        x = tf.reshape(x, [N_FRAMES*SKIP_FRAME, C_IN, HEIGHT, WIDTH])

        x = tf.py_func(self.slice_frames, [x], tf.uint8)

        x = tf.reshape(x, [N_FRAMES, C_IN, HEIGHT, WIDTH])

        x = tf.cast(x, tf.float32)

        x = tf.train.shuffle_batch([x], batch_size=N_STEPS, capacity=2000, min_after_dequeue=1000)

        return x

    def init_lists(self):

        self.traj_x = []

        self.traj_actions = []

        self.traj_d_logits = []

        self.traj_a_logits = []

        self.traj_done = []

    def init_env(self):

        # self.env.game.new_episode()
        self.env.reset()

        for i in range(4):

            state, _, done, info = self.env.step(1)

        self.x_buffer = utils.Buffer(N_FRAMES, utils.get_frame(state, C_IN, HEIGHT, WIDTH))

        self.reset = False

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

            # self.env.game.make_action(action.tolist())
            state, _, done, info = self.env.step(action)

            # if self.env.game.is_episode_finished():
            if info['ale.lives'] < 5:
                return None

        return state

    def slice_frames(self, x):

        offset = np.random.randint(0, SKIP_FRAME, 1)[0]

        return x[offset::SKIP_FRAME]

    def human_input(self):
        MOVE_LEFT = [True, False]
        MOVE_RIGHT = [False, True]
        NULL_ACTION = [False, False]

        keys = pygame.key.get_pressed()
        a = NULL_ACTION
        if keys[K_LEFT]:
            a = MOVE_LEFT
        elif keys[K_RIGHT]:
            a = MOVE_RIGHT
        event = pygame.event.poll()
        if (event.type == pygame.KEYDOWN) and (event.key == pygame.K_LEFT):
            a = MOVE_LEFT
        elif (event.type == pygame.KEYDOWN) and (event.key == pygame.K_RIGHT):
            a = MOVE_RIGHT
        time.sleep(0.25)
        return np.asarray(a)

    def test(self):

        with tf.Session() as sess:

            self.load(sess)

            sess.run(self.model.set_validation_mode)

            # tf.train.start_queue_runners(sess)

            while True:

                if self.reset:

                    self.init_env()

                self.env.render()

                x = np.asarray(self.x_buffer).swapaxes(0, 1)

                a_logits, d_logits = sess.run([self.a_logits, self.d_logits], {self.x: x})

                a_logits = np.clip(a_logits, -MAX_A_LOGITS, MAX_A_LOGITS)

                a_t = utils.sample(a_logits[0], 1000)

                if HUMAN_MODE:

                    a_t = self.human_input()

                state = self.interact(a_t, SKIP_FRAME)

                if state is None:

                    self.reset = True

                    continue

                self.x_buffer.append(utils.get_frame(state, C_IN, HEIGHT, WIDTH))

                self.fake_acc = 0.9 * self.fake_acc + 0.1 * self.calc_accuracy(d_logits, 1)

                print("acc: %f" % (self.fake_acc))

    def train(self):

        with tf.Session() as sess:

            if MODEL is None:

                init_op = tf.global_variables_initializer()

                sess.run(init_op)

            else:

                self.load(sess)

            tf.train.start_queue_runners(sess)

            sess.run(self.model.set_training_mode)

            # self.init_agent()

            for ts in range(N_TRAIN_STEPS):

                # 1. interact with envs. for N_STEPS

                self.init_lists()

                for ns in range(N_STEPS):

                    if self.reset:

                        self.init_env()

                    self.env.render()

                    x = np.asarray(self.x_buffer).swapaxes(0, 1)

                    a_logits, d_logits = sess.run([self.a_logits, self.d_logits], {self.x: x})

                    a_logits = np.clip(a_logits, -MAX_A_LOGITS, MAX_A_LOGITS)

                    a_t = utils.sample(a_logits[0])

                    state = self.interact(a_t, SKIP_FRAME)

                    if state is None:

                        self.reset = True

                        if len(self.traj_done) > 0:

                            self.traj_done[-1] = True

                        continue

                    self.x_buffer.append(utils.get_frame(state, C_IN, HEIGHT, WIDTH))

                    self.traj_x.append(list(self.x_buffer))

                    self.traj_actions.append(a_t)

                    self.traj_d_logits.append(d_logits[0])

                    self.traj_a_logits.append(a_logits[0])

                    self.traj_done.append(False)

                # 2. define the advantage

                adv = np.asarray(self.traj_d_logits)[1:, 0] - np.asarray(self.traj_d_logits)[:-1, 0]

                adv = adv * (~np.asarray(self.traj_done[:-1]))

                action_counts = np.histogram(self.traj_actions, NUM_ACTIONS)[0]

                mean_a_logits = np.asarray(self.traj_a_logits).mean(axis=0)

                adv_vec = np.expand_dims(adv, 1) * utils.one_hot(self.traj_actions[:-1], NUM_ACTIONS)

                adv_vec = np.clip(adv_vec, -MAX_ADV, MAX_ADV)

                mean_adv = adv_vec.mean(axis=0)

                min_adv = adv_vec.min(axis=0)

                max_adv = adv_vec.max(axis=0)

                a_idx = np.asarray(self.traj_actions[:-1])

                # 3. train

                x = np.reshape(self.traj_x[:-1], [-1, N_FRAMES*C_IN, HEIGHT, WIDTH])

                y = np.ones(x.shape[0], np.int32)

                _, d_fake, d_expert, g_norm, w_norm, expert_seq, pg_loss, entropy, d_loss = sess.run([self.grad_op,
                                                                            self.d_logits,
                                                                            self.d_logits_ex,
                                                                            self.g_norm,
                                                                            self.w_norm,
                                                                            self.expert_sequence,
                                                                            self.pg_loss,
                                                                            self.entropy,
                                                                            self.d_loss
                                                                            ],
                                                                           {self.x: x,
                                                                            self.y: y,
                                                                            self.adv: adv,
                                                                            self.a_idx: a_idx,
                                                                            })

                self.fake_acc = 0.9 * self.fake_acc + 0.1 * self.calc_accuracy(d_fake, 1)

                self.expert_acc = 0.9 * self.expert_acc + 0.1 * self.calc_accuracy(d_expert, 0)

                if ts % SAVE_INTRVL == 0 and MODE == "train":

                        self.save(sess, ts)

                if ts % PRINT_INTRVL == 0:

                    print("iter: %5d, fake acc: %.3f, expert acc: %.3f, action_count: %s,"
                          "mean_a_logits %s, min_adv: %s, mean_adv: %s, max_adv: %s, w_norm: %f,"
                          "g_norm: %f, pg_loss: %.6f, entropy: %.2f, d_loss: %.2f" %
                          (ts, self.fake_acc, self.expert_acc, action_counts, mean_a_logits, min_adv, mean_adv, max_adv,
                           w_norm, g_norm, pg_loss, entropy, d_loss))

    def calc_accuracy(self, logits, label_idx):

        labels = np.zeros_like(logits)

        labels[:, label_idx] = 1

        correct = np.sum(logits * labels, 1) > np.sum(logits * (1-labels), 1)

        acc = np.sum(correct) / correct.shape[0]

        return acc

if __name__ == "__main__":
    trainer = Trainer()
    if MODE == "train":
        trainer.train()
    elif MODE == "test":
        pygame.init()
        pygame.display.set_mode((10, 10))
        trainer.test()
