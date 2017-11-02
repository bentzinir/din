import tensorflow as tf
import numpy as np
from din import DIN
from params import *
import utils
from env.doom_game import DoomClass
from experience_replay.rank_based import Experience


def read_data():
    filename_queue = tf.train.string_input_producer(["env/take_cover/ep_" + str(i) + "_frame.bin" for i in range(1)], num_epochs=None)
    reader = tf.FixedLengthRecordReader()
    _, serialized_example = reader.read(filename_queue)
    example = tf.decode_raw(serialized_example, tf.uint8)
    x_batch, x_im_batch, ref_im_batch = tf.train.shuffle_batch([example], batch_size=10, capacity=2000, min_after_dequeue=1000)
    return x_batch, x_im_batch, ref_im_batch


def main():
    conf = {'size': 10000}

    agent_buffer = Experience(conf)


    # 1. create graph
    din = DIN(num_actions=2)


    env = DoomClass(scenario='env/take_cover', timeout=1000, width=320, height=240, render=True)


    opt = tf.train.AdamOptimizer(learning_rate=POLICY_LR)


    # Actor training graph
    trainable_vars = tf.trainable_variables()


    one_step_vars = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in trainable_vars]


    accumed_vars = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in trainable_vars]


    zero_accumed_op = [gv.assign(tf.zeros_like(gv)) for gv in accumed_vars]


    zero_one_step_grad = [gv.assign(tf.zeros_like(gv)) for gv in one_step_vars]


    d_fake_ = tf.placeholder(shape=(None, 2), dtype=tf.float32)


    if IS_TRAINING:
        in_shape = [None, N_FRAMES*C_IN, 240, 320]
    else:
        in_shape = [1, N_FRAMES*C_IN, 240, 320]

    x_fake = tf.placeholder(shape=in_shape, dtype=tf.float32, name="input")


    a_logits, d_fake = din.forward(x_fake)


    pi = tf.nn.softmax(a_logits)


    action = tf.multinomial(pi, 1)


    uniform_logits = tf.ones_like(a_logits)


    causal_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=uniform_logits, logits=a_logits)


    actor_loss = tf.reduce_mean(tf.log(pi[action]) + BETA * causal_entropy)


    advantage = d_fake - d_fake_


    one_step_gvs = opt.compute_gradients(loss=actor_loss)


    calc_one_step_grad = [one_step_vars[i].assign_add(gv[0]) for i, gv in enumerate(one_step_gvs)]


    score_one_step_grad = [one_step_vars[i].assign(advantage * one_step_vars[i]) for i, gv in enumerate(one_step_vars)]


    accum_grad = [accumed_vars[i].assign_add(gv[0]) for i, gv in enumerate(one_step_vars)]


    apply_grad = opt.apply_gradients([(accumed_vars[i], one_step_vars[1]) for i, gv in enumerate(one_step_vars)])


    # Discriminator training graph
    x_expert = read_data("bla")


    _, d_expert = din.forward(x_expert)

    d = tf.concat([d_expert, d_fake], axis=0)

    y = np.ones(x_expert.get_shape().as_list()[0])

    y_expert = tf.concat(values=[y, 1-y], axis=1)

    y_fake = tf.concat(values=[1-y, 1], axis=1)

    y = tf.concat([y_expert, y_fake])

    discriminator_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=d))

    discriminator_grad_op = opt.minimize(discriminator_loss)

    init_op = tf.global_variables_initializer()


    with tf.Session() as sess:
        sess.run(init_op)

        for i in range(N_TRAIN_STEPS):

            # train the discriminator
            # x_expert = expert_buffer.sample()

            x_fake = agent_buffer.sample()

            sess.run(discriminator_grad_op, {})

            # train the actor
            trajectory = [env.game.new_episode()]

            # set the environment at state s_t
            env.initialize(trajectory[-1])

            sess.run(zero_accumed_op)
            for _ in range(N_ACCUM_STEPS):
                while not env.game.is_episode_finished():
                    sess.run(zero_one_step_grad)
                    # sample action from state s, load the gradient to gradient_vars
                    a, d_, _ = sess.run([action, d_fake, calc_one_step_grad], {din.input_tensor: trajectory[-N_FRAMES:]})

                    s_t = env.game.get_state()

                    env.game.make_action(a)

                    trajectory.append(s_t)

                    sess.run(score_one_step_grad, {din.input_tensor: trajectory[-N_FRAMES:], d_fake_: d_})

                    sess.run(accum_grad)

            sess.run(apply_grad)

            agent_buffer.add(trajectory)


if __name__ == "__main__":
    main()