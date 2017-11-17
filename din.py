import tensorflow as tf
from tensorflow.python.ops import init_ops
from params import *

torch_conv_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1 / 3.0, mode='FAN_IN', uniform=True)
gamma_initializer = init_ops.random_uniform_initializer
batch_initializer = {'gamma': init_ops.random_uniform_initializer}


class DIN:
    def __init__(self, num_actions, n_frames=4, num_layers=8, is_training=True, pool_func=tf.layers.max_pooling2d):

        self.num_actions = num_actions
        self.inplanes = 64
        self.n_frames = n_frames
        self.num_layers = num_layers

        self.is_training = tf.Variable(is_training, dtype=tf.bool, name='is_training', trainable=False)

        self.pool_func = pool_func
        self.set_training_mode = self.is_training.assign(True)
        self.set_validation_mode = self.is_training.assign(False)

    def forward(self, x, reuse=True):
        with tf.variable_scope("din") as scope:
            if reuse:
                scope.reuse_variables()

            # x = (?, 64, in_res, in_res)
            x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=2, padding='SAME',
                                 kernel_initializer=torch_conv_initializer, bias_initializer=torch_conv_initializer,
                                 data_format="channels_first")

            if BN:
                x = tf.layers.batch_normalization(x, epsilon=0.00001, axis=1, training=self.is_training)

            x = tf.nn.relu(x)

            # (?, 64, in_res, in_res)
            x = self._residual(x, 16, 'layer1')
            x = self.pool_func(x, 2, strides=2, data_format="channels_first")
            # x = tf.layers.max_pooling2d(x, 2, strides=2, data_format="channels_first")
            # (?, 64, in_res/2, in_res/2)
            x = self._residual(x, 16, 'layer2')
            x = self._residual(x, 16, 'layer3')
            x = tf.layers.max_pooling2d(x, 2, strides=2, data_format="channels_first")
            # (?, 64, in_res/4, in_res/4)
            x = self._residual(x, 32, 'layer4')
            x = self._residual(x, 32, 'layer5')
            x = tf.layers.max_pooling2d(x, 2, strides=2, data_format="channels_first")
            # (?, 64, in_res/8, in_res/8)
            # x = self._residual(x, 64, 'layer6')
            # x = self._residual(x, 64, 'layer7')
            x = tf.contrib.layers.flatten(x)
            x = self._affine(x, 256, activation=self.lrelu)
            # x = self._affine(x, 128, activation=self.lrelu)

            a_logits = self._affine(x, self.num_actions, bn=False)
            d_logits = self._affine(x, 2, bn=False)

        return a_logits, d_logits

    def lrelu(self, x, alpha=0.2):
        return tf.maximum(x, alpha * x)

    def _residual(self, input_tensor, planes, name, blocks=1, stride=1):
        with tf.variable_scope(name):

            residual = input_tensor

            inplanes = residual.get_shape().as_list()[1]

            mid_channels = int(inplanes/2)

            downsample = None

            if stride != 1 or inplanes != planes:
                downsample = lambda y: tf.layers.conv2d(y, planes, data_format="channels_first",
                                                        strides=stride, kernel_size=1)

            if BN:
                x = tf.layers.batch_normalization(residual, epsilon=0.00001, axis=1, training=self.is_training)
            else:
                x = residual

            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, mid_channels, kernel_size=1, data_format="channels_first")

            if BN:
                x = tf.layers.batch_normalization(x, epsilon=0.00001, axis=1, training=self.is_training)

            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, mid_channels, kernel_size=3, strides=stride, data_format="channels_first",
                                 padding='Same', use_bias=True)

            if BN:
                x = tf.layers.batch_normalization(x, epsilon=0.00001, axis=1, training=self.is_training)

            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, planes, kernel_size=1, data_format="channels_first",)

            if downsample is not None:
                residual = downsample(input_tensor)

            x += residual

            return x

    def _bottleneck(self, midChannels, input_tensor, stride=1, downsample=None):
        with tf.variable_scope("Bottleneck"):
            residual = input_tensor

            if BN:
                x = tf.layers.batch_normalization(input_tensor, epsilon=0.00001, axis=1, training=self.is_training)
            else:
                x = input_tensor

            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, midChannels, kernel_size=1, data_format="channels_first")

            if BN:
                x = tf.layers.batch_normalization(x, epsilon=0.00001, axis=1, training=self.is_training)

            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, midChannels, kernel_size=3, strides=stride, data_format="channels_first",
                                 padding='Same', use_bias=True,)

            if BN:
                x = tf.layers.batch_normalization(x, epsilon=0.00001, axis=1, training=self.is_training)

            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, 2 * midChannels, kernel_size=1, data_format="channels_first")

            if downsample is not None:
                residual = downsample(input_tensor)

            x += residual
        return x

    def _affine(self, x, n_feats, activation=None, bn=True):
        x = tf.layers.dense(x, n_feats)
        if bn:
            if BN:
                x = tf.layers.batch_normalization(x, epsilon=0.00001, axis=1, training=self.is_training)

        if activation is not None:
            x = activation(x)
        return x