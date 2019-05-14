import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
import numpy as np


class NoisyBatchNormalization(tf.keras.layers.BatchNormalization):

    def __init__(self, alpha=0.01, p=0.25, *args, **kwargs):
        tf.keras.layers.BatchNormalization.__init__(self, *args, **kwargs)
        self.alpha = alpha
        self.p = p

    def call(self, inputs, training=None, alpha=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        input_shape = tf.keras.backend.int_shape(inputs)
        
        # inputs = tf.cast(inputs, tf.float64)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis[0]]

        # print(reduction_axes)
        # inputs = tf.cast(inputs, tf.float64)


        def broadcast_mean_var(x, ndim):
            if ndim == 2:
                return tf.reshape(x, shape=[1,-1])
            elif ndim == 4:
                return tf.reshape(x, shape=[1,1,1,-1])
            else:
                raise NotImplementedError

        def mean_var_eval():
            mean, var = self.moving_mean, self.moving_variance
            return [mean, var]

        def mean_var_train():
            mean, var = tf.nn.moments(inputs,axes=reduction_axes)
            return [mean, var]
        
        mean, var = tf_utils.smart_cond(training, mean_var_train, mean_var_eval)
        
        self.moving_mean.assign(self.momentum * self.moving_mean + (1. - self.momentum) * mean)
        self.moving_variance.assign(self.momentum * self.moving_variance + (1. - self.momentum) * var)

        mean = broadcast_mean_var(mean, ndim)
        var = broadcast_mean_var(var, ndim)
        # print(inputs, mean, var, self.epsilon)
        # norm = tf.cast((inputs - mean), tf.float64) / tf.sqrt(tf.cast(var + self.epsilon, tf.float64))
        norm = (inputs - mean) / tf.sqrt(var + self.epsilon)
        noise = tf.random.normal(tf.shape(inputs))
        mask = tf.cast(tf.random.uniform(tf.shape(inputs)) < self.p, tf.float32)
        # norm = tf.cast(norm, tf.float32)
        
        if alpha is None:
            norm_noise = tf_utils.smart_cond(
                    training,
                    lambda: norm + self.alpha * noise * mask,
                    lambda: norm
                )
        else:

            norm_noise = norm + alpha * noise * mask

        return self.gamma * norm_noise + self.beta 



class BetterNoisyBatchNormalization(tf.keras.layers.BatchNormalization):

    def __init__(self, alpha=0.01, p=0.25, *args, **kwargs):
        tf.keras.layers.BatchNormalization.__init__(self, *args, **kwargs)
        self.alpha = alpha
        self.p = p

    def call(self, inputs, training=None, alpha=None, p=None):
        normal_out = super().call(inputs, training)

        def inject_noise(inp):
            noise = tf.random.normal(tf.shape(inputs))
            if p is None:
                mask = tf.cast(tf.random.uniform(tf.shape(inputs)) < self.p, tf.float32)
            else:
                mask = tf.cast(tf.random.uniform(tf.shape(inputs)) < p, tf.float32)

            return noise * mask

        if alpha is None:
            output = tf_utils.smart_cond(
                    training,
                    lambda: normal_out + self.alpha * self.gamma * inject_noise(normal_out),
                    lambda: normal_out
                )
        else:
            output = normal_out + alpha * self.gamma * inject_noise(normal_out)

        return output

