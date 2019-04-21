import tensorflow as tf


class NoisyBatchNormalization(tf.keras.layers.BatchNormalization):

    def __init__(self, alpha=0.01, *args, **kwargs):
        tf.keras.layers.BatchNormalization.__init__(self, *args, **kwargs)
        self.alpha = alpha

    def call(self, inputs, training=None):
        input_shape = tf.keras.backend.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        print(reduction_axes, self.axis[0])
        del reduction_axes[self.axis[0]]


        def broadcast_mean_var(x, ndim):
            if ndim == 2:
                return tf.reshape(x, shape=[1,-1])
            elif ndim == 4:
                return tf.reshape(x, shape=[1,1,1,-1])
            else:
                raise NotImplementedError

        if training:
            mean, var = tf.nn.moments(inputs,axes=reduction_axes)
            self.moving_mean.assign(momentum * self.moving_mean + (1 - momentum) * mean)
            self.moving_variance.assign(momentum * self.moving_var + (1 - momentum) * var)
        else:
            mean, var = self.moving_mean, self.moving_variance

        mean = broadcast_mean_var(mean, ndim)
        var = broadcast_mean_var(var, ndim)
        norm = (inputs - mean) / tf.sqrt(var + self.epsilon)
        noise = tf.random.normal(input_shape)
        
        if training:
            norm_noise = norm + self.alpha * noise
        else:
            norm_noise = norm

        return self.gamma * norm_noise + self.beta

