import tensorflow as tf


def sinkhorn_loop_fn(x,
                     step,
                     iterations):
    """Calculate the result of applying the Sinkhorn Operator
    to a permutation matrix in log space

    Arguments:

    x: tf.Tensor
        a permutation matrix in log space that will be
        processed with the sinkhorn operator
    step: tf.Tensor
        the current number of iterations of the Sinkhorn operator
        that have been applied
    iterations: tf.Tensor
        the total number of iterations of the Sinkhorn operator
        to apply to the data matrix

    Returns:

    x: tf.Tensor
        a permutation matrix in log space that has been
        processed with the sinkhorn operator
    step: tf.Tensor
        the current number of iterations of the Sinkhorn operator
        that have been applied
    iterations: tf.Tensor
        the total number of iterations of the Sinkhorn operator
        to apply to the data matrix"""

    x = tf.math.log_softmax(x, axis=-2)
    x = tf.math.log_softmax(x, axis=-1)
    return x, step + 1, iterations


def sinkhorn_cond_fn(x,
                     step,
                     iterations):
    """Calculate the result of applying the Sinkhorn Operator
    to a permutation matrix in log space

    Arguments:

    x: tf.Tensor
        a permutation matrix in log space that will be
        processed with the sinkhorn operator
    step: tf.Tensor
        the current number of iterations of the Sinkhorn operator
        that have been applied
    iterations: tf.Tensor
        the total number of iterations of the Sinkhorn operator
        to apply to the data matrix

    Returns:

    condition: tf.Tensor
        a boolean that determines if the loop that applies
        the Sinkhorn Operator should exit"""

    return tf.less(step, iterations)


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    tf.TensorSpec(shape=None, dtype=tf.int32)])
def sinkhorn(x,
             iterations):
    """Calculate the result of applying the Sinkhorn Operator
    to a permutation matrix in log space

    Arguments:

    x: tf.Tensor
        a permutation matrix in log space that will be
        processed with the sinkhorn operator
    iterations: tf.Tensor
        the total number of iterations of the Sinkhorn operator
        to apply to the data matrix

    Returns:

    x: tf.Tensor
        a permutation matrix in log space that has been
        processed with the sinkhorn operator"""

    args = [x, tf.constant(0, dtype=tf.int32), iterations]
    return tf.while_loop(
        sinkhorn_cond_fn, sinkhorn_loop_fn, args)[0]


class Sinkhorn(tf.keras.layers.Layer):

    def __init__(self,
                 iterations=20):
        """Calculate the result of applying the Sinkhorn Operator
        to a permutation matrix in log space

        Arguments:

        iterations: tf.Tensor
            the total number of iterations of the Sinkhorn operator
            to apply to the data matrix"""
        super(Sinkhorn, self).__init__()

        self.iterations = iterations

    def call(self, inputs, **kwargs):
        """Runs a forward pass on a pointer network that generates
        permutation matrices in log space

        Arguments:

        inputs: TransformerInput
            a dataclass instance that contains queries, keys
            and values along with masks

        Returns:

        outputs: tf.Tensor
            a permutation matrix in log space that has the same shape
            as the transformer attention weights"""

        # apply the sinkhorn operator
        return tf.exp(sinkhorn(inputs, tf.constant(self.iterations)))

    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process

        Returns:

        config: dict
            a dictionary that contains all parameters to the
            layers base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict(iterations=self.iterations)

        base_config = super(Sinkhorn, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))
