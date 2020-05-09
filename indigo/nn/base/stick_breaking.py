import tensorflow as tf


def stick_breaking_loop_fn(x,
                           x_mask,
                           b,
                           step):
    """Calculate the result of applying the Stick Breaking Operator
    to a permutation matrix in log space

    Arguments:

    x: tf.Tensor
        a permutation matrix that will be generated using the stick
        breaking procedure
    x_mask: tf.Tensor
        a mask that specifies which elements of the permutation
        matrix are allowed to be non-zero
    b: tf.Tensor
        a permutation matrix in logistic space that will be
        processed with the Stick Breaking operator
    step: tf.Tensor
        the current number of iterations of the Stick Breaking operator
        that have been applied=

    Returns:

    x: tf.Tensor
        a permutation matrix that will be generated using the stick
        breaking procedure
    x_mask: tf.Tensor
        a mask that specifies which elements of the permutation
        matrix are allowed to be non-zero
    b: tf.Tensor
        a permutation matrix in logistic space that will be
        processed with the Stick Breaking operator
    step: tf.Tensor
        the current number of iterations of the Stick Breaking operator
        that have been applied"""

    m = tf.math.floordiv(step, tf.shape(b)[2])
    n = tf.math.floormod(step, tf.shape(b)[2])
    N = tf.shape(b)[2]

    max_future_vals = tf.maximum(0., x_mask[
        :, m, (n + 1):] - tf.reduce_sum(x[:, :m, (n + 1):], axis=1))
    max_future_mass = tf.reduce_sum(max_future_vals, axis=1)

    lower_bound = x_mask[:, m, n] * tf.maximum(
        0.,
        1. - tf.reduce_sum(x[:, m, :n], axis=1) - max_future_mass)
    upper_bound = x_mask[:, m, n] * tf.minimum(
        1. - tf.reduce_sum(x[:, m, :n], axis=1),
        1. - tf.reduce_sum(x[:, :m, n], axis=1))

    p = lower_bound + b[:, m, n] * (upper_bound - lower_bound)
    p = p[:, tf.newaxis, tf.newaxis]

    i, j = tf.meshgrid(tf.range(N), tf.range(N), indexing='ij')
    mask = tf.logical_and(
        tf.equal(i, [[m]]), tf.equal(j, [[n]]))[tf.newaxis]

    return tf.where(mask, p, x), x_mask, b, step + 1


def stick_breaking_cond_fn(x,
                           x_mask,
                           b,
                           step):
    """Calculate the result of applying the Stick Breaking Operator
    to a permutation matrix in log space

    Arguments:

    x: tf.Tensor
        a permutation matrix that will be generated using the stick
        breaking procedure
    x_mask: tf.Tensor
        a mask that specifies which elements of the permutation
        matrix are allowed to be non-zero
    b: tf.Tensor
        a permutation matrix in logistic space that will be
        processed with the Stick Breaking operator
    step: tf.Tensor
        the current number of iterations of the Stick Breaking operator
        that have been applied

    Returns:

    x: tf.Tensor
        a permutation matrix that will be generated using the stick
        breaking procedure
    x_mask: tf.Tensor
        a mask that specifies which elements of the permutation
        matrix are allowed to be non-zero
    b: tf.Tensor
        a permutation matrix in logistic space that will be
        processed with the Stick Breaking operator
    step: tf.Tensor
        the current number of iterations of the Stick Breaking operator
        that have been applied"""

    return tf.less(step, tf.square(tf.shape(b)[2]))


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)])
def stick_breaking(x,
                   x_mask):
    """Calculate the result of applying the Stick Breaking Operator
    to a permutation matrix in logistic space

    Arguments:

    x: tf.Tensor
        a permutation matrix in log space that will be
        processed with the Stick Breaking operator
    x_mask: tf.Tensor
        a mask that specifies which elements of the permutation
        matrix are allowed to be non-zero

    Returns:

    x: tf.Tensor
        a permutation matrix in log space that has been
        processed with the Stick Breaking operator"""

    args = [tf.zeros_like(x), x_mask, x, tf.constant(0, dtype=tf.int32)]
    return tf.while_loop(
        stick_breaking_cond_fn, stick_breaking_loop_fn, args)[0]


class StickBreaking(tf.keras.layers.Layer):

    def __init__(self):
        """Calculate the result of applying the Stick Breaking Operator
        to a permutation matrix in log space"""
        super(StickBreaking, self).__init__()

    def call(self, inputs, **kwargs):
        """Runs a forward pass on a pointer network that generates
        permutation matrices in logistic space

        Arguments:

        inputs: TransformerInput
            a dataclass instance that contains queries, keys
            and values along with masks

        Returns:

        outputs: tf.Tensor
            a permutation matrix in logistic space that has the same shape
            as the transformer attention weights"""

        # apply the stick breaking operator
        return stick_breaking(tf.math.sigmoid(inputs[0]), inputs[1])

    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process

        Returns:

        config: dict
            a dictionary that contains all parameters to the
            layers base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict()

        base_config = super(StickBreaking, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))
