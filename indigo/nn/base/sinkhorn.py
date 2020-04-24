import tensorflow as tf

def matching(matrix_batch):
    """Solves a matching problem for a batch of matrices.
    Modified from 
    https://github.com/google/gumbel_sinkhorn/blob/master/sinkhorn_ops.py
    
    This is a wrapper for the scipy.optimize.linear_sum_assignment function. It
    solves the optimization problem max_P sum_i,j M_i,j P_i,j with P a
    permutation matrix. Notice the negative sign; the reason, the original
    function solves a minimization problem
    Args:
    matrix_batch: A 3D tensor (a batch of matrices) with
      shape = [batch_size, N, N]. If 2D, the input is reshaped to 3D with
      batch_size = 1.
    Returns:
    listperms, a 3D integer tensor of permutations with shape [batch_size, N, N]
      so that listperms[n, :, :] is the permutation matrix P of size N*N that solves the
      problem  max_P sum_i,j M_i,j P_i,j with M = matrix_batch[n, :, :].
    """

    def hungarian(x):
        if x.ndim == 2:
            x = np.reshape(x, [1, x.shape[0], x.shape[1]])
        sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
        for i in range(x.shape[0]):
            sol[i, :] = linear_sum_assignment(-x[i, :])[1].astype(np.int32)
        return sol

    listperms = tf.py_func(hungarian, [matrix_batch], tf.int32) # 2D
    listperms = tf.one_hot(listperms, tf.shape(listperms)[1], dtype=tf.int32) # 3D
    return listperms

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


def sinkhorn(x,
             iterations=20):
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
        return tf.exp(sinkhorn(inputs, self.iterations))

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
