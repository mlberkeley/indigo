from indigo.nn.base.block import Block
import tensorflow as tf


class PointerLayer(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_size,
                 output_size):
        """Creates a ops layer for sampling permutation matrices
        rather than soft max attention masks

        Arguments:

        hidden_size: int
            the number of hidden units in the network blocks
            used by this layer
        output_size: int
            the number of output units used by the network blocks
            used by this layer"""
        super(PointerLayer, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.block = Block(hidden_size,
                           output_size * 2)

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

        # map into a common latent space
        q, k = tf.split(self.block(inputs.queries, **kwargs), 2, axis=2)
        scores = tf.matmul(q, k, transpose_b=True)

        # prevent the permutation matrix from assigning mass to
        # out of bounds elements
        mask = tf.logical_and(tf.expand_dims(inputs.queries_mask, 2),
                              tf.expand_dims(inputs.queries_mask, 1))
        return tf.where(
            mask, scores, tf.fill(tf.shape(scores), -999999.))

    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process

        Returns:

        config: dict
            a dictionary that contains all parameters to the
            keras base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict(hidden_size=self.hidden_size,
                      output_size=self.output_size)

        base_config = super(Block, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))
