from indigo.nn.base.block import Block
from indigo.nn.ops.attention import causal_mask
from indigo.nn.base.logits import Logits
import tensorflow as tf


class PointerAndLogits(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_size,
                 output_size,
                 logits_size,
                 causal=True,
                 logits_per_slot=2,
                 **kwargs):
        """Creates a pointer network using the first operation
        in the self attention mechanism

        Arguments:

        hidden_size: int
            the number of hidden units in the network blocks
            used by this layer
        output_size: int
            the number of output units used by the network blocks
            used by this layer
        logits_size: int
            the number of units in the vector space of the logits
            of a transformer model
        causal: bool
            specifies is the transformer should decoding using
            a causal mask to preserve the auto regressive property
        logits_per_slot: int
            specifies the number of logits per element the pointer
            network attends to; default is 2"""
        super(PointerAndLogits, self).__init__()

        # the core processing layers
        self.logits = Logits(logits_size, **kwargs)
        self.block = Block(hidden_size,
                           output_size * (1 + logits_per_slot),
                           **kwargs)
        self.logits_embedding = tf.keras.layers.Embedding(
            logits_size, output_size, **kwargs)

        # these parameters need to be stored so that
        # tf.keras.model.save_model works
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.logits_size = logits_size
        self.causal = causal
        self.logits_per_slot = logits_per_slot
        self.kwargs = kwargs

    def call(self, inputs, **kwargs):
        """Runs a forward pass on the logits of a transformer
        inputs is an instance of TransformerInput

        Arguments:

        inputs: TransformerInput
            a dataclass instance that contains queries, keys
            and values along with masks

        Returns:

        logits: tf.Tensor
            the logits of a transformer model used for word prediction
            or during a search algorithm
        pointer_logits: tf.Tensor
            the logits of a pointer network used to select locations to
            insert words in a transformer"""

        # map the sequence into a latent space
        features = self.block(inputs.queries, **kwargs)
        q = features[..., :self.output_size]
        q = q + self.logits_embedding(inputs.targets)

        # reshape keys to have logits_per_slot more time steps
        shape = tf.multiply(tf.shape(q), [1, self.logits_per_slot, 1])
        k = tf.reshape(features[..., self.output_size:], shape)
        scores = tf.matmul(q, k, transpose_b=True)

        # prevent the permutation matrix from assigning mass to
        # out of bounds elements
        mask = tf.logical_and(tf.expand_dims(inputs.queries_mask, 2),
                              tf.expand_dims(inputs.queries_mask, 1))
        if self.causal:
            mask = tf.logical_and(mask, causal_mask(scores))

        # filter by removing logits for elements that are invalid
        # mask must be repeated to correct the shape
        mask = tf.repeat(mask, [1, 1, self.logits_per_slot])
        return self.logits(inputs, **kwargs), tf.where(
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
                      output_size=self.output_size,
                      logits_size=self.logits_size,
                      causal=self.causal,
                      logits_per_slot=self.logits_per_slot,
                      **self.kwargs)

        base_config = super(PointerAndLogits, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))
