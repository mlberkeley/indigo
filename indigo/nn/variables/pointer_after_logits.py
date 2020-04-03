from indigo.nn.base.block import Block
from indigo.nn.base.attention import causal_mask
from indigo.nn.variables.pointer import Pointer
import tensorflow as tf


class PointerAfterLogits(Pointer):

    def __init__(self,
                 hidden_size,
                 output_size,
                 logits_size,
                 causal=True,
                 logits_per_slot=1,
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
            network attends to; default is 1"""
        super(PointerAfterLogits, self).__init__(
            hidden_size,
            output_size,
            causal=causal,
            logits_per_slot=logits_per_slot,
            **kwargs)

        # the core processing variables
        self.logits_embedding = tf.keras.layers.Embedding(
            logits_size, output_size, **kwargs)

        # these parameters need to be stored so that
        # tf.layers.model.save_model works
        self.logits_size = logits_size

    def call(self, inputs, **kwargs):
        """Runs a forward pass on the logits of a transformer
        inputs is an instance of TransformerInput

        Arguments:

        inputs: TransformerInput
            a dataclass instance that contains queries, keys
            and values along with masks

        Returns:

        pointer_logits: tf.Tensor
            the logits of a pointer network used to select locations to
            insert words in a transformer"""

        # map the sequence into a latent space
        features = self.block(inputs.queries, **kwargs)
        q = features[..., :self.output_size]
        q = q + self.logits_embedding(inputs.ids)

        # reshape keys to have logits_per_slot more time steps
        shape = tf.multiply(tf.shape(q), [1, self.logits_per_slot, 1])
        k = tf.reshape(features[..., self.output_size:], shape)
        scores = tf.matmul(q, k, transpose_b=True)

        # prevent the permutation matrix from assigning mass to
        # out of bounds elements
        mask = tf.logical_and(tf.expand_dims(inputs.queries_mask, 2),
                              tf.expand_dims(inputs.queries_mask, 1))
        if self.causal:
            mask = tf.logical_and(
                mask, causal_mask(scores[:, :, ::self.logits_per_slot]))

        # filter by removing logits for elements that are invalid
        # mask must be repeated to correct the shape
        mask = tf.repeat(mask, self.logits_per_slot, axis=2)
        return tf.where(
            mask, scores, tf.fill(tf.shape(scores), -999999.))

    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process

        Returns:

        config: dict
            a dictionary that contains all parameters to the
            layers base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict(hidden_size=self.hidden_size,
                      output_size=self.output_size,
                      logits_size=self.logits_size,
                      causal=self.causal,
                      logits_per_slot=self.logits_per_slot,
                      **self.kwargs)

        base_config = super(PointerAfterLogits, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))
