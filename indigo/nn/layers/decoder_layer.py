from indigo.nn.wrappers.layer import Layer
from indigo.nn.base.block import Block
from indigo.nn.base.attention import Attention
from indigo.nn.input import AttentionInput
import tensorflow as tf


class DecoderLayer(Layer):

    def __init__(self,
                 input_size,
                 hidden_size,
                 heads,
                 queries_dropout=0.,
                 keys_dropout=0.,
                 values_dropout=0.,
                 causal=True,
                 **kwargs):
        """Creates a Transformer decoder layer by applying a
        multi head attention layer

        Arguments:

        input_size: int
            the number of units in the input tensor to this layer
            also the output size of the model
        hidden_size: int
            the number of units in the hidden variables used
            in each multi head attention layer
        heads: int
            the number of heads in each multi head attention layer
            a good default is 4 or 8
        queries_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        keys_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        values_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        causal: bool
            specifies is the transformer should decoding using
            a causal mask to preserve the auto regressive property"""
        super(DecoderLayer, self).__init__()

        # the core attention and processing variables
        self.attention0 = Attention(
            queries_dropout=queries_dropout,
            keys_dropout=keys_dropout,
            values_dropout=values_dropout,
            causal=causal)
        self.block0 = Block(hidden_size // 2,
                            hidden_size * 3,
                            **kwargs)
        self.block1 = Block(input_size // 2,
                            input_size,
                            **kwargs)

        # the core attention and processing variables
        self.attention1 = Attention(
            queries_dropout=queries_dropout,
            keys_dropout=keys_dropout,
            values_dropout=values_dropout,
            causal=False)
        self.block2 = Block(hidden_size // 2,
                            hidden_size,
                            **kwargs)
        self.block3 = Block(hidden_size // 2,
                            hidden_size * 2,
                            **kwargs)
        self.block4 = Block(input_size // 2,
                            input_size,
                            **kwargs)

        # these parameters need to be stored so that
        # tf.layers.model.save_model works
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.heads = heads
        self.queries_dropout = queries_dropout
        self.keys_dropout = keys_dropout
        self.values_dropout = values_dropout
        self.causal = causal
        self.kwargs = kwargs

    def call(self, inputs, **kwargs):
        """Runs a forward pass on a multi head attention layer
        inputs is an instance of TransformerInput

        Arguments:

        inputs: TransformerInput
            a dataclass instance that contains queries, keys
            and values along with masks

        Returns:

        outputs: TransformerInput
            the result of applying a multi head attention mechanism
            same shape as inputs"""

        # calculate the shape of the values tensor before performing attention
        # used when separating the heads from channels
        s0 = tf.shape(inputs.queries)
        s1 = tf.shape(inputs.values)
        hidden_dim = self.hidden_size // self.heads

        # pass the input through a feed forward processing block and
        # separate heads from channels
        activations = self.block0(inputs.queries, **kwargs)
        activations = tf.transpose(tf.reshape(activations, [
            s0[0], s0[1], self.heads, hidden_dim * 3]), [0, 2, 1, 3])

        # convert the inputs into the standard data class format expected
        # by the attention class
        queries_mask = tf.expand_dims(inputs.queries_mask, 1)
        att_input = AttentionInput(
            queries=activations[..., :hidden_dim],
            keys=activations[..., hidden_dim:2 * hidden_dim],
            values=activations[..., 2 * hidden_dim:],
            queries_mask=queries_mask,
            values_mask=queries_mask)

        # pass the input through an attention processing block and
        # flatten the heads and channels
        activations = self.attention0(att_input, **kwargs)
        activations = tf.reshape(tf.transpose(activations, [
            0, 2, 1, 3]), [s0[0], s0[1], self.heads * hidden_dim])

        # pass the outputs of the attention through another feed forward
        # processing block a residual connection
        activations = self.block1(activations, **kwargs)
        inputs.queries = inputs.queries + activations

        # pass the input through a feed forward processing block and
        # separate heads from channels
        activations = self.block2(inputs.queries, **kwargs)
        queries = tf.transpose(tf.reshape(activations, [
            s0[0], s0[1], self.heads, hidden_dim]), [0, 2, 1, 3])

        # pass the input through a feed forward processing block and
        # separate heads from channels
        activations = self.block3(inputs.values, **kwargs)
        activations = tf.transpose(tf.reshape(activations, [
            s1[0], s1[1], self.heads, hidden_dim * 2]), [0, 2, 1, 3])

        # convert the inputs into the standard data class format expected
        # by the attention class
        values_mask = tf.expand_dims(inputs.values_mask, 1)
        att_input = AttentionInput(
            queries=queries,
            keys=activations[..., :hidden_dim],
            values=activations[..., hidden_dim:],
            queries_mask=queries_mask,
            values_mask=values_mask)

        # pass the input through an attention processing block and
        # flatten the heads and channels
        activations = self.attention1(att_input, **kwargs)
        activations = tf.reshape(tf.transpose(activations, [
            0, 2, 1, 3]), [s0[0], s0[1], self.heads * hidden_dim])

        # pass the outputs of the attention through another feed forward
        # processing block a residual connection
        activations = self.block4(activations, **kwargs)
        inputs.queries = inputs.queries + activations
        return inputs

    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process

        Returns:

        config: dict
            a dictionary that contains all parameters to the
            layers base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict(input_size=self.input_size,
                      hidden_size=self.hidden_size,
                      heads=self.heads,
                      queries_dropout=self.queries_dropout,
                      keys_dropout=self.keys_dropout,
                      values_dropout=self.values_dropout,
                      causal=self.causal,
                      ** self.kwargs)

        base_config = super(DecoderLayer, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))
