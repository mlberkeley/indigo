from indigo.nn.base.block import Block
from indigo.nn.ops.attention import Attention
from indigo.input import AttentionInput
from indigo.input import TransformerInput
import tensorflow as tf


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self,
                 input_size,
                 hidden_size,
                 heads,
                 queries_dropout=0.,
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
            the number of units in the hidden layers used
            in each multi head attention layer
        heads: int
            the number of heads in each multi head attention layer
            a good default is 4 or 8
        queries_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        values_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        causal: bool
            specifies is the transformer should decoding using
            a causal mask to preserve the auto regressive property"""
        super(DecoderLayer, self).__init__()

        # the core attention and processing layers
        self.attention0 = Attention(
            heads,
            queries_dropout=queries_dropout,
            values_dropout=values_dropout,
            causal=causal)
        self.block0 = Block(hidden_size * heads // 2,
                            hidden_size * heads * 3,
                            **kwargs)
        self.block1 = Block(input_size // 2,
                            input_size,
                            **kwargs)

        # the core attention and processing layers
        self.attention1 = Attention(
            heads,
            queries_dropout=queries_dropout,
            values_dropout=values_dropout,
            causal=False)
        self.block2 = Block(hidden_size * heads // 2,
                            hidden_size * heads,
                            **kwargs)
        self.block3 = Block(hidden_size * heads // 2,
                            hidden_size * heads * 2,
                            **kwargs)
        self.block4 = Block(input_size // 2,
                            input_size,
                            **kwargs)

        # these parameters need to be stored so that
        # tf.keras.model.save_model works
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.heads = heads
        self.queries_dropout = queries_dropout
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

        activations = self.block0(inputs.queries, **kwargs)
        queries, keys, values = tf.split(
            activations, 3, axis=2)

        attention_input = AttentionInput(
            queries=queries,
            keys=keys,
            values=values,
            queries_mask=inputs.queries_mask,
            values_mask=inputs.queries_mask)

        y = self.attention0(attention_input, **kwargs)
        y = self.block1(y, **kwargs)

        inputs = TransformerInput(
            queries=inputs.queries + y,
            values=inputs.values,
            queries_mask=inputs.queries_mask,
            values_mask=inputs.values_mask)

        queries = self.block2(inputs.queries, **kwargs)
        keys, values = tf.split(
            self.block3(inputs.values, **kwargs), 2, axis=2)

        attention_input = AttentionInput(
            queries=queries,
            keys=keys,
            values=values,
            queries_mask=inputs.queries_mask,
            values_mask=inputs.values_mask)

        y = self.attention1(attention_input, **kwargs)
        y = self.block4(y, **kwargs)

        return TransformerInput(
            queries=inputs.queries + y,
            values=inputs.values,
            queries_mask=inputs.queries_mask,
            values_mask=inputs.values_mask)

    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process

        Returns:

        config: dict
            a dictionary that contains all parameters to the
            keras base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict(input_size=self.input_size,
                      hidden_size=self.hidden_size,
                      heads=self.heads,
                      queries_dropout=self.queries_dropout,
                      values_dropout=self.values_dropout,
                      causal=self.causal,
                      ** self.kwargs)

        base_config = super(DecoderLayer, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))