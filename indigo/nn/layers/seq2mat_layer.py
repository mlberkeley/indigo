from indigo.nn.wrappers.layer import Layer
from indigo.nn.base.block import Block
from indigo.nn.base.sequence_to_mat import SequenceToMat
from indigo.nn.base.stick_breaking import StickBreaking
from indigo.nn.input import AttentionInput
import tensorflow as tf
import tensorflow_probability as tfp


class Seq2MatLayer(Layer):

    def __init__(self,
                 input_size,
                 hidden_size,
                 queries_dropout=0.,
                 keys_dropout=0.,
                 temperature=1.,
                 **kwargs):
        """Creates a Transformer permutation layer by applying a multi
        head sequence to matrix layer; and then applying sinkhorn
        normalization to the activations

        Arguments:

        hidden_size: int
            the number of units in the hidden variables used
            in each multi head attention layer
        queries_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        keys_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        temperature: float
            a positive number to divide the permutation logits by prior
            to applying sinkhorn normaliozation"""
        super(Seq2MatLayer, self).__init__()

        # the core attention and processing variables
        self.sequence_to_mat = SequenceToMat(
            queries_dropout=queries_dropout, keys_dropout=keys_dropout)
        self.block0 = Block(hidden_size, input_size * 2,
                            activation='linear', **kwargs)

        # these parameters need to be stored so that
        # tf.layers.model.save_model works
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.queries_dropout = queries_dropout
        self.keys_dropout = keys_dropout
        self.temperature = temperature
        self.kwargs = kwargs

    def call(self, inputs, **kwargs):
        """Runs a forward pass on a multi head attention layer
        inputs is an instance of TransformerInput

        Arguments:

        inputs: TransformerInput
            a dataclass instance that contains queries, keys
            and values along with masks

        Returns:

        permutation: TransformerInput
            the result of applying a sequence to matrix layer and
            sinkhorn normalization; a doubly stochastic matrix
            with shape [batch, seq_length, seq_length]"""

        # calculate the shape of the values tensor before performing attention
        # used when separating the heads from channels
        shape = tf.shape(inputs.queries)
        hidden_dim = self.input_size // 2

        # pass the input through a feed forward processing block and
        # separate heads from channels
        activations = self.block0(inputs.queries, **kwargs)
        activations = tf.transpose(tf.reshape(activations, [
            shape[0], shape[1], 2, hidden_dim * 2]), [0, 2, 1, 3])

        # convert the inputs into the standard data class format expected
        # by the attention class
        queries_mask = tf.expand_dims(inputs.queries_mask, 1)
        attention_input = AttentionInput(
            queries=activations[..., :hidden_dim],
            keys=activations[..., hidden_dim:],
            queries_mask=queries_mask,
            values_mask=queries_mask)

        # pass the input through an attention processing block and
        # take the sum over the parallel attention heads
        return self.sequence_to_mat(attention_input, **kwargs)[:, 0]

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
                      queries_dropout=self.queries_dropout,
                      keys_dropout=self.keys_dropout,
                      temperature=self.temperature,
                      ** self.kwargs)

        base_config = super(Seq2MatLayer, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))
