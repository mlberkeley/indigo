import tensorflow as tf


def causal_mask(scores):
    """Creates a boolean mask that enforces the auto
    regressive property

    Arguments:

    scores: tf.Tensor
        a tensor representing a sequence with space
        [batch_size, seq_len_one, seq_len_two]

    Returns:

    mask: tf.Tensor
        a boolean tensor with the same shape as the input
        used to mask invalid attention weights"""

    # only the sequence axes need not be one
    shape = tf.shape(scores)
    shape = tf.concat([
        tf.ones_like(shape[:-2]), shape[-2:]], axis=0)

    # clever mask for creating a lower triangular matrix
    row_idx = tf.math.cumsum(
        tf.ones(shape=shape, dtype=tf.int32), axis=-2)
    col_idx = tf.math.cumsum(
        tf.ones(shape=shape, dtype=tf.int32), axis=-1)
    return tf.greater_equal(row_idx, col_idx)


class Attention(tf.keras.layers.Layer):

    def __init__(self,
                 queries_dropout=0.,
                 keys_dropout=0.,
                 values_dropout=0.,
                 causal=True):
        """Creates the backbone for multi headed attention
        and supports dropout on the attention mask

        Arguments:

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
        super(Attention, self).__init__()

        self.q_dropout = tf.keras.layers.Dropout(queries_dropout)
        self.k_dropout = tf.keras.layers.SpatialDropout2D(keys_dropout)
        self.v_dropout = tf.keras.layers.SpatialDropout2D(values_dropout)

        # these parameters need to be stored so that
        # tf.layers.model.save_model works
        self.queries_dropout_rate = queries_dropout
        self.keys_dropout_rate = keys_dropout
        self.values_dropout_rate = values_dropout
        self.causal = causal

    @tf.function(experimental_relax_shapes=True)
    def static_call(self, queries, keys, values,
                    queries_mask, values_mask,
                    bias, **kwargs):
        """Runs a forward pass on a multi head attention layer
        inputs is an instance of AttentionInput

        Arguments:

        inputs: AttentionInput
            a dataclass instance that contains queries, keys
            and values along with masks

        Returns:

        outputs: tf.Tensor
            the result of applying a multi head attention mechanism
            will be shaped [batch_dim, seq_dim, channels]"""

        # apply dropout to the queries keys and values tensor
        # requires all to be like [batch, heads, ]
        queries = self.q_dropout(queries, **kwargs)
        keys = self.k_dropout(keys, **kwargs)
        values = self.v_dropout(values, **kwargs)

        # compute the multi head soft attention weights using
        # scaled dot product attention
        size = tf.math.sqrt(
            tf.cast(tf.shape(queries)[-1], tf.float32))
        scores = tf.matmul(
            queries, keys, transpose_b=True) / size

        # if an attention bias is provided that add the attention bias
        # to the pre softmax scores matrix
        scores = scores + bias

        # apply a causal mask to the soft attention weights
        mask = tf.expand_dims(values_mask, -2)
        if self.causal:
            mask = tf.logical_and(mask, causal_mask(scores))

        # apply a boolean mask to the keys and values
        scores = tf.math.softmax(tf.where(
            mask, scores, tf.fill(tf.shape(scores), -999999.)))

        # mask the output sequence where appropriate
        outputs = tf.matmul(scores, values)
        return tf.where(tf.expand_dims(queries_mask, -1),
                        outputs, tf.zeros_like(outputs))

    def call(self, inputs, **kwargs):
        """Runs a forward pass on a multi head attention layer
        inputs is an instance of AttentionInput

        Arguments:

        inputs: AttentionInput
            a dataclass instance that contains queries, keys
            and values along with masks

        Returns:

        outputs: tf.Tensor
            the result of applying a multi head attention mechanism
            will be shaped [batch_dim, seq_dim, channels]"""

        bias = inputs.bias if hasattr(
            inputs, 'bias') and inputs.bias is not None else tf.zeros([])
        return self.static_call(inputs.queries, inputs.keys, inputs.values,
                                inputs.queries_mask, inputs.values_mask,
                                bias, **kwargs)

    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process

        Returns:

        config: dict
            a dictionary that contains all parameters to the
            layers base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict(queries_dropout=self.queries_dropout_rate,
                      keys_dropout=self.keys_dropout_rate,
                      values_dropout=self.values_dropout_rate,
                      causal=self.causal)

        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))
