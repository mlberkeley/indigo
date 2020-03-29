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
        tf.ones(shape=shape, dtype=tf.int32), axis=1)
    col_idx = tf.math.cumsum(
        tf.ones(shape=shape, dtype=tf.int32), axis=2)
    return tf.greater_equal(row_idx, col_idx)


class Attention(tf.keras.layers.Layer):

    def __init__(self,
                 heads,
                 queries_dropout=0.,
                 values_dropout=0.,
                 causal=True):
        """Creates the backbone for multi headed attention
        and supports dropout on the attention mask

        Arguments:

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
        super(Attention, self).__init__()

        self.queries_dropout = tf.keras.layers.Dropout(
            queries_dropout)
        self.keys_dropout = tf.keras.layers.SpatialDropout1D(
            values_dropout)
        self.values_dropout = tf.keras.layers.SpatialDropout1D(
            values_dropout)

        # these parameters need to be stored so that
        # tf.keras.model.save_model works
        self.heads = heads
        self.queries_dropout_rate = queries_dropout
        self.values_dropout_rate = values_dropout
        self.causal = causal

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

        # apply dropout to the queries tensor
        queries = self.queries_dropout(inputs.queries, **kwargs)
        queries = tf.concat(
            tf.split(queries, self.heads, axis=2), axis=0)

        # apply dropout to the keys tensor
        keys = self.keys_dropout(inputs.keys, **kwargs)
        keys = tf.concat(
            tf.split(keys, self.heads, axis=2), axis=0)

        # apply dropout to the values tensor
        values = self.values_dropout(inputs.values, **kwargs)
        values = tf.concat(
            tf.split(values, self.heads, axis=2), axis=0)

        # compute the multi head attention weights
        size = tf.math.sqrt(
            tf.cast(tf.shape(queries)[2], tf.float32))
        scores = tf.matmul(
            queries, keys, transpose_b=True) / size

        # apply a causal mask to the attention weights
        mask = tf.tile(inputs.values_mask, [self.heads, 1])
        mask = tf.expand_dims(mask, 1)
        if self.causal:
            mask = tf.logical_and(mask, causal_mask(scores))

        # apply a boolean mask to the keys and values
        scores = tf.math.softmax(
            scores - 999999. * tf.cast(mask, tf.float32))

        # apply the attention weights to compute an output sequence
        outputs = tf.matmul(scores, values)
        outputs = tf.concat(
            tf.split(outputs, self.heads, axis=0), axis=2)

        # mask the output sequence where appropriate
        return tf.where(
            tf.expand_dims(inputs.queries_mask, 2),
            outputs, tf.zeros_like(outputs))

    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process

        Returns:

        config: dict
            a dictionary that contains all parameters to the
            keras base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict(heads=self.heads,
                      queries_dropout=self.queries_dropout_rate,
                      values_dropout=self.values_dropout_rate,
                      causal=self.causal)

        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))
