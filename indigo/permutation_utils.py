import tensorflow as tf


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)])
def permutation_to_pointer(permutation):
    """Converts a permutation matrix to the label distribution of
    a pointer network for training a language model

    Arguments:

    permutation: tf.Tensor
        a permutation matrix that defines the order in which
        words are inserted by the language model

    Returns:

    pointer: tf.Tensor
        a ternary matrix that contains relative positions of words
        inserted by a language model non-sequentially"""

    # make sure the permutation is an int or the below computation
    # does not make sense
    permutation = tf.cast(permutation, tf.int32)
    n = tf.shape(permutation)[-1]

    # this first section will convert the one-hot style indexing to
    # a ternary indexing where -1 means insert to the right of
    # and 1 means insert to the left of word x
    unsorted_relative = -tf.math.cumsum(
        permutation, axis=-1, exclusive=True) + tf.math.cumsum(
            permutation, axis=-1, exclusive=True, reverse=True)

    # sort the relative positions into the decoding order induced
    # by the permutation
    sorted_relative = tf.matmul(
        permutation, unsorted_relative, transpose_b=True)

    # get the one hot distribution of pointer labels; should contain
    # a sparse lower triangular matrix
    sorted_ptr = tf.cast(tf.reduce_sum(tf.maximum(
        0, tf.linalg.band_part(sorted_relative, 0, -1)), axis=-2), tf.int32)

    # the variable sorted_ptr is in sorted partial positions but the pointer
    # network reuses state and does not sort as decoding progresses
    # so we need to convert into unsorted ptr positions
    partial_pos = tf.repeat(
        sorted_relative[..., tf.newaxis, :, :], n, axis=-3)
    partial_pos = tf.linalg.band_part(
        tf.transpose(partial_pos, [0, 1, 4, 3, 2]), 0, -1)
    partial_pos = tf.linalg.band_part(
        tf.transpose(partial_pos, [0, 1, 3, 2, 4]), 0, -1)
    partial_pos = tf.cast(tf.reduce_sum(tf.maximum(
        0, tf.transpose(partial_pos, [0, 1, 4, 2, 3])), axis=-2), tf.int32)

    # lookup the sorted positions in a table of unsorted positions
    unsorted_ptr = tf.argmin(tf.abs(sorted_ptr[
        ..., tf.newaxis] - 1 - partial_pos), axis=-1, output_type=tf.int32)

    # the start token is never inserted so we can slice out the first channel
    # in addition there are only n - 1 valid insertion locations
    return tf.one_hot(unsorted_ptr[..., 1:], n - 1)


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)])
def permutation_to_relative(permutation):
    """Converts a permutation matrix to a relative position
    matrix for training a language model

    Arguments:

    permutation: tf.Tensor
        a permutation matrix that defines the order in which
        words are inserted by the language model

    Returns:

    relative: tf.Tensor
        a ternary matrix that contains relative positions of words
        inserted by a language model non-sequentially"""

    # make sure the permutation is an int or the below computation
    # does not make sense
    permutation = tf.cast(permutation, tf.int32)

    # this first section will convert the one-hot style indexing to
    # a ternary indexing where -1 means insert to the right of
    # and 1 means insert to the left of word x
    unsorted_relative = -tf.math.cumsum(
        permutation, axis=-1, exclusive=True) + tf.math.cumsum(
            permutation, axis=-1, exclusive=True, reverse=True)

    # sort the relative positions into the decoding order induced
    # by the permutation
    sorted_relative = tf.matmul(
        permutation, unsorted_relative, transpose_b=True)

    # get the one hot distribution of relative positions; contains
    # a one at location i when [left, center, right]_i
    return tf.one_hot(sorted_relative[..., :-1, :-1] + 1, 3)


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    tf.TensorSpec(shape=[None, None], dtype=tf.int32),
    tf.TensorSpec(shape=None, dtype=tf.string)])
def get_permutation(mask, words, order):
    """Construct a discrete permutation matrix for training a non sequential
    autoregressive model using gradient descent

    Arguments:

    mask: tf.Tensor
        a tensor containing zeros and ones which indicate which elements
        of words are out of bounds
    words: tf.Tensor
        the batch of word ids that will be used to determine the
        permutation when using rare or common
    order: tf.Tensor
        the autoregressive ordering to train Transformer-InDIGO using;
        l2r, r2l, rare, or common

    Returns:

    permutation: tf.Tensor
        a permutation matrix for training a non sequential autoregressive
        model using gradient descent"""

    # the dataset is not compiled with an ordering so one must
    # be generated on the fly during training; only applies
    # when using a pointer layer; note that the end token
    # must always be last and start token must always  be first
    b, n = tf.shape(words)[0], tf.shape(words)[1]

    if tf.equal(order, 'r2l'):  # corresponds to right-to-left
        length = tf.cast(tf.reduce_sum(mask, axis=1), tf.int32)
        ind = tf.tile(tf.range(n - 1)[tf.newaxis], [b, 1])
        ind = tf.reverse_sequence(ind, length - 2, seq_axis=1, batch_axis=0)
        ind = tf.concat([tf.fill([b, 1], 0), 1 + ind], axis=1)

    elif tf.equal(order, 'rare'):  # corresponds to rare-first
        upper_bound = tf.reduce_max(words, axis=1, keepdims=True) + 1
        scores = tf.where(tf.equal(words, 0), -tf.ones_like(words), words)
        scores = tf.where(tf.equal(words, 1), upper_bound, scores)
        scores = tf.where(tf.equal(words, 2), upper_bound + 1, scores)
        scores = tf.where(tf.equal(words, 3), tf.zeros_like(words), scores)
        ind = tf.argsort(scores, direction='DESCENDING')

    elif tf.equal(order, 'common'):  # corresponds to common-first
        upper_bound = tf.reduce_max(words, axis=1, keepdims=True) + 1
        scores = tf.where(tf.equal(words, 0), upper_bound + 2, words)
        scores = tf.where(tf.equal(words, 1), upper_bound, scores)
        scores = tf.where(tf.equal(words, 2), tf.zeros_like(words), scores)
        scores = tf.where(tf.equal(words, 3), upper_bound + 1, scores)
        ind = tf.argsort(scores, direction='ASCENDING')

    else:  # corresponds to left-to-right
        ind = tf.tile(tf.range(n)[tf.newaxis], [b, 1])

    return tf.one_hot(ind, n)
