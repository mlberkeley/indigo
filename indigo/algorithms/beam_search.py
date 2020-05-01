import tensorflow as tf


def beam_search(inputs,
                model,
                beam_size=8,
                max_iterations=20):
    """Perform a beam search using a transformer model that accepts
    region features as input

    Arguments:

    inputs: TransformerInput
        a dataclass that contains input features that will be used
        when decoding using the transformer
    model: Transformer
        a layers model that accepts inputs in the form of the dataclass
        TransformerInput and returns logits
    beam_size: int
        the number of beams to use when calculating a beam search
        a beam size of zero is a greedy search
    max_iterations: int
        the maximum number of decoding steps to use when performing
        beam search; the maximum sequence length

    Returns:

    sequence: tf.Tensor
        a tensor that contains output word ids that were taken
        when decoding using the transformer
    log_p: tf.Tensor
        the log probability of predicted sentences under the
        current transformer model"""

    # meta data to keep track of which beams have completed
    # during the decoding step
    batch_size = tf.shape(inputs.values_mask)[0]
    closed = tf.fill([batch_size], False)
    last_beam_size = 1

    # replace the model inputs with an empty sentence that will be
    # appended to during the decoding step
    inputs.queries = tf.fill([batch_size, 1], 2)
    inputs.queries_mask = tf.fill([batch_size, 1], True)
    inputs.ids = tf.fill([batch_size, 0], 2)
    inputs.relative_positions = tf.one_hot(tf.fill([batch_size, 1, 1], 1), 3)
    inputs.absolute_positions = None
    inputs.log_probs = tf.zeros([batch_size])
    inputs.region = inputs.values

    # loop for a maximum of max_iterations decoding steps
    for i in range(max_iterations):

        # exit if all beams have finished decoding
        if tf.reduce_all(closed):
            break

        # format the inputs for the transformer in the next round
        inputs, closed, last_beam_size = model.beam_search(
            inputs, closed, last_beam_size, beam_size)

        # the transformer modifies in place the input data class so
        # we need to replace the transformer inputs at every
        # iteration of decoding
        inputs.values = inputs.region
        start = tf.fill([batch_size * last_beam_size, 1], 2)
        inputs.queries = tf.concat([start, inputs.ids], axis=1)
        inputs.queries_mask = tf.concat([
            inputs.queries_mask,
            tf.logical_not(closed)[:, tf.newaxis]], axis=1)

    # helper function for un flattening the beam size from the batch axis
    def expand(x):
        return tf.reshape(x, tf.concat([[
            batch_size, last_beam_size], tf.shape(x)[1:]], axis=0))

    # decoding is finished so un flatten the beam dimension
    # returns a shape like [batch_size, beam_size, sequence_length]
    ids = expand(inputs.ids)

    # when the model decodes permutation matrices in additions to ids;
    # then sort ids according to the decoded permutation
    if model.final_layer == 'indigo':
        pos = inputs.relative_positions
        pos = tf.argmax(pos, axis=-1, output_type=tf.int32) - 1
        pos = tf.reduce_sum(tf.nn.relu(expand(pos[:, 1:, 1:])), axis=2)
        pos = tf.one_hot(pos, tf.shape(pos)[2], dtype=tf.int32)
        ids = tf.squeeze(tf.matmul(tf.expand_dims(ids, 2), pos), 2)

    return ids, tf.reshape(
        inputs.log_probs, [batch_size, last_beam_size])
