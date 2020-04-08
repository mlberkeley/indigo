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
    inputs.positions = tf.fill([batch_size, 1, 1], 0)
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
        start = tf.fill([batch_size * last_beam_size, 1], 2)
        inputs.queries = tf.concat([start, inputs.ids], axis=1)
        inputs.queries_mask = tf.fill(tf.shape(inputs.queries), True)
        inputs.values = inputs.region

    # decoding is finished so un flatten the beam dimension
    # returns a shape like [batch_size, beam_size, sequence_length]
    queries = tf.stack(tf.split(inputs.ids, beam_size, axis=0), axis=1)
    return queries, tf.reshape(
        inputs.log_probs, [batch_size, beam_size])
