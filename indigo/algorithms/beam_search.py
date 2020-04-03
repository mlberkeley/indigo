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

    batch_size = tf.shape(inputs.values_mask)[0]
    closed_beams = tf.fill([batch_size * beam_size], False)

    # tile the batch size of the region feature by beam size
    # TODO: Brandon
    #  is there a way to do this recursively without knowing
    #  the structure beforehand
    inputs.values.features = tf.tile(
        inputs.values.features, [beam_size, 1, 1])
    inputs.values.boxes = tf.tile(
        inputs.values.boxes, [beam_size, 1, 1])
    inputs.values.detections = tf.tile(
        inputs.values.detections, [beam_size, 1])
    inputs.values_mask = tf.tile(
        inputs.values_mask, [beam_size, 1])

    # create an empty partial sentence and corresponding mask
    inputs.queries = tf.fill([batch_size * beam_size, 1], 2)
    inputs.queries_mask = tf.fill([batch_size * beam_size, 1], True)
    inputs.ids = tf.fill([batch_size * beam_size, 1], 2)
    inputs.positions = tf.fill([batch_size * beam_size, 1, 1], 0)
    inputs.log_probs = tf.zeros([batch_size * beam_size])

    # loop for a maximum of max_iterations decoding steps
    for i in range(max_iterations):

        # exit if all beams have finished decoding
        if tf.reduce_all(closed_beams):
            break

        # format the inputs for the transformer in the next round
        inputs, closed_beams = model.beam_search(inputs, closed_beams, beam_size)
        inputs.queries = inputs.ids
        inputs.queries_mask = tf.fill(tf.shape(inputs.ids), True)

    # decoding is finished so un flatten the beam dimension
    # returns a shape like [batch_size, beam_size, sequence_length]
    queries = tf.stack(tf.split(
        inputs.ids, beam_size, axis=0), axis=1)

    return queries, tf.reshape(
        inputs.log_probs, [batch_size, beam_size])
