import tensorflow as tf


def greedy_search(inputs,
                  model,
                  max_iterations=100):
    """Perform a greedy search using a transformer model that
    accepts region features as input

    Arguments:

    inputs: TransformerInput
        a dataclass that contains input features that will be used
        when decoding using the transformer
    model: Transformer
        a layers model that accepts inputs in the form of the dataclass
        TransformerInput and returns logits
    max_iterations: int
        the maximum number of decoding steps to use when performing
        greedy search; the maximum sequence length

    Returns:

    sequence: tf.Tensor
        a tensor that contains output word ids that were taken
        when decoding using the transformer
    log_p: tf.Tensor
        the log probability of predicted sentences under the
        current transformer model"""

    batch_size = tf.shape(inputs.values_mask)[0]
    closed_beams = tf.fill([batch_size], False)

    # create an empty partial sentence and corresponding mask
    inputs.queries = tf.fill([batch_size, 1], 2)
    inputs.queries_mask = tf.fill([batch_size, 1], True)
    inputs.ids = tf.fill([batch_size, 1], 2)
    inputs.positions = tf.fill([batch_size, 1, 1], 0)
    inputs.log_probs = tf.zeros([batch_size])

    # loop for a maximum of max_iterations decoding steps
    for i in range(max_iterations):

        # exit if all beams have finished decoding
        if tf.reduce_all(closed_beams):
            break

        # format the inputs for the transformer in the next round
        inputs, closed_beams = model.greedy_search(inputs, closed_beams)
        inputs.queries = inputs.ids
        inputs.queries_mask = tf.fill(tf.shape(inputs.ids), True)

    return inputs.ids, inputs.log_probs
