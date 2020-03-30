from indigo.input import TransformerInput
import tensorflow as tf


def greedy_search(model_features,
                  model,
                  max_iterations=100):
    """Perform a greedy search using a transformer model that
    accepts region features as input

    Arguments:

    model_features: TransformerInput
        a dataclass that contains input features that will be used
        when decoding using the transformer
    model: Transformer
        a keras model that accepts inputs in the form of the dataclass
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

    batch_size = tf.shape(model_features.queries)[0]
    closed_beams = tf.fill([batch_size], False)
    log_p = tf.zeros([batch_size])

    # create an empty partial sentence and corresponding mask
    model_features = TransformerInput(
        queries=tf.fill([batch_size, 1], 2),
        values=model_features.values,
        queries_mask=tf.fill([batch_size, 1], True),
        values_mask=model_features.values_mask)

    # loop for a maximum of max_iterations decoding steps
    for i in range(max_iterations):

        # exit if all beams have finished being decoded
        if tf.reduce_all(closed_beams):
            break

        # run the model for a single step
        logits = tf.math.log_softmax(model(model_features)[:, -1, :])
        values, ids = tf.math.top_k(logits, k=1)

        # calculate the log probability for every candidate sentence
        v = tf.squeeze(values, 1)
        log_p = log_p + tf.where(closed_beams, tf.zeros_like(v), v)

        # prepare the decoded word ids
        ids = tf.squeeze(ids, 1)
        ids = tf.where(closed_beams, tf.zeros_like(ids), ids)

        # record if any of the beams have finished decoding
        closed_beams = tf.logical_or(
            closed_beams, tf.equal(ids, 3))

        # concatenate the selected words with a partial sentence
        next_queries = tf.concat(
            [model_features.queries, ids[:, tf.newaxis]], 1)

        # format the inputs for the transformer in the next round
        model_features = TransformerInput(
            queries=next_queries,
            values=model_features.values,
            queries_mask=tf.fill(tf.shape(next_queries), True),
            values_mask=model_features.values_mask)

    return model_features.queries, log_p

