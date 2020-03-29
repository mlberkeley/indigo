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
        when decoding using the transformer"""

    batch_size = tf.shape(model_features.queries)[0]
    closed_beams = tf.fill([batch_size], False)
    log_p = tf.zeros([batch_size])

    model_features = TransformerInput(
        queries=tf.fill([batch_size, 1], 2),
        values=model_features.values,
        queries_mask=tf.ones([batch_size, 1], tf.float32),
        values_mask=model_features.values_mask)

    for i in range(max_iterations):

        if tf.reduce_all(closed_beams):
            break

        logits = tf.math.log_softmax(model(model_features)[:, -1, :])
        values, ids = tf.math.top_k(logits, k=1)

        log_p = log_p + tf.where(
            closed_beams, tf.zeros_like(values), values)

        ids = tf.squeeze(ids, 1)
        ids = tf.where(closed_beams, tf.zeros_like(ids), ids)

        closed_beams = tf.logical_or(
            closed_beams, tf.equal(ids, 3))
        next_queries = tf.concat(
            [model_features.queries, ids[:, tf.newaxis]], 1)
        model_features = TransformerInput(
            queries=next_queries,
            values=model_features.values,
            queries_mask=tf.ones(tf.shape(next_queries), tf.float32),
            values_mask=model_features.values_mask)

    return model_features.queries, log_p
