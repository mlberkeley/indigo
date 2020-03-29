from indigo.input import TransformerInput
from indigo.input import RegionFeatureInput
import tensorflow as tf


def beam_search(model_features,
                model,
                beam_size=8,
                max_iterations=20):
    """Perform a beam search using a transformer model that accepts
    region features as input

    Arguments:

    model_features: TransformerInput
        a dataclass that contains input features that will be used
        when decoding using the transformer
    model: Transformer
        a keras model that accepts inputs in the form of the dataclass
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

    batch_size = tf.shape(model_features.queries)[0]
    closed_beams = tf.fill([batch_size * beam_size], False)
    log_p = tf.zeros([batch_size * beam_size])

    # tile the batch size of teh region feature by beam size
    region_features = RegionFeatureInput(
        features=tf.tile(
            model_features.values.features, [beam_size, 1, 1]),
        boxes=tf.tile(
            model_features.values.boxes, [beam_size, 1, 1]),
        detections=tf.tile(
            model_features.values.detections, [beam_size, 1]))

    # tile the batch size of the model inputs by the beam size
    model_features = TransformerInput(
        queries=tf.fill([batch_size * beam_size, 1], 2),
        values=region_features,
        queries_mask=tf.fill([batch_size * beam_size, 1], True),
        values_mask=tf.tile(
            model_features.values_mask, [beam_size, 1]))

    # loop for a maximum of max_iterations decoding steps
    for i in range(max_iterations):

        # exit if all beams have finished being decoded
        if tf.reduce_all(closed_beams):
            break

        # run the model for a single step
        logits = tf.math.log_softmax(model(model_features)[:, -1, :])
        values, ids = tf.math.top_k(logits, k=beam_size)

        # calculate the log probability for every candidate sentence
        log_p = log_p[:, tf.newaxis] + tf.where(
            closed_beams[:, tf.newaxis], tf.zeros_like(values), values)
        log_p = tf.reshape(log_p, [batch_size, beam_size * beam_size])
        log_p, beam_ids = tf.math.top_k(log_p, k=beam_size)
        log_p = tf.reshape(log_p, [batch_size * beam_size])

        # select the sentence with maximum log probability
        ids = tf.where(
            closed_beams[:, tf.newaxis], tf.zeros_like(ids), ids)
        ids = tf.reshape(ids, [batch_size, beam_size * beam_size])
        ids = tf.gather(ids, beam_ids, batch_dims=1)
        ids = tf.reshape(ids, [batch_size * beam_size])

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

    # decoding is finished so un flatten the beam dimension
    # returns a shape like [batch_size, beam_size, sequence_length]
    queries = tf.stack(tf.split(
        model_features.queries, beam_size, axis=0), axis=1)

    return queries, tf.reshape(
        log_p, [batch_size, beam_size])
