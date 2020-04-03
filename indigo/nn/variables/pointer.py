from indigo.nn.engine.layer import Layer
from indigo.nn.base.block import Block
from indigo.nn.base.attention import causal_mask
import tensorflow as tf


class Pointer(Layer):

    def __init__(self,
                 hidden_size,
                 output_size,
                 causal=True,
                 logits_per_slot=1,
                 **kwargs):
        """Creates a pointer network using the first operation
        in the self attention mechanism

        Arguments:

        hidden_size: int
            the number of hidden units in the network blocks
            used by this layer
        output_size: int
            the number of output units used by the network blocks
            used by this layer
        causal: bool
            specifies is the transformer should decoding using
            a causal mask to preserve the auto regressive property
        logits_per_slot: int
            specifies the number of logits per element the pointer
            network attends to; default is 1"""
        super(Pointer, self).__init__()

        # the core processing variables
        self.block = Block(
            hidden_size, output_size * (1 + logits_per_slot), **kwargs)

        # these parameters need to be stored so that
        # tf.layers.model.save_model works
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.causal = causal
        self.logits_per_slot = logits_per_slot
        self.kwargs = kwargs

    def call(self, inputs, **kwargs):
        """Runs a forward pass on a pointer network that generates
        permutation matrices in log space

        Arguments:

        inputs: TransformerInput
            a dataclass instance that contains queries, keys
            and values along with masks

        Returns:

        outputs: tf.Tensor
            a permutation matrix in log space that has the same shape
            as the transformer attention weights"""

        # map the sequence into a latent space
        features = self.block(inputs.queries, **kwargs)
        q = features[..., :self.output_size]

        # reshape keys to have logits_per_slot more time steps
        shape = tf.multiply(tf.shape(q), [1, self.logits_per_slot, 1])
        k = tf.reshape(features[..., self.output_size:], shape)
        scores = tf.matmul(q, k, transpose_b=True)

        # prevent the permutation matrix from assigning mass to
        # out of bounds elements
        mask = tf.logical_and(tf.expand_dims(inputs.queries_mask, 2),
                              tf.expand_dims(inputs.queries_mask, 1))
        if self.causal:
            mask = tf.logical_and(mask, causal_mask(scores))

        # filter by removing logits for elements that are invalid
        # mask must be repeated to correct the shape
        mask = tf.repeat(mask, [1, 1, self.logits_per_slot])
        return tf.where(
            mask, scores, tf.fill(tf.shape(scores), -999999.))

    def loss(self, inputs, **kwargs):
        """Runs a forward pass on the logits of a transformer
        inputs is an instance of TransformerInput

        Arguments:

        inputs: TransformerInput
            a dataclass instance that contains queries, keys
            and values along with masks

        Returns:

        loss: tf.Tensor
            a loss function that computes the contribution this layer
            makes to the total model loss
        outputs: tf.Tensor
            the logits of a transformer model used for word prediction
            or a pointer network"""

        pointer = self.call(inputs, **kwargs)
        absolute_pos = tf.reduce_sum(tf.nn.relu(inputs.positions), axis=2)
        return tf.keras.losses.sparse_categorical_crossentropy(
            absolute_pos, pointer, from_logits=True), inputs

    def greedy_search(self, inputs, closed, **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using greedy search

        Arguments:

        batch: Dataclass
            a dataclass that stores partial decoding information that will
            be mutated by this layer during decoding
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified

        Returns:

        decoding: Dataclass
            a dataclass that stores partial decoding information that will
            be mutated by this layer during decoding
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified"""

        pointer = tf.math.log_softmax(self.call(inputs.queries, **kwargs)[:, -1])
        log_probs, samples = tf.math.top_k(pointer, k=1)

        log_probs = tf.where(
            closed[:, tf.newaxis], tf.zeros_like(log_probs), log_probs)
        samples = tf.where(
            closed[:, tf.newaxis], tf.fill(
                tf.shape(samples), tf.shape(pointer)[1]), samples)

        r = tf.gather(inputs.positions, samples, batch_dims=1)
        r = tf.squeeze(tf.where(tf.equal(r, 0), tf.ones_like(r), r), axis=1)

        inputs.positions = tf.concat([
            tf.concat([inputs.positions, -r[:, :, tf.newaxis]], axis=2),
            tf.pad(r, [[0, 0], [0, 1]])[:, tf.newaxis, :]], axis=1)

        inputs.log_probs = inputs.log_probs + log_probs[..., 0]
        return inputs, closed

    def beam_search(self, inputs, closed, beam_size, **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using a beam search

        Arguments:

        batch: Dataclass
            a dataclass that stores partial decoding information that will
            be mutated by this layer during decoding
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified

        Returns:

        decoding: Dataclass
            a dataclass that stores partial decoding information that will
            be mutated by this layer during decoding
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified"""

        pointer = tf.math.log_softmax(self.call(inputs.queries, **kwargs)[:, -1])
        log_probs, samples = tf.math.top_k(pointer, k=beam_size)

        log_probs = tf.where(
            closed[:, tf.newaxis], tf.zeros_like(log_probs), log_probs)
        samples = tf.where(
            closed[:, tf.newaxis], tf.fill(
                tf.shape(samples), tf.shape(pointer)[1]), samples)

        log_probs = tf.reshape(log_probs, [-1, beam_size, beam_size])
        log_probs = tf.reshape(inputs.log_probs, [-1, beam_size, 1]) + log_probs
        log_probs = tf.reshape(log_probs, [-1, beam_size * beam_size])
        inputs.log_probs, beam_ids = tf.math.top_k(log_probs, k=beam_size)

        new_beam_ids = tf.math.floormod(beam_ids, beam_size)
        samples = tf.gather(samples, new_beam_ids, batch_dims=1)
        samples = tf.reshape(samples, [-1, 1])

        old_beam_ids = tf.math.floordiv(beam_ids, beam_size)
        pos = tf.reshape(inputs.positions, [-1,
                                            beam_size,
                                            tf.shape(inputs.positions)[1],
                                            tf.shape(inputs.positions)[2]])
        pos = tf.gather(pos, old_beam_ids, batch_dims=1)
        pos = tf.reshape(pos, [-1,
                               tf.shape(inputs.positions)[1],
                               tf.shape(inputs.positions)[2]])

        r = tf.gather(pos, samples, batch_dims=1)
        r = tf.squeeze(tf.where(tf.equal(r, 0), tf.ones_like(r), r), axis=1)

        inputs.positions = tf.concat([
            tf.concat([pos, -r[:, :, tf.newaxis]], axis=2),
            tf.pad(r, [[0, 0], [0, 1]])[:, tf.newaxis, :]], axis=1)

        return inputs, closed

    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process

        Returns:

        config: dict
            a dictionary that contains all parameters to the
            layers base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict(hidden_size=self.hidden_size,
                      output_size=self.output_size,
                      causal=self.causal,
                      logits_per_slot=self.logits_per_slot,
                      **self.kwargs)

        base_config = super(Pointer, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))
