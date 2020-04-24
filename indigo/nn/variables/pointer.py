from indigo.nn.wrappers.layer import Layer
from indigo.nn.base.block import Block
from indigo.nn.base.attention import causal_mask
import tensorflow as tf
import tree


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
        size = tf.math.sqrt(tf.cast(tf.shape(q)[2], tf.float32))
        scores = tf.matmul(q, k, transpose_b=True) / size

        # prevent the permutation matrix from assigning mass to
        # out of bounds elements
        mask = tf.logical_and(tf.expand_dims(inputs.queries_mask, 2),
                              tf.expand_dims(inputs.queries_mask, 1))
        if self.causal:
            mask = tf.logical_and(
                mask, causal_mask(scores[:, :, ::self.logits_per_slot]))

        # filter by removing logits for elements that are invalid
        # mask must be repeated to correct the shape
        mask = tf.repeat(mask, self.logits_per_slot, axis=2)
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
        return tf.keras.losses.categorical_crossentropy(
            inputs.pointer_labels, pointer, from_logits=True), inputs

    def greedy_search(self,
                      inputs,
                      closed,
                      **kwargs):
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

        # compute a distribution over tokens; note that only one token
        # is sampled yet top_k is a convenient function
        logits = tf.math.log_softmax(self.call(inputs, **kwargs)[:, -1])
        log_probs, ids = tf.math.top_k(logits, k=1)

        # mask the log probabilities and tokens of already completed
        # beams so that they are unchanged when decoding
        mask = closed[:, tf.newaxis]
        log_probs = tf.where(mask, tf.zeros_like(log_probs), log_probs)
        ids = tf.where(mask, tf.fill(tf.shape(ids), tf.shape(logits)[1]), ids)

        # compute the relative position update vector using the samples ids
        # this equals -1 if ids are to the left and +1 if to the right
        R = inputs.relative_positions
        R = tf.argmax(R, axis=-1, output_type=tf.int32) - 1
        r = tf.gather(R, ids, batch_dims=1, axis=2)
        r = tf.squeeze(tf.where(tf.equal(r, 0), tf.ones_like(r), r), axis=2)

        # concatenate the relative position vector to the left and to the
        # bottom of the relative position matrix; see the paper
        # https://arxiv.org/pdf/1902.01370.pdf
        inputs.relative_positions = tf.one_hot(tf.concat([
            tf.concat([R, r[:, :, tf.newaxis]], axis=2),
            tf.pad(-r, [[0, 0], [0, 1]])[:, tf.newaxis, :]], axis=1) + 1, 3)

        # compute the update log probability and note that the pointer network
        # does not specify a termination condition by itself
        inputs.log_probs = inputs.log_probs + log_probs[..., 0]
        return inputs, closed

    def beam_search(self,
                    inputs,
                    closed,
                    last_beam_size,
                    beam_size,
                    **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using a beam search

        Arguments:

        batch: Dataclass
            a dataclass that stores partial decoding information that will
            be mutated by this layer during decoding
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified
        last_beam_size: int
            the number of beams that were expanded by the last layer in an
            autoregressive model
        beam_size: int
            the number of beams to be expanded by this layer in an
            autoregressive model

        Returns:

        decoding: Dataclass
            a dataclass that stores partial decoding information that will
            be mutated by this layer during decoding
        closed: tf.Tensor
            a boolean tensor where true values indicate that a beam has
            finished decoding and should not be modified
        beam_size: int
            the number of beams to be expanded by this layer in an
            autoregressive model"""

        # compute a distribution over pointer locations
        logits = tf.math.log_softmax(self.call(inputs, **kwargs)[:, -1])
        batch_size = tf.shape(logits)[0] // last_beam_size

        # note that when the sequence length is small the number of locations
        # that are visible to the pointer network may be too small; the
        # effective beam size is reduced in these cases
        sample_size = tf.minimum(tf.shape(logits)[1], beam_size)

        # sample the top beam_size candidates
        log_probs, ids = tf.math.top_k(logits, k=sample_size)

        # when a beam is closed all candidates are the same
        # this prevents the same candidates from being sampled twice
        first = tf.one_hot(tf.fill(tf.shape(log_probs)[:1], 0), sample_size)
        closed_log_probs = tf.where(tf.equal(first, 0), tf.fill(
            tf.shape(first), -999999.), tf.fill(tf.shape(first), 0.))

        # when a beam is closed special behavior is required
        # do not change the log probability and append only pad tokens
        mask = closed[:, tf.newaxis]
        log_probs = tf.where(mask, closed_log_probs, log_probs)
        ids = tf.where(mask, tf.fill(tf.shape(ids), tf.shape(logits)[1]), ids)

        # manipulate the log probabilities to extract all possible
        # next beam candidates and their probability
        log_probs = tf.reshape(log_probs, [
            batch_size, last_beam_size, sample_size])
        log_probs = tf.reshape(inputs.log_probs, [
            batch_size, last_beam_size, 1]) + log_probs
        log_probs = tf.reshape(log_probs, [
            batch_size, last_beam_size * sample_size])

        # note that when the sequence length is small the number of locations
        # that are visible to the pointer network may be too small; the
        # effective beam size is reduced in these cases
        cand_size = tf.minimum(tf.shape(log_probs)[1], beam_size)

        # select the top beam_size candidates
        log_probs, beam_ids = tf.math.top_k(log_probs, k=cand_size)

        # these indices may be a bit subtle; they work as follows
        # the last dim has last_beam_size * beam_size elements
        # the first beam_size elements represent candidate proposals
        # from a single original beam
        old_beam_ids = tf.math.floordiv(beam_ids, sample_size)

        # select the ids based on their beams that are from the beams with
        # highest log probability
        ids = tf.reshape(ids, [batch_size, last_beam_size * sample_size])
        ids = tf.gather(ids, beam_ids, batch_dims=1)
        ids = tf.reshape(ids, [batch_size * cand_size, 1])

        # this function helps select the hidden activations from
        # inputs that correspond to old selected beams
        # this is necessary because future layers may depend on activations
        # that are a function of which beam was selected
        def select(x):
            shape = tf.shape(x)[1:]
            s0 = tf.concat([[batch_size, last_beam_size], shape], axis=0)
            s1 = tf.concat([[batch_size * cand_size], shape], axis=0)
            return tf.reshape(tf.gather(
                tf.reshape(x, s0), old_beam_ids, batch_dims=1), s1)

        # this function helps perform the previously described selection
        # operation over the contents of a python 3.7 dataclass
        # dataclasses are used as input containers for our models and
        # help deal with multiple-input layers and losses
        def map_dataclass(func, obj):
            data_dict = tree.map_structure(func, {
                x: y for x, y in obj.__dict__.items() if tf.is_tensor(y)})
            for key, value in data_dict.items():
                obj.__dict__[key] = value

        # select the image features and the transformer hidden activations
        # that correspond to the selected beams after performing
        # a step of beam search
        map_dataclass(select, inputs)
        map_dataclass(select, inputs.region)

        # compute the relative position update vector using the samples ids
        # this equals -1 if ids are to the left and +1 if to the right
        R = select(inputs.relative_positions)
        R = tf.argmax(R, axis=-1, output_type=tf.int32) - 1
        r = tf.gather(R, ids, batch_dims=1, axis=2)
        r = tf.squeeze(tf.where(tf.equal(r, 0), tf.ones_like(r), r), axis=2)

        # concatenate the relative position vector to the left and to the
        # bottom of the relative position matrix; see the paper
        # https://arxiv.org/pdf/1902.01370.pdf
        inputs.relative_positions = tf.one_hot(tf.concat([
            tf.concat([R, r[:, :, tf.newaxis]], axis=2),
            tf.pad(-r, [[0, 0], [0, 1]])[:, tf.newaxis, :]], axis=1) + 1, 3)

        # update log probability and note that the pointer network
        # does not specify a termination condition by itself
        inputs.log_probs = tf.reshape(log_probs, [batch_size * cand_size])
        return inputs, closed, cand_size

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
