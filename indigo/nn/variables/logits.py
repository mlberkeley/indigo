from indigo.nn.wrappers.layer import Layer
import tensorflow as tf
import tree


class Logits(Layer):

    def __init__(self,
                 output_size,
                 **kwargs):
        """Creates a Transformer embedding layer by applying a
        lookup operation to the queries

        Arguments:

        output_size: int
            the number of units in the vector space of the logits
            of a transformer model"""
        super(Logits, self).__init__()

        # the core processing variables
        self.dense = tf.keras.layers.Dense(output_size, **kwargs)

        # these parameters need to be stored so that
        # tf.layers.model.save_model works
        self.output_size = output_size
        self.kwargs = kwargs

    def call(self, inputs, **kwargs):
        """Runs a forward pass on the logits of a transformer
        inputs is an instance of TransformerInput

        Arguments:

        inputs: TransformerInput
            a dataclass instance that contains queries, keys
            and values along with masks

        Returns:

        outputs: tf.Tensor
            the logits of a transformer model used for word prediction
            or a pointer network"""

        return self.dense(inputs.queries, **kwargs)

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

        logits = self.call(inputs, **kwargs)
        return tf.keras.losses.sparse_categorical_crossentropy(
            inputs.ids, logits, from_logits=True), inputs

    def greedy_search(self,
                      inputs,
                      closed,
                      **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using greedy search

        Arguments:

        inputs: Dataclass
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
        ids = tf.where(mask, tf.zeros_like(ids), ids)

        # concatenate the sampled tokens to the beam and prepare the
        # model outputs for the next layer; also compute if we
        # has finished decoding by predicting the end token
        inputs.ids = tf.concat([inputs.ids, ids], 1)
        inputs.log_probs = inputs.log_probs + log_probs[..., 0]
        return inputs, tf.logical_or(
            closed, tf.equal(ids[:, 0], 3))

    def beam_search(self,
                    inputs,
                    closed,
                    last_beam_size,
                    beam_size,
                    **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using a beam search

        Arguments:

        inputs: Dataclass
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

        # compute a distribution over tokens
        logits = tf.math.log_softmax(self.call(inputs, **kwargs)[:, -1])
        batch_size = tf.shape(logits)[0] // last_beam_size

        # sample the top beam_size candidates
        log_probs, ids = tf.math.top_k(logits, k=beam_size)

        # when a beam is closed all candidates are the same
        # this prevents the same candidates from being sampled twice
        first = tf.one_hot(tf.fill(tf.shape(log_probs)[:1], 0), beam_size)
        closed_log_probs = tf.where(tf.equal(first, 0), tf.fill(
            tf.shape(first), -999999.), tf.fill(tf.shape(first), 0.))

        # when a beam is closed special behavior is required
        # do not change the log probability and append only pad tokens
        mask = closed[:, tf.newaxis]
        log_probs = tf.where(mask, closed_log_probs, log_probs)
        ids = tf.where(mask, tf.zeros_like(ids), ids)

        # manipulate the log probabilities to extract all possible
        # next beam candidates and their probability
        log_probs = tf.reshape(log_probs, [
            batch_size, last_beam_size, beam_size])
        log_probs = tf.reshape(inputs.log_probs, [
            batch_size, last_beam_size, 1]) + log_probs
        log_probs = tf.reshape(log_probs, [
            batch_size, last_beam_size * beam_size])

        # select the top beam_size candidates
        log_probs, beam_ids = tf.math.top_k(log_probs, k=beam_size)

        # these indices may be a bit subtle; they work as follows
        # the last dim has last_beam_size * beam_size elements
        # the first beam_size elements represent candidate proposals
        # from a single original beam
        new_beam_ids = tf.math.floormod(beam_ids, beam_size)
        old_beam_ids = tf.math.floordiv(beam_ids, beam_size)

        # select the ids based on their beams that are from the beams with
        # highest log probability
        ids = tf.reshape(ids, [batch_size, last_beam_size * beam_size])
        ids = tf.gather(ids, new_beam_ids, batch_dims=1)
        ids = tf.reshape(ids, [batch_size * beam_size, 1])

        # this function helps select the hidden activations from
        # inputs that correspond to old selected beams
        # this is necessary because future layers may depend on activations
        # that are a function of which beam was selected
        def select(x):
            shape = tf.shape(x)[1:]
            s0 = tf.concat([[batch_size, last_beam_size], shape], axis=0)
            s1 = tf.concat([[batch_size * beam_size], shape], axis=0)
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

        # concatenate the sampled tokens to the beam and prepare the
        # model outputs for the next layer; also compute if we
        # has finished decoding by predicting the end token
        inputs.ids = tf.concat([inputs.ids, ids], 1)
        inputs.log_probs = tf.reshape(log_probs, [batch_size * beam_size])
        return inputs, tf.logical_or(
            select(closed), tf.equal(ids[:, 0], 3)), beam_size

    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process

        Returns:

        config: dict
            a dictionary that contains all parameters to the
            layers base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict(output_size=self.output_size,
                      ** self.kwargs)

        base_config = super(Logits, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))
