from indigo.nn.engine.layer import Layer
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

        logits = tf.math.log_softmax(self.call(inputs, **kwargs)[:, -1])
        log_probs, samples = tf.math.top_k(logits, k=1)

        log_probs = tf.where(closed[:, tf.newaxis],
                             tf.zeros_like(log_probs), log_probs)
        samples = tf.where(closed[:, tf.newaxis],
                           tf.zeros_like(samples), samples)

        inputs.ids = tf.concat([inputs.ids, samples], 1)
        inputs.log_probs = inputs.log_probs + log_probs[..., 0]
        return inputs, tf.logical_or(closed, tf.equal(samples[:, 0], 3))

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

        # TODO: Brandon
        #  something is not quite right with this method

        logits = tf.math.log_softmax(self.call(inputs, **kwargs)[:, -1])
        log_probs, samples = tf.math.top_k(logits, k=beam_size)
        batch_size = int(tf.shape(logits)[0].numpy() // last_beam_size)

        log_probs = tf.where(closed[:, tf.newaxis], tf.zeros_like(log_probs), log_probs)
        samples = tf.where(closed[:, tf.newaxis], tf.zeros_like(samples), samples)

        log_probs = tf.reshape(log_probs, [-1, last_beam_size, beam_size])
        log_probs = tf.reshape(inputs.log_probs, [-1, last_beam_size, 1]) + log_probs
        log_probs = tf.reshape(log_probs, [-1, last_beam_size * beam_size])
        log_probs, beam_ids = tf.math.top_k(log_probs, k=beam_size)

        new_beam_ids = tf.math.floormod(beam_ids, beam_size)
        samples = tf.reshape(samples, [-1, last_beam_size * beam_size])
        samples = tf.gather(samples, new_beam_ids, batch_dims=1)
        samples = tf.reshape(samples, [-1, 1])

        old_beam_ids = tf.math.floordiv(beam_ids, beam_size)

        def select_beams(tensor):
            tensor = tf.stack(tf.split(tensor, batch_size, axis=0), axis=0)
            tensor = tf.gather(tensor, old_beam_ids, batch_dims=1)
            return tf.concat(tf.unstack(tensor, axis=0), axis=0)

        data_dict = {x: y for x, y in inputs.__dict__.items() if tf.is_tensor(y)}
        data_dict = tree.map_structure(select_beams, data_dict)
        for key, value in data_dict.items():
            inputs.__dict__[key] = value

        data_dict = {x: y for x, y in inputs.region.__dict__.items() if tf.is_tensor(y)}
        data_dict = tree.map_structure(select_beams, data_dict)
        for key, value in data_dict.items():
            inputs.region.__dict__[key] = value

        inputs.ids = tf.concat([inputs.ids, samples], 1)
        inputs.log_probs = tf.reshape(log_probs, [batch_size * beam_size])
        return inputs, tf.logical_or(
            select_beams(closed), tf.equal(samples[:, 0], 3)), beam_size

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
