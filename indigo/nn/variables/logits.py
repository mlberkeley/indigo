from indigo.nn.engine.layer import Layer
import tensorflow as tf


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

    def greedy_update(self, inputs, **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using greedy search

        Arguments:

        batch: Dataclass
            a dataclass that stores partial decoding information that will
            be mutated by this layer during decoding
        inputs: Dataclass
            a dataclass that manages inputs and outputs for layers variables
            is mutable and will be mutated by this layer

        Returns:

        decoding: Dataclass
            a dataclass that stores partial decoding information that will
            be mutated by this layer during decoding
        outputs: Dataclass
            a dataclass that manages inputs and outputs for layers variables
            is mutable and will be mutated by this layer"""

        logits = self.call(inputs.queries, **kwargs)[:, -1]
        log_probs, samples = tf.math.top_k(logits, k=1)
        inputs.ids = tf.concat([inputs.ids, samples], 1)
        inputs.log_probs = inputs.log_probs + log_probs[..., 0]
        return inputs

    def beam_update(self, inputs, beam_size=8, **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using a beam search

        Arguments:

        batch: Dataclass
            a dataclass that stores partial decoding information that will
            be mutated by this layer during decoding
        inputs: Dataclass
            a dataclass that manages inputs and outputs for layers variables
            is mutable and will be mutated by this layer

        Returns:

        decoding: Dataclass
            a dataclass that stores partial decoding information that will
            be mutated by this layer during decoding
        outputs: Dataclass
            a dataclass that manages inputs and outputs for layers variables
            is mutable and will be mutated by this layer"""

        logits = self.call(inputs.queries, **kwargs)[:, -1]
        log_probs, samples = tf.math.top_k(logits, k=beam_size)
        log_probs = tf.reshape(log_probs, [-1, beam_size, beam_size])

        log_probs = tf.reshape(inputs.log_probs, [-1, beam_size, 1]) + log_probs
        log_probs = tf.reshape(log_probs, [-1, beam_size * beam_size])
        inputs.log_probs, beam_ids = tf.math.top_k(log_probs, k=beam_size)

        new_beam_ids = tf.math.floormod(beam_ids, beam_size)
        samples = tf.gather(samples, new_beam_ids, batch_dims=1)
        samples = tf.reshape(samples, [-1, 1])

        old_beam_ids = tf.math.floordiv(beam_ids, beam_size)
        ids = tf.reshape(inputs.ids, [-1, beam_size, tf.shape(inputs.ids)[1]])
        ids = tf.gather(ids, old_beam_ids, batch_dims=1)
        ids = tf.reshape(ids, [-1,  tf.shape(inputs.ids)[1]])

        inputs.ids = tf.concat([ids, samples], 1)
        return inputs

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
