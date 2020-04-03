import tensorflow as tf


class Layer(tf.keras.layers.Layer):

    def loss(self, inputs, **kwargs):
        """A function that implements a forward pass and calculates the
        loss function for this layers layer

        Arguments:

        batch: Dataclass
            a dataclass that stores ground truth information used for
            calculating the loss function

        Returns:

        loss: tf.Tensor
            a tensor representing the contribution this layer makes to the
            total model loss function
        outputs: Dataclass
            a dataclass that manages inputs and outputs for layers variables
            is mutable and will be mutated by this layer"""

        return 0.0, self.call(inputs, **kwargs)

    def greedy_update(self, inputs, **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using greedy search

        Arguments:

        batch: Dataclass
            a dataclass that stores partial decoding information that will
            be mutated by this layer during decoding

        Returns:

        decoding: Dataclass
            a dataclass that stores partial decoding information that will
            be mutated by this layer during decoding"""

        return self.call(inputs, **kwargs)

    def beam_update(self, inputs, **kwargs):
        """A function that implements a forward pass and updates the decoding
        partial sequence using a beam search

        Arguments:

        batch: Dataclass
            a dataclass that stores partial decoding information that will
            be mutated by this layer during decoding

        Returns:

        decoding: Dataclass
            a dataclass that stores partial decoding information that will
            be mutated by this layer during decoding"""

        return self.call(inputs, **kwargs)
