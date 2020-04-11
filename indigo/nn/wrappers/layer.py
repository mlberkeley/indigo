import tensorflow as tf


class Layer(tf.keras.layers.Layer):

    def loss(self,
             inputs,
             **kwargs):
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

        return tf.zeros([]), self.call(inputs, **kwargs)

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

        return self.call(inputs, **kwargs), closed

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

        return self.call(inputs, **kwargs), closed, last_beam_size
