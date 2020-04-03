import tensorflow as tf


class Logits(tf.keras.layers.Layer):

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

        # the core processing layers
        self.dense = tf.keras.layers.Dense(
            output_size, **kwargs)

        # these parameters need to be stored so that
        # tf.keras.model.save_model works
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

    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process

        Returns:

        config: dict
            a dictionary that contains all parameters to the
            keras base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict(output_size=self.output_size,
                      ** self.kwargs)

        base_config = super(Logits, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))
