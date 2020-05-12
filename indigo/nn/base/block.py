import tensorflow as tf


class Block(tf.keras.Sequential):

    def __init__(self,
                 hidden_size,
                 output_size,
                 activation='relu',
                 **kwargs):
        """Creates a 'network-in-network' style block that is used
        in self attention variables

        Arguments:

        hidden_size: int
            the number of units in the network hidden layer
            processed using a convolution
        output_size: int
            the number of units in the network output layer
            processed using a convolution
        activation: str
            an input to tf.layers.variables.Activation for building
            an activation function"""

        # order of variables is the same as a typical 'resnet'
        norm0 = tf.keras.layers.LayerNormalization(**kwargs)
        relu0 = tf.keras.layers.Activation(activation)
        fc0 = tf.keras.layers.Dense(hidden_size,
                                    activation=None,
                                    **kwargs)

        norm1 = tf.keras.layers.LayerNormalization(**kwargs)
        relu1 = tf.keras.layers.Activation(activation)
        fc1 = tf.keras.layers.Dense(output_size,
                                    activation=None,
                                    **kwargs)

        # the sequential provides a common interface
        # for forward propagation
        super(Block, self).__init__([norm0,
                                     relu0,
                                     fc0,
                                     norm1,
                                     relu1,
                                     fc1])

        # these parameters need to be stored so that
        # tf.layers.model.save_model works
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.kwargs = kwargs

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
                      activation=self.activation,
                      ** self.kwargs)

        base_config = super(Block, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))
