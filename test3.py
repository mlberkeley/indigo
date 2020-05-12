from indigo.nn.base.stick_breaking import stick_breaking, inv_stick_breaking
from indigo.nn.base.sequence_to_mat import SequenceToMat
from indigo.nn.input import AttentionInput
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
        act0 = tf.keras.layers.Activation(activation)
        fc0 = tf.keras.layers.Dense(hidden_size,
                                    activation=None,
                                    **kwargs)

        norm1 = tf.keras.layers.LayerNormalization(**kwargs)
        act1 = tf.keras.layers.Activation(activation)
        fc1 = tf.keras.layers.Dense(output_size,
                                    activation=None,
                                    **kwargs)

        # the sequential provides a common interface
        # for forward propagation
        super(Block, self).__init__([norm0,
                                     act0,
                                     fc0,
                                     norm1,
                                     act1,
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


if __name__ == "__main__":

    I = tf.one_hot(tf.random.shuffle(tf.range(22)), 22)[tf.newaxis]
    yp = tf.Variable(tf.random.normal([1, 1, 22, 64]))
    optim = tf.keras.optimizers.Adam(learning_rate=0.1)

    # this is the assumption violation
    d = Block(256, 128, activation='linear')

    layer = SequenceToMat()

    for i in range(1000):

        with tf.GradientTape()as tape:

            h = d(yp)
            y0 = h[..., :64]
            y1 = h[..., 64:]

            mask = tf.ones([1, 22, 22])
            x = mask / 22.

            z = inv_stick_breaking(x, mask)

            attention_input = AttentionInput(
                queries=y0,
                keys=y1,
                queries_mask=tf.equal(tf.ones_like(y0[..., 0]), 1),
                values_mask=tf.equal(tf.ones_like(y0[..., 0]), 1))

            y = layer(attention_input)[:, 0]

            p = stick_breaking(tf.math.sigmoid(
                y - tf.math.log(1. / z - 1.)), mask)

            loss = tf.reduce_mean((I - p) ** 2)
            print(loss.numpy())
            print(y[0, 0])
            print(p[0, 0])
            print(I[0, 0])

        var_list = d.trainable_variables
        optim.apply_gradients(
            zip(tape.gradient(loss, var_list), var_list))

