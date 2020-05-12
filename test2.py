from indigo.nn.layers.permutation_layer import PermutationLayer
from indigo.nn.layers.decoder_layer import DecoderLayer
from indigo.nn.input import TransformerInput
import tensorflow as tf


if __name__ == "__main__":

    I = tf.one_hot(tf.random.shuffle(tf.range(22)), 22)[tf.newaxis]
    yp = tf.Variable(tf.random.normal([1, 22, 64]))
    xp = tf.Variable(tf.random.normal([1, 13, 64]))
    optim = tf.keras.optimizers.Adam(learning_rate=0.001)

    d0 = DecoderLayer(64, 256, 4, causal=False)
    d1 = DecoderLayer(64, 256, 4, causal=False)
    d2 = DecoderLayer(64, 256, 4, causal=False)
    layer = PermutationLayer(64, 256)

    for i in range(1000):

        with tf.GradientTape()as tape:

            attention_input = TransformerInput(
                queries=yp,
                values=xp,
                queries_mask=tf.equal(tf.ones_like(yp[..., 0]), 1),
                values_mask=tf.equal(tf.ones_like(xp[..., 0]), 1))

            p = layer(d2(d1(d0(attention_input))))

            loss = tf.reduce_mean((I - p) ** 2)
            print(loss.numpy())
            print(p[0, 0])
            print(I[0, 0])

        var_list = layer.trainable_weights + \
                   d0.trainable_weights + \
                   d1.trainable_weights + \
                   d2.trainable_weights
        optim.apply_gradients(
            zip(tape.gradient(loss, var_list), var_list))

