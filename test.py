from indigo.nn.base.stick_breaking import stick_breaking, inv_stick_breaking
from indigo.nn.base.sequence_to_mat import SequenceToMat
from indigo.nn.input import AttentionInput
import tensorflow as tf


if __name__ == "__main__":

    I = tf.one_hot(tf.random.shuffle(tf.range(22)), 22)[tf.newaxis]
    y0 = tf.Variable(tf.random.normal([1, 1, 22, 64]) + 1.)
    y1 = tf.Variable(tf.random.normal([1, 1, 22, 64]) + 1.)
    optim = tf.keras.optimizers.Adam(learning_rate=0.1)
    


    layer = SequenceToMat()

    for i in range(1000):

        with tf.GradientTape()as tape:

            tape.watch(y0)
            tape.watch(y1)

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

        optim.apply_gradients(zip(tape.gradient(loss, [y0, y1]), [y0, y1]))

