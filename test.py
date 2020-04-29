from indigo.birkoff import birkhoff_von_neumann
import tensorflow as tf


if __name__ == "__main__":

    log_x = tf.random.normal([32, 21, 21])

    x = tf.exp(log_x)

    p, c = birkhoff_von_neumann(x, tf.constant(20))

    c = tf.math.softmax(tf.math.log(c))

    y = tf.reduce_sum(p * c[..., tf.newaxis, tf.newaxis], axis=1)

    print(x[0])
    print(y[0])

    print(tf.reduce_sum(y, axis=-1)[0])

    print(tf.reduce_sum(y, axis=-2)[0])
