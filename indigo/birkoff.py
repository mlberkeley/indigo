from networkx.algorithms.bipartite.matching import maximum_matching
from networkx import from_numpy_matrix
import tensorflow as tf
import numpy as np

TOLERANCE = np.finfo(np.float).eps * 10.


def get_permutation_np(bipartite_matrix):
    """Calculates the maximum cardinality perfect matching using networkx
    that corresponds to a permutation matrix

    Arguments:

    bipartite_matrix: tf.Tensor
        a binary matrix that corresponds to a bipartite graph; used for
        constructing a bipartite graph in networkx

    Returns:

    permutation: tf.Tensor
        a tensor containing the Berkhoff-Von-Neumann perumtation matrices
        found using the Berkhoff-Von-Neumann decomposition"""

    n = bipartite_matrix.shape[1] // 2
    matches = maximum_matching(from_numpy_matrix(bipartite_matrix), range(n))
    matches = np.array([[u, v % n] for u, v in matches.items() if u < n])

    permutation = np.zeros((n, n), dtype=np.float32)
    permutation[(matches[:, 0], matches[:, 1])] = 1
    return permutation


def get_permutation(bipartite_matrix):
    """Calculates the maximum cardinality perfect matching using networkx
    that corresponds to a permutation matrix

    Arguments:

    bipartite_matrix: tf.Tensor
        a binary matrix that corresponds to a bipartite graph; used for
        constructing a bipartite graph in networkx

    Returns:

    permutation: tf.Tensor
        a tensor containing the Berkhoff-Von-Neumann perumtation matrices
        found using the Berkhoff-Von-Neumann decomposition"""

    return tf.numpy_function(
        get_permutation_np, [bipartite_matrix], tf.float32)


def birkhoff_von_neumann_step(matrix):
    """Returns the Berkhoff-Von-Neumann decomposition of a permutation matrix
    using the greedy birkhoff heuristic

    Arguments:

    matrix: tf.Tensor
        a soft permutation matrix in the Birkhoff-Polytope whose shape is
        like [batch_dim, sequence_len, sequence_len]

    Returns:

    permutation: tf.Tensor
        a tensor containing the Berkhoff-Von-Neumann perumtation matrix
        found for the remaining values in matrix
    coefficient: tf.Tensor
        a tensor containing the Berkhoff-Von-Neumann coefficient
        found for the remaining values in matrix"""

    b, m, n = tf.shape(matrix)[0], tf.shape(matrix)[1], tf.shape(matrix)[2]
    pattern_matrix = tf.cast(tf.greater(matrix, 0), tf.float32)

    top = tf.concat([tf.zeros([b, m, m]), pattern_matrix], axis=2)
    bipartite_matrix = tf.concat([top, tf.concat([tf.transpose(
        pattern_matrix, [0, 2, 1]), tf.zeros([b, n, n])], axis=2)], axis=1)

    permutation = tf.map_fn(get_permutation, bipartite_matrix)
    permutation.set_shape(matrix.get_shape())
    upper_bound = tf.fill(tf.shape(matrix), 999999.)

    return permutation, tf.reduce_min(tf.where(
        tf.equal(permutation, 0), upper_bound, matrix), axis=[1, 2])


def birkhoff_von_neumann(x):
    """Returns the Berkhoff-Von-Neumann decomposition of a permutation matrix
    using the greedy birkhoff heuristic

    Arguments:

    x: tf.Tensor
        a soft permutation matrix in the Birkhoff-Polytope whose shape is
        like [batch_dim, sequence_len, sequence_len]

    Returns:

    permutations: tf.Tensor
        a tensor containing the Berkhoff-Von-Neumann perumtation matrices
        found using the Berkhoff-Von-Neumann decomposition
    coefficients: tf.Tensor
        a tensor containing the Berkhoff-Von-Neumann coefficients
        found using the Berkhoff-Von-Neumann decomposition"""

    b, n = x.get_shape()[0], tf.cast(tf.shape(x)[2], tf.float32)
    x = x * n

    coefficients = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    permutations = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    j = tf.constant(-1)
    d = tf.reduce_all(tf.equal(x, 0), axis=[1, 2])

    while tf.logical_not(tf.reduce_all(d)):
        j = j + 1

        p, c = birkhoff_von_neumann_step(x)
        d = tf.logical_or(d, tf.less(tf.reduce_sum(p, axis=[1, 2]), n))
        p = tf.where(d[:, tf.newaxis, tf.newaxis], tf.zeros_like(p), p)
        c = tf.where(d, tf.zeros_like(c), c)
        x = x - c * p
        x = tf.where(tf.less(tf.abs(x), TOLERANCE), tf.zeros_like(x), x)
        d = tf.logical_or(d, tf.reduce_all(tf.equal(x, 0), axis=[1, 2]))

        permutations = permutations.write(j, p)
        coefficients = coefficients.write(j, c)

    return (tf.transpose(permutations.stack(), [1, 0, 2, 3]),
            tf.transpose(coefficients.stack(), [1, 0]) / n)
