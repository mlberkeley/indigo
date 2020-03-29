from indigo.input import TransformerInput
import tensorflow as tf


class ImageFeature(tf.keras.layers.Layer):

    def __init__(self,
                 num_embeddings,
                 hidden_size,
                 **kwargs):
        """Creates a Transformer embedding layer by applying a
        lookup operation to the queries

        Arguments:

        num_embeddings: int
            the number of elements in the vocabulary which
            input sequences contain elements of
        hidden_size: int
            the number of units in the hidden layers used
            in each multi head attention layer"""
        super(ImageFeature, self).__init__()

        # these parameters need to be stored so that
        # tf.keras.model.save_model works
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.kwargs = kwargs

        # the core processing layers
        self.embedding = tf.keras.layers.Embedding(
            num_embeddings, hidden_size, **kwargs)
        self.dense = tf.keras.layers.Dense(
            hidden_size, **kwargs)

    def call(self, inputs, **kwargs):
        """Runs a forward pass on the embeddings of a transformer
        inputs is an instance of TransformerInput

        Arguments:

        inputs: TransformerInput
            a dataclass instance that contains queries, keys
            and values along with masks

        Returns:

        outputs: TransformerInput
            the result of applying a multi head attention mechanism
            same shape as inputs"""

        return TransformerInput(
            queries=self.embedding(inputs.queries, **kwargs),
            values=self.dense(inputs.values, **kwargs),
            queries_mask=inputs.queries_mask,
            values_mask=inputs.values_mask)

    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process

        Returns:

        config: dict
            a dictionary that contains all parameters to the
            keras base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict(num_embeddings=self.num_embeddings,
                      hidden_size=self.hidden_size,
                      ** self.kwargs)

        base_config = super(ImageFeature, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))