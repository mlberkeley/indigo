from indigo.nn.wrappers.sequential import Sequential
from indigo.nn.layers.encoder_layer import EncoderLayer
from indigo.nn.layers.decoder_layer import DecoderLayer
from indigo.nn.layers.permutation_layer import PermutationLayer
from indigo.nn.features.discrete_feature import DiscreteFeature
from indigo.nn.features.continuous_feature import ContinuousFeature
from indigo.nn.features.region_feature import RegionFeature


class PermutationTransformer(Sequential):

    def __init__(self,
                 num_embeddings,
                 hidden_size,
                 heads,
                 num_layers,
                 queries_dropout=0.,
                 keys_dropout=0.,
                 values_dropout=0.,
                 first_layer='region',
                 temperature=1.,
                 **kwargs):
        """Creates a Transformer Keras model for processing sequences
        and uses the tf.layers.Sequential as backend

        Arguments:

        num_embeddings: int
            the number of elements in the vocabulary which
            input sequences contain elements of
        hidden_size: int
            the number of units in the hidden variables used
            in each multi head attention layer
        heads: int
            the number of heads in each multi head attention layer
            a good default is 4 or 8
        num_layers: int
            the number of variables in the encoder and the decoder modules
            each layer consists of attention residual connections
        queries_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        keys_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        values_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        first_layer: class
            specifies the class to use for the first layer in the transformer
            defaults to WordFeature if not specified
        temperature: float
            a positive number to divide the permutation logits by prior
            to applying sinkhorn normalization"""

        # TODO: Sequential does not technically support nested inputs
        layers = []

        # the first layer in the transformer depends on the data modality
        # for image captioning using RCNN features select 'region'
        if first_layer == 'discrete':
            layers.extend([DiscreteFeature(
                num_embeddings, hidden_size, **kwargs)])
        if first_layer == 'continuous':
            layers.extend([ContinuousFeature(
                num_embeddings, hidden_size, **kwargs)])
        if first_layer == 'region':
            layers.extend([RegionFeature(
                num_embeddings, hidden_size, **kwargs)])

        # the encoder processes values and the decoder processes queries
        # build the encoder first in the stack
        layers.extend([EncoderLayer(
            hidden_size, hidden_size * 4, heads,
            queries_dropout=queries_dropout,
            keys_dropout=keys_dropout,
            values_dropout=values_dropout,
            causal=False, **kwargs) for _ in range(num_layers)])

        # depending on the type of network possibly condition on position
        # build the decoder second in the stack
        layers.extend([DecoderLayer(
            hidden_size, hidden_size * 4, heads,
            queries_dropout=queries_dropout,
            keys_dropout=keys_dropout,
            values_dropout=values_dropout,
            causal=False, **kwargs) for _ in range(num_layers)])

        # the final layer in the transformer depends on the model purpose
        # to run Transformer-InDIGO select 'indigo'
        layers.extend([PermutationLayer(
            hidden_size, hidden_size * 4,
            queries_dropout=queries_dropout,
            keys_dropout=keys_dropout,
            temperature=temperature, **kwargs)])

        super(PermutationTransformer, self).__init__(layers)

        # these parameters need to be stored so that
        # tf.layers.model.save_model works
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.heads = heads
        self.num_layers = num_layers
        self.queries_dropout = queries_dropout
        self.keys_dropout = keys_dropout
        self.values_dropout = values_dropout
        self.first_layer = first_layer
        self.temperature = temperature
        self.kwargs = kwargs

    def get_config(self):
        """Creates a state dictionary that can be used to rebuild
        the layer in another python process

        Returns:

        config: dict
            a dictionary that contains all parameters to the
            layers base class and all class parameters"""

        # these are all that is needed to rebuild this class
        config = dict(num_embeddings=self.num_embeddings,
                      hidden_size=self.hidden_size,
                      heads=self.heads,
                      num_layers=self.num_layers,
                      queries_dropout=self.queries_dropout,
                      keys_dropout=self.keys_dropout,
                      values_dropout=self.values_dropout,
                      first_layer=self.first_layer,
                      temperature=self.temperature,
                      ** self.kwargs)

        base_config = super(PermutationTransformer, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))
