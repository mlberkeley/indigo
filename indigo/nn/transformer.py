from indigo.nn.wrappers.sequential import Sequential
from indigo.nn.layers.encoder_layer import EncoderLayer
from indigo.nn.layers.decoder_layer import DecoderLayer
from indigo.nn.features.discrete_feature import DiscreteFeature
from indigo.nn.features.continuous_feature import ContinuousFeature
from indigo.nn.features.region_feature import RegionFeature
from indigo.nn.variables.logits import Logits
from indigo.nn.variables.pointer_after_logits import PointerAfterLogits


class Transformer(Sequential):

    def __init__(self,
                 num_embeddings,
                 hidden_size,
                 heads,
                 num_layers,
                 queries_dropout=0.,
                 values_dropout=0.,
                 causal=True,
                 logits_per_slot=1,
                 first_layer='region',
                 final_layer='logits',
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
        values_dropout: float
            the ratio of units to drop during training to the
            number of units in each attention layer
        causal: bool
            specifies is the transformer should decoding using
            a causal mask to preserve the auto regressive property
        logits_per_slot: int
            specifies the number of logits per element the pointer
            network attends to; default is 2
        first_layer: class
            specifies the class to use for the first layer in the transformer
            defaults to WordFeature if not specified
        final_layer: class
            specifies the class to use for the final layer in the transformer
            defaults to Logits if not specified"""

        # TODO: the layers sequential does not technically yet
        #  support nested inputs but it should
        layers = []

        # the first layer in the transformer depends on the data modality
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
        layers.extend([EncoderLayer(
            hidden_size, hidden_size // 2, heads,
            queries_dropout=queries_dropout, values_dropout=values_dropout,
            causal=False, **kwargs) for _ in range(num_layers)])
        layers.extend([DecoderLayer(
            hidden_size, hidden_size // 2, heads,
            queries_dropout=queries_dropout, values_dropout=values_dropout,
            causal=causal, **kwargs) for _ in range(num_layers)])

        # the final layer in the transformer depends on the model purpose
        if final_layer == 'logits' or final_layer == 'indigo':
            layers.extend([Logits(num_embeddings, **kwargs)])
        if final_layer == 'indigo':
            layers.extend([PointerAfterLogits(
                hidden_size // 2, hidden_size, num_embeddings,
                causal=causal, logits_per_slot=logits_per_slot, **kwargs)])

        super(Transformer, self).__init__(layers)

        # these parameters need to be stored so that
        # tf.layers.model.save_model works
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.heads = heads
        self.num_layers = num_layers
        self.queries_dropout = queries_dropout
        self.values_dropout = values_dropout
        self.causal = causal
        self.logits_per_slot = logits_per_slot
        self.first_layer = first_layer
        self.final_layer = final_layer
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
                      values_dropout=self.values_dropout,
                      causal=self.causal,
                      logits_per_slot=self.logits_per_slot,
                      first_layer=self.first_layer,
                      final_layer=self.final_layer,
                      ** self.kwargs)

        base_config = super(Transformer, self).get_config()
        return dict(list(base_config.items()) +
                    list(config.items()))
