from indigo.data.load import faster_rcnn_dataset
from indigo.nn.input import TransformerInput
from indigo.nn.input import RegionFeatureInput
from indigo.algorithms.beam_search import beam_search
from indigo.nn.base.sinkhorn import matching
from indigo.birkoff import birkhoff_von_neumann
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os


np.set_printoptions(threshold=np.inf)

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = tf.zeros(shape)
        self.var = tf.ones(shape)
        self.count = epsilon

    def update(self, x):
        batch_mean = tf.reduce_mean(x, axis=0)
        batch_var = tf.reduce_std(x, axis=0) ** 2
        # We don't want gradient to propagate back through self.mean and self.var
        batch_mean = tf.stop_gradient(batch_mean)
        batch_var = tf.stop_gradient(batch_var)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = uopdate_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

# Helper function to update running mean std        
def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + tf.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

def permutation_to_pointer(permutation):
    """Converts a permutation matrix to the label distribution of
    a pointer network for training a language model

    Arguments:

    permutation: tf.Tensor
        a permutation matrix that defines the order in which
        words are inserted by the language model

    Returns:

    pointer: tf.Tensor
        a ternary matrix that contains relative positions of words
        inserted by a language model non-sequentially"""

    # make sure the permutation is an int or the below computation
    # does not make sense
    permutation = tf.cast(permutation, tf.int32)
    n = tf.shape(permutation)[-1]

    # this first section will convert the one-hot style indexing to
    # a ternary indexing where -1 means insert to the right of
    # and 1 means insert to the left of word x
    unsorted_relative = -tf.math.cumsum(
        permutation, axis=-1, exclusive=True) + tf.math.cumsum(
            permutation, axis=-1, exclusive=True, reverse=True)

    # sort the relative positions into the decoding order induced
    # by the permutation
    sorted_relative = tf.matmul(
        permutation, unsorted_relative, transpose_b=True)

    # get the one hot distribution of pointer labels; should contain
    # a sparse lower triangular matrix
    return tf.one_hot(tf.cast(
        tf.reduce_sum(tf.maximum(0, tf.linalg.band_part(
            sorted_relative, 0, -1)), axis=-2), tf.int32), n)


def permutation_to_relative(permutation):
    """Converts a permutation matrix to a relative position
    matrix for training a language model

    Arguments:

    permutation: tf.Tensor
        a permutation matrix that defines the order in which
        words are inserted by the language model

    Returns:

    relative: tf.Tensor
        a ternary matrix that contains relative positions of words
        inserted by a language model non-sequentially"""

    # make sure the permutation is an int or the below computation
    # does not make sense
    permutation = tf.cast(permutation, tf.int32)

    # this first section will convert the one-hot style indexing to
    # a ternary indexing where -1 means insert to the right of
    # and 1 means insert to the left of word x
    unsorted_relative = -tf.math.cumsum(
        permutation, axis=-1, exclusive=True) + tf.math.cumsum(
            permutation, axis=-1, exclusive=True, reverse=True)

    # sort the relative positions into the decoding order induced
    # by the permutation
    sorted_relative = tf.matmul(
        permutation, unsorted_relative, transpose_b=True)

    # get the one hot distribution of relative positions; contains
    # a one at location i when [left, center, right]_i
    return tf.one_hot(sorted_relative + 1, 3)

def prepare_batch(batch, vocab_size, permutation_generator=None):
    
    """Transform a batch dictionary into a dataclass standard format
    for the transformer to process

    Arguments:

    batch: dict of tf.Tensors
        a dictionary that contains tensors from a tfrecord dataset;
        this function assumes region-features are used
    vocab_size: tf.Tensor
        the number of words in the vocabulary of the model; used in order
        to calculate labels for the language model logits
    baseline_order: str
        the baseline ordering (l2r or r2l) to use, if permutation_generator is None
    permutation_generator: PermutationTransformer
        the PermutationTransformer that generates soft permutation from
        ground truth sentence

    Returns:

    inputs: TransformerInput
        the input to be passed into a transformer model with attributes
        necessary for also computing the loss function"""

    # select all relevant features from the batch dictionary
    image_ind = batch["image_indicators"]
    boxes_features = batch["boxes_features"]
    boxes = batch["boxes"]
    detections = batch["labels"]
    words = batch["words"]
    token_ind = batch["token_indicators"]

    # build a region feature input for the first layer of the model
    region = RegionFeatureInput(features=boxes_features,
                                boxes=boxes,
                                detections=detections)

    # build the inputs to the transformer model by left
    # shifting the target sequence
    inputs = TransformerInput(
        queries=words[:, :-1],
        values=region,
        queries_mask=tf.greater(token_ind[:, :-1], 0),
        values_mask=tf.greater(image_ind, 0))

    # this assignment is necessary for the pointer after logits layer
    # used in Transformer-InDIGO
    inputs.ids = words[:, 1:]
    inputs.logits_labels = tf.one_hot(
        words[:, 1:], tf.cast(vocab_size, tf.int32))
    
    if permutation_generator is None:
        return inputs, None
    else:
        start_end_or_pad = tf.logical_or(tf.equal(
        words, 0), tf.logical_or(tf.equal(words, 2), tf.equal(words, 3)))

        # build the inputs to the transformer model by left
        # shifting the target sequence
        permu_inputs = TransformerInput(
            queries=words,
            values=region,
            queries_mask=tf.logical_not(start_end_or_pad),
            values_mask=tf.greater(image_ind, 0))
        
        return inputs, permu_inputs

def train_faster_rcnn_dataset(train_folder,
                              validate_folder,
                              batch_size,
                              beam_size,
                              num_epoch,
                              baseline_order,
                              vocab,
                              model,
                              permutation_generator,
                              model_ckpt,
                              permutations_per_batch,
                              use_policy_gradient,
                              entropy_coeff,
                              use_birkhoff_von_neumann):
    
    """Trains a transformer based caption model using features extracted
    using a facter rcnn object detection model

    Arguments:

    train_folder: str
        the path to a folder that contains tfrecord files
        ready to be loaded from the disk;
        used for training
    validate_folder: str
        the path to a folder that contains tfrecord files
        ready to be loaded from the disk;
        used for validation
    batch_size: int
        the maximum number of training examples in a
        single batch
    beam_size: int
        the maximum number of beams to use when decoding in a
        single batch
    num_epochs: int
        the number of loops through the entire dataset to
        make before termination
    baseline_order: str
        the autoregressive ordering to train Transformer-InDIGO using
        l2r or r2l; if permutation_generator is None, this baseline order
        is applied, otherwise the ordering is given by the output
        of permutation_generator
    vocab: Vocabulary
        the model vocabulary which contains mappings
        from words to integers
    model: Decoder
        the caption model to be trained; an instance of Transformer that
        returns a data class TransformerInput
    permutation_generator: Decoder
        a network that generates soft-permutation to permute the ground-truth
        caption; None iff args.use_permutation_generator is False
    model_ckpt: str
        the path to an existing model (Transformer + PermutationTransformer) checkpoint 
        or the path to be written to when training
    permutations_per_batch: int
        number of permutation matrices to sample for each training data
    use_policy_gradient: bool
        whether to use policy gradient to train the permutation_generator;
        if use_policy_gradient is True, then permutation_generator must not be None
        if use_birkhoff_von_neumann is False, the probability is determined by the 
        Gumbel-Matching distribution (i.e. exp(<X,P>_F), see https://arxiv.org/abs/1802.08665);
        if use_birkhoff_von_neumann is True, then probability is determined by the 
        weight of Birkhoff Von Neumann decomposition
    entropy_coeff: float
        the entropy regularization coefficient for policy gradient, if 
        use_policy_gradient==True
    use_birkhoff_von_neumann: bool
        whether to use Birkhoff Von Neumann decomposition to get a distribution
        of hard permutation matrices along with their probability"""

    # create a training pipeline
    init_lr = 0.001
    optim = tf.keras.optimizers.Adam(learning_rate=init_lr)
    if use_policy_gradient:
        permu_gen_optim = tf.keras.optimizers.Adam(learning_rate=init_lr)
        reward_normalizer = RunningMeanStd()
    train_dataset = faster_rcnn_dataset(train_folder, batch_size)
    validate_dataset = faster_rcnn_dataset(validate_folder, batch_size)

    # this loss function is applied if we are not using policy gradient, 
    # or if we are using policy gradient and we are sampling from Birkhoff 
    # von Neumann distribution; in these two cases, the permutation probability 
    # normalizing factor equals to 1    
    def loss_function_no_prob_normalization_init(it, b, verbose=False):
           
        # process the dataset batch dictionary into the standard
        # model input format
        inputs, permu_inputs = prepare_batch(b, vocab.size(), permutation_generator)
        token_ind = b['token_indicators']

        def loss_function():
            if permutation_generator is None:
                # this assignment corresponds to left-to-right dncoding; note that
                # the end token must ALWAYS be decoded last and also the start
                # token must ALWAYS be decoded first
                if baseline_order == 'r2l':  # corresponds to right-to-left decoding
                    ind = tf.tile(tf.range(tf.shape(
                        token_ind)[1] - 1)[tf.newaxis], [tf.shape(token_ind)[0], 1])
                    ind = tf.reverse_sequence(ind, tf.cast(tf.reduce_sum(
                        token_ind, axis=1), tf.int32) - 2, seq_axis=1, batch_axis=0)
                    ind = tf.concat([tf.fill([
                        tf.shape(token_ind)[0], 1], 0), 1 + ind], axis=1)

                if baseline_order == 'l2r':  # corresponds to left-to-right decoding
                    ind = tf.tile(tf.range(tf.shape(
                        token_ind)[1])[tf.newaxis], [tf.shape(token_ind)[0], 1])

                # convert permutation indices into a matrix
                inputs.permutation = tf.one_hot(ind, tf.shape(token_ind)[1])
            else:
                # use the permutation generator to generate soft-permutation 
                # for the ground truth sentence;
                permutation = permutation_generator.call(permu_inputs) # Soft-permutation from Sinkhorn operator
                inputs.permutation = permutation

            # apply the Birkhoff-von Neumann decomposition to support general
            # doubly stochastic matrices; note that if inputs.permutation is 
            # already a hard permutation matrix, the decomposition returns the
            # same permutation matrix with probability 1
            permus, distribs = birkhoff_von_neumann(inputs.permutation)
            # since we set all invalid permutations to have probability zero,
            # we need to normalize the distribution here, as the sum of probability
            # of all valid permutations can be smaller than 1
            distribs = distribs / tf.reduce_sum(distribs, axis=1, keepdims=True)

            # this applies when we are not using policy gradient and not using
            # Birkhoff-von Neumann decomposition (e.g. left-to-right decoding)
            # or if we are not using policy gradient but using Birkhoff von 
            # Neumann decomposition
            if not use_policy_gradient:
                # convert the permutation to absolute positions
                inputs.absolute_positions = inputs.permutation[:, :-1, :-1]

                # convert the permutation to relative positions
                inputs.relative_positions = tf.reduce_sum(
                    permutation_to_relative(permus) * distribs[
                        ..., tf.newaxis, tf.newaxis, tf.newaxis], axis=1)[:, :-1, :-1, :]

                # convert the permutation to label distributions
                inputs.pointer_labels = tf.reduce_sum(
                    permutation_to_pointer(permus) * distribs[
                        ..., tf.newaxis, tf.newaxis], axis=1)[:, 1:, 1:]
                inputs.logits_labels = tf.matmul(
                    inputs.permutation[:, 1:, 1:], inputs.logits_labels)        
            else:
                # construct categorical distribution over the distribution
                # of hard-permutation matrices according to 
                # Birkhoff von Neumann decomposition
                tfp_distribs = tfp.distributions.Categorical(probs=distribs)     
                
                # Sampling from Birkhoff von Neumann distribution to construct input
                indices = tfp_distribs.sample()
                row_indices = tf.range(tf.shape(indices)[0])
                full_indices = tf.stack([row_indices, indices], axis=1)
                distribs_selected = tf.reshape(tf.gather_nd(distribs, full_indices), (-1,1))
                permus_selected = tf.gather_nd(permus, full_indices)
                inputs.absolute_positions = permus_selected[:, :-1, :-1]
                inputs.relative_positions = permutation_to_relative(
                    permus_selected)[:, :-1, :-1]
                inputs.pointer_labels = permutation_to_pointer(
                    permus_selected)[:, 1:, 1:]
                inputs.logits_labels = tf.matmul(
                    permus_selected[:, 1:, 1:], inputs.logits_labels)              
                
            # calculate the loss function of the decoder model;
            # if we are not using policy gradient but we are using
            # Birkhoff von Neumann decomposition, then this decoder_loss
            # is the loss for BOTH the decoder and the permutation generator;
            # in this case, we optimize both of these networks at once using decoder_loss
            decoder_loss, _ = model.loss(inputs, training=True)
            decoder_loss = tf.reduce_sum(decoder_loss * token_ind[:, :-1], axis=1)
            decoder_loss = decoder_loss / tf.reduce_sum(token_ind[:, :-1], axis=1)
            decoder_loss = tf.reduce_mean(decoder_loss)
            # if we are using policy gradient, calculate the loss function of 
            # the permutation generator; decoder is optimized using decoder_loss
            # and permutation generator is optimized using permutation_loss
            if use_policy_gradient:
                # raw reward
                reward_unnorm = model.final_layer.label_log_prob
                reward_normalizer.update(reward_unnorm)
                # normalized reward
                reward_norm = (reward_unnorm - reward_normalizer.mean) \
                                / (tf.sqrt(reward_normalizer.var) + 1e-4)
                # Permutation loss with entropy
                permutation_loss = -tf.reduce_mean(tf.log(distribs_selected) * reward_norm) 
                entropy_reg = tf.reduce_mean(entropy_coeff * tfp_distribs.entropy())  
                permutation_loss -= entropy_reg
            
            if verbose:
                if not use_policy_gradient:
                    print('It: {} Train Loss: {}'.format(it, decoder_loss))
                else:
                    print('It: {} Decoder Loss: {} Permutation Loss: {} Entropy: {}' \
                          .format(it, decoder_loss, permutation_loss, entropy_reg))
            if use_policy_gradient:
                return decoder_loss, permutation_loss
            else:
                return decoder_loss, None
                
        # occasionally do some extra processing during training
        # such as printing the labels and model predictions
        def decode():
            
            # process the dataset batch dictionary into the standard
            # model input format
            inputs, _ = prepare_batch(b, vocab.size(), None)

            # calculate the ground truth sequence for this batch; and
            # perform beam search using the current model
            out = tf.strings.reduce_join(
                vocab.ids_to_words(inputs.ids), axis=1, separator=' ')
            cap, log_p = beam_search(
                inputs, model, beam_size=beam_size, max_iterations=20)
            cap = tf.strings.reduce_join(
                vocab.ids_to_words(cap), axis=2, separator=' ')

            # show several model predicted sequences and their likelihoods
            for i in range(cap.shape[0]):
                print("Label: {}".format(out[i].numpy().decode('utf8')))
                for c, p in zip(cap[i].numpy(), tf.math.exp(log_p)[i].numpy()):
                    print("[p = {}] Model: {}".format(p, c.decode('utf8')))

        return loss_function, decode
    
    # this loss function is applied only if we are using policy gradient 
    # and if we are not sampling from Birkhoff von Neumann distribution; 
    # in this case, for each training data, we need to sample multiple
    # hard permutation matrices in order to obtain an approximation
    # of the distribution of permutation matrices with potential
    # proportional to exp(<X,P>_F)
    def loss_function_with_prob_normalization_init(it, b, verbose=False):
        # TODO
        raise NotImplementedError

    # run an initial forward pass using the model in order to build the
    # weights and define the shapes at every layer
    for batch in train_dataset.take(1):
        if use_policy_gradient and not use_birkhoff_von_neumann:
            loss_function, decode = \
            loss_function_with_prob_normalization_init(-1, batch, verbose=False)
        else:
            loss_function, decode = \
            loss_function_no_prob_normalization_init(-1, batch, verbose=False)
        loss_function()

    # restore an existing model if one exists and create a directory
    # if the ckpt directory does not exist
    tf.io.gfile.makedirs(os.path.dirname(model_ckpt))
    if tf.io.gfile.exists(model_ckpt + '.index'):
        model.load_weights(model_ckpt)
    if permutation_generator is not None:
        if tf.io.gfile.exists(model_ckpt + '.order.index'):
            permutation_generator.load_weights(model_ckpt + '.order')

    # set up variables for early stopping; only save checkpoints when
    # best validation loss has improved
    best_loss = 999999.0
    var_list = model.trainable_variables
    if permutation_generator is not None:
        permu_gen_var_list = permutation_generator.trainable_variables

    # training for a pre specified number of epochs while also annealing
    # the learning rate linearly towards zero
    iteration = 0
    for epoch in range(num_epoch):

        # loop through the entire dataset once (one epoch)
        for batch in train_dataset:
            
            if use_policy_gradient and not use_birkhoff_von_neumann:
                loss_function, decode = \
                loss_function_with_prob_normalization_init(iteration, batch, verbose=False)
            else:
                loss_function, decode = \
                loss_function_no_prob_normalization_init(iteration, batch, verbose=False)
                
            for permu_itr in range(permutations_per_batch):
                # keras requires the loss be a function
                if use_policy_gradient:
                    optim.minimize(lambda: loss_function()[0], var_list)
                    permu_gen_optim.minimize(lambda: loss_function()[1], permu_gen_var_list)
                elif permutation_generator is not None:
                    optim.minimize(lambda: loss_function()[0], var_list + permu_gen_var_list)
                else:
                    optim.minimize(lambda: loss_function()[0], var_list)
            
            if iteration % (100 // permutations_per_batch) == 0:
                decode()
                
            # increment the number of training steps so far; note this
            # does not save with the model and is reset when loading a
            # pre trained model from the disk
            iteration += 1

        # anneal the model learning rate after an epoch
        optim.lr.assign(init_lr * (1 - (epoch + 1) / num_epoch))
        if permutation_generator is not None:
            permu_gen_optim.lr.assign(init_lr * (1 - (epoch + 1) / num_epoch))

        # keep track of the validation loss
        validation_loss = 0.0
        denom = 0.0

        # loop through the entire dataset once (one epoch)
        for batch in validate_dataset:

            # accumulate the validation loss across the entire dataset
            n = tf.cast(tf.shape(batch['words'])[0], tf.float32)
            if use_policy_gradient and not use_birkhoff_von_neumann:
                loss_function, decode = \
                loss_function_with_prob_normalization_init(iteration, batch, verbose=False)
            else:
                loss_function, decode = \
                loss_function_no_prob_normalization_init(iteration, batch, verbose=False)    
            decoder_loss, permutation_loss = loss_function()
            validation_loss += decoder_loss * n
            denom += n

        # normalize the validation loss per validation example
        validation_loss = validation_loss / denom
        print('It: {} Val Loss: {}'.format(iteration, validation_loss))

        # save once at the end of every epoch; but only save when
        # the validation loss becomes smaller
        if best_loss > validation_loss:
            best_loss = validation_loss
            model.save_weights(model_ckpt, save_format='tf')
            if permutation_generator is not None:
                permutation_generator.save_weights(model_ckpt + '.order', 
                                                   save_format='tf')
