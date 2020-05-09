from indigo.data.load import faster_rcnn_dataset
from indigo.nn.input import TransformerInput
from indigo.nn.input import RegionFeatureInput
from indigo.algorithms.beam_search import beam_search
from indigo.birkoff_utils import birkhoff_von_neumann
from indigo.permutation_utils import permutation_to_pointer
from indigo.permutation_utils import permutation_to_relative
from indigo.permutation_utils import get_permutation
import tensorflow as tf
import numpy as np
import os


np.set_printoptions(threshold=np.inf)


def prepare_batch_for_lm(batch):
    """Transform a batch dictionary into a dataclass standard format
    for the transformer to process

    Arguments:

    batch: dict of tf.Tensors
        a dictionary that contains tensors from a tfrecord dataset;
        this function assumes region-features are used
    vocab_size: tf.Tensor
        the number of words in the vocabulary of the model; used in order
        to calculate labels for the language model logits

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
    mask = batch["token_indicators"]

    # build a region feature input for the first layer of the model
    region = RegionFeatureInput(features=boxes_features,
                                boxes=boxes,
                                detections=detections)

    # build the inputs to the transformer model by left
    # shifting the target sequence
    inputs = TransformerInput(
        queries=words[:, :-1],
        values=region,
        queries_mask=tf.greater(mask[:, :-1], 0),
        values_mask=tf.greater(image_ind, 0))

    # this assignment is necessary for the pointer after logits layer
    # used in Transformer-InDIGO
    inputs.ids = words[:, 1:]

    return inputs


def prepare_batch_for_pt(batch):
    """Transform a batch dictionary into a dataclass standard format
    for the transformer to process

    Arguments:

    batch: dict of tf.Tensors
        a dictionary that contains tensors from a tfrecord dataset;
        this function assumes region-features are used
    vocab_size: tf.Tensor
        the number of words in the vocabulary of the model; used in order
        to calculate labels for the language model logits

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

    # build a region feature input for the first layer of the model
    region = RegionFeatureInput(features=boxes_features,
                                boxes=boxes,
                                detections=detections)

    start_end_or_pad = tf.logical_or(tf.equal(
        words, 0), tf.logical_or(tf.equal(words, 2), tf.equal(words, 3)))

    # build the inputs to the transformer model by left
    # shifting the target sequence
    inputs = TransformerInput(
        queries=words,
        values=region,
        queries_mask=tf.logical_not(start_end_or_pad),
        values_mask=tf.greater(image_ind, 0))

    return inputs


def prepare_permutation(batch,
                        vocab_size,
                        order):
    """Transform a batch dictionary into a dataclass standard format
    for the transformer to process

    Arguments:

    batch: dict of tf.Tensors
        a dictionary that contains tensors from a tfrecord dataset;
        this function assumes region-features are used
    vocab_size: tf.Tensor
        the number of words in the vocabulary of the model; used in order
        to calculate labels for the language model logits
    order: str or callable
        the autoregressive ordering to train Transformer-InDIGO using;
        l2r or r2l for now, will support soft orders later

    Returns:

    inputs: TransformerInput
        the input to be passed into a transformer model with attributes
        necessary for also computing the loss function"""

    # process the dataset batch dictionary into the standard
    # model input format
    inputs = prepare_batch_for_lm(batch)

    # the order is fixed
    if order in ['r2l', 'l2r', 'rare', 'common']:
        inputs.permutation = get_permutation(
            batch['token_indicators'], batch['words'], tf.constant(order))

    # pass the training example through the permutation transformer
    # to obtain a doubly stochastic matrix
    if isinstance(order, tf.keras.Model):  # corresponds to soft orderings
        inputs.permutation = order(prepare_batch_for_pt(batch))

    # apply the birkhoff-von neumann decomposition to support general
    # doubly stochastic matrices
    p, c = birkhoff_von_neumann(inputs.permutation, tf.constant(20))
    c = (c + 1e-7) / tf.reduce_sum(c + 1e-7, axis=1, keepdims=True)
    c = tf.stop_gradient(c)

    # convert the permutation to absolute and relative  positions
    inputs.absolute_positions = inputs.permutation[:, :-1, :-1]
    inputs.relative_positions = tf.reduce_sum(permutation_to_relative(
        p) * c[..., tf.newaxis, tf.newaxis, tf.newaxis], axis=1)

    # convert the permutation to label distributions
    inputs.pointer_labels = tf.reduce_sum(
        permutation_to_pointer(p) * c[..., tf.newaxis, tf.newaxis], axis=1)
    inputs.logits_labels = tf.matmul(inputs.permutation[
        :, 1:, 1:], tf.one_hot(inputs.ids, tf.cast(vocab_size, tf.int32)))

    return inputs


def train_faster_rcnn_dataset(train_folder,
                              validate_folder,
                              batch_size,
                              beam_size,
                              num_epoch,
                              model,
                              model_ckpt,
                              order,
                              vocab):
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
    model: Decoder
        the caption model to be trained; an instance of Transformer that
        returns a data class TransformerInput
    model_ckpt: str
        the path to an existing model checkpoint or the path
        to be written to when training
    order: str
        the autoregressive ordering to train Transformer-InDIGO using;
        l2r or r2l for now, will support soft orders later
    vocab: Vocabulary
        the model vocabulary which contains mappings
        from words to integers"""

    # create a training pipeline
    init_lr = 0.001
    optim = tf.keras.optimizers.Adam(learning_rate=init_lr)
    train_dataset = faster_rcnn_dataset(train_folder, batch_size)
    validate_dataset = faster_rcnn_dataset(validate_folder, batch_size)

    def loss_function(it, b, verbose=False):

        # process the dataset batch dictionary into the standard
        # model input format
        inputs = prepare_permutation(b, vocab.size(), order)
        mask = b['token_indicators']
        loss, _ = model.loss(inputs, training=True)
        loss = tf.reduce_sum(loss * mask[:, 1:], axis=1)
        loss = loss / tf.reduce_sum(mask[:, 1:], axis=1)
        loss = tf.reduce_mean(loss)
        if verbose:
            print('It: {} Train Loss: {}'.format(it, loss))
        return loss

    def decode(b):

        # calculate the ground truth sequence for this batch; and
        # perform beam search using the current model
        inputs = prepare_batch_for_lm(b)
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

    def validate():

        # accumulate the validation loss across the entire dataset
        # weight the loss by the batch size and normalize
        # the loss to an expected value
        denom, loss = 0.0, 0.0
        for b in validate_dataset:
            n = tf.cast(tf.shape(b['words'])[0], tf.float32)
            denom, loss = denom + n, loss + n * loss_function(0, b)
        return loss / denom

    # run an initial forward pass using the model in order to build the
    # weights and define the shapes at every layer
    for batch in train_dataset.take(1):
        loss_function(-1, batch, verbose=False)

    # restore an existing model if one exists and create a directory
    # if the ckpt directory does not exist
    tf.io.gfile.makedirs(os.path.dirname(model_ckpt))
    if tf.io.gfile.exists(model_ckpt + '.index'):
        model.load_weights(model_ckpt)
    if tf.io.gfile.exists(model_ckpt + '.order.index'):
        order.load_weights(model_ckpt + '.order')

    # set up variables for early stopping; only save checkpoints when
    # best validation loss has improved
    best_loss = validate()
    var_list = model.trainable_variables
    if isinstance(order, tf.keras.Model):
        var_list = var_list + order.trainable_variables

    # training for a pre specified number of epochs while also annealing
    # the learning rate linearly towards zero
    iteration = 0
    for epoch in range(num_epoch):

        # loop through the entire dataset once (one epoch)
        for batch in train_dataset:

            # keras requires the loss be a function
            optim.minimize(lambda: loss_function(
                iteration, batch, verbose=True), var_list)
            if iteration % 100 == 0:
                decode(batch)

            # increment the number of training steps so far; note this
            # does not save with the model and is reset when loading a
            # pre trained model from the disk
            iteration += 1

        # anneal the model learning rate after an epoch
        optim.lr.assign(init_lr * (1 - (epoch + 1) / num_epoch))

        # normalize the validation loss per validation example
        validation_loss = validate()
        print('It: {} Val Loss: {}'.format(iteration, validation_loss))

        # save once at the end of every epoch; but only save when
        # the validation loss becomes smaller
        if best_loss > validation_loss:
            best_loss = validation_loss
            model.save_weights(model_ckpt, save_format='tf')
            if isinstance(order, tf.keras.Model):
                order.save_weights(model_ckpt + '.order')
