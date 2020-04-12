from indigo.data.load import faster_rcnn_dataset
from indigo.nn.input import TransformerInput
from indigo.nn.input import RegionFeatureInput
from indigo.algorithms.beam_search import beam_search
import tensorflow as tf
import os


def prepare_batch(batch):
    """Transform a batch dictionary into a dataclass standard format
    for the transformer to process

    Arguments:

    batch: dict of tf.Tensors
        a dictionary that contains tensors from a tfrecord dataset;
        this function assumes region-features are used

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

    # this assignment corresponds to left-to-right encoding; note that
    # the end token must ALWAYS be decoded last and also the start
    # token must ALWAYS be decoded first
    permutation = tf.eye(tf.shape(words)[1],
                         batch_shape=tf.shape(words)[:1], dtype=tf.int32)
    words = tf.squeeze(
        tf.matmul(tf.expand_dims(words, 1), permutation), 1)

    # build a region feature input for the first layer of the model
    region = RegionFeatureInput(features=boxes_features,
                                boxes=boxes,
                                detections=detections)

    # build the inputs to the transformer model by left
    # shifting the target sequence
    inputs = TransformerInput(
        queries=words[:, :-1],
        values=region,
        queries_mask=tf.greater(token_ind[:, :-1], 0.),
        values_mask=tf.greater(image_ind, 0.))

    # this assignment is necessary for the logits loss
    inputs.ids = words[:, 1:]

    # the dataset is not compiled with an encoding so one must
    # be generated on the fly during training; only
    # applies when using a pointer layer; note that we remove the final
    # row and column which corresponds to the end token
    inp = permutation[:, :-1, :-1]
    inputs.positions = tf.math.cumsum(
        inp, axis=2, exclusive=True) - tf.math.cumsum(
            inp, axis=2, exclusive=True, reverse=True)

    return inputs


def train_faster_rcnn_dataset(train_folder,
                              validate_folder,
                              batch_size,
                              beam_size,
                              num_epoch,
                              model,
                              model_ckpt,
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
    vocab: Vocabulary
        the model vocabulary which contains mappings
        from words to integers"""

    # create a training pipeline
    init_lr = 0.001
    optim = tf.keras.optimizers.Adam(learning_rate=init_lr)
    train_dataset = faster_rcnn_dataset(train_folder, batch_size)
    validate_dataset = faster_rcnn_dataset(validate_folder, batch_size)

    def loss_function(it, b, decode=False, verbose=False):

        # process the dataset batch dictionary into the standard
        # model input format
        inputs = prepare_batch(b)
        token_ind = b['token_indicators']

        # calculate the loss function using the transformer model
        total_loss, _ = model.loss(inputs, training=True)
        total_loss = tf.reduce_sum(total_loss * token_ind[:, :-1], axis=1)
        total_loss = total_loss / tf.reduce_sum(token_ind[:, :-1], axis=1)
        total_loss = tf.reduce_mean(total_loss)
        if verbose:
            print('It: {} Train Loss: {}'.format(it, total_loss))

        # occasionally do some extra processing during training
        # such as printing the labels and model predictions
        if decode:

            # process the dataset batch dictionary into the standard
            # model input format
            inputs = prepare_batch(b)

            # calculate the ground truth sequence for this batch; and
            # perform beam search using the current model
            out = tf.strings.reduce_join(
                vocab.ids_to_words(inputs.ids), axis=1, separator=' ')
            cap, log_p = beam_search(
                inputs, model, beam_size=beam_size, max_iterations=20)

            # show several model predicted sequences and their likelihoods
            for i in range(cap.shape[0]):
                print("Ground Truth: {}".format(out[i].numpy().decode('utf8')))
                cpa = tf.strings.reduce_join(
                    vocab.ids_to_words(cap)[i], axis=1, separator=' ').numpy()
                for c, p in zip(cpa, tf.math.exp(log_p)[i].numpy()):
                    print("[p = {}] Prediction: {}".format(p, c.decode('utf8')))

        return total_loss

    # run an initial forward pass using the model in order to build the
    # weights and define the shapes at every layer
    for batch in train_dataset.take(1):
        loss_function(-1, batch, decode=False, verbose=False)

    # restore an existing model if one exists and create a directory
    # if the ckpt directory does not exist
    tf.io.gfile.makedirs(os.path.dirname(model_ckpt))
    if tf.io.gfile.exists(model_ckpt + '.index'):
        model.load_weights(model_ckpt)

    # set up variables for early stopping; only save checkpoints when
    # best validation loss has improved
    best_loss = 999999.0
    var_list = model.trainable_variables

    # training for a pre specified number of epochs while also annealing
    # the learning rate linearly towards zero
    iteration = 0
    for epoch in range(num_epoch):

        # loop through the entire dataset once (one epoch)
        for batch in train_dataset:

            # keras requires the loss be a function
            optim.minimize(lambda: loss_function(
                iteration, batch, decode=iteration % 100 == 0, verbose=True),
                           var_list)

            # increment the number of training steps so far; note this
            # does not save with the model and is reset when loading a
            # pre trained model from the disk
            iteration += 1

        # anneal the model learning rate after an epoch
        optim.lr.assign(init_lr * (1 - (epoch + 1) / num_epoch))

        # keep track of the validation loss
        validation_loss = 0.0
        denom = 0.0

        # loop through the entire dataset once (one epoch)
        for batch in validate_dataset:

            # accumulate the validation loss across the entire dataset
            n = tf.cast(tf.shape(batch['words'])[0], tf.float32)
            validation_loss += loss_function(
                iteration, batch, decode=False, verbose=False) * n
            denom += n

        # normalize the validation loss per validation example
        validation_loss = validation_loss / denom
        print('It: {} Val Loss: {}'.format(iteration, validation_loss))

        # save once at the end of every epoch; but only save when
        # the validation loss becomes smaller
        if best_loss > validation_loss:
            best_loss = validation_loss
            model.save_weights(model_ckpt, save_format='tf')
