from indigo.data.load import faster_rcnn_dataset
from indigo.nn.input import TransformerInput
from indigo.nn.input import RegionFeatureInput
from indigo.algorithms.beam_search import beam_search
import tensorflow as tf
import os


def train_faster_rcnn_dataset(tfrecord_folder,
                              batch_size,
                              num_epoch,
                              model,
                              model_ckpt,
                              vocab):
    """Trains a transformer based caption model using features extracted
    using a facter rcnn object detection model

    Arguments:

    tfrecord_folder: str
        the path to a folder that contains tfrecord files
        ready to be loaded from the disk
    batch_size: int
        the maximum number of training examples in a
        single batch
    num_epochs: int
        the number of loops through the entire dataset to
        make before termination
    model: Decoder
        the caption model to be trained
    model_ckpt: str
        the path to an existing model checkpoint or the path
        to be written to when training
    vocab: Vocabulary
        the model vocabulary which contains mappings
        from words to integers"""

    # create a training pipeline
    init_lr = 0.001
    dataset = faster_rcnn_dataset(tfrecord_folder, batch_size)
    optim = tf.keras.optimizers.Adam(learning_rate=init_lr)

    def loss_function(iteration, batch):

        # select all relevant image features
        image_indicators = batch["image_indicators"]
        boxes_features = batch["boxes_features"]
        boxes = batch["boxes"]
        detections = batch["labels"]

        # select all relevant language features
        words = batch["words"]
        token_indicators = batch["token_indicators"]

        region = RegionFeatureInput(features=boxes_features,
                                    boxes=boxes,
                                    detections=detections)

        inputs = TransformerInput(
            queries=words[:, :-1],
            values=region,
            queries_mask=tf.greater(token_indicators[:, :-1], 0.),
            values_mask=tf.greater(image_indicators, 0.))

        # these labels correspond to left-to-right ordering
        inputs.ids = words[:, 1:]
        R = tf.eye(tf.shape(words)[1] - 1,
                   batch_shape=tf.shape(words)[:1], dtype=tf.int32)
        inputs.positions = tf.math.cumsum(
            R, axis=2, exclusive=True) - tf.math.cumsum(
                R, axis=2, exclusive=True, reverse=True)

        # perform a forward pass using the transformer model
        total_loss, _ = model.loss(inputs, training=True)

        total_loss = tf.reduce_sum(
            total_loss * token_indicators[:, :-1], axis=1)
        total_loss = total_loss / tf.reduce_sum(
                token_indicators[:, :-1], axis=1)

        total_loss = tf.reduce_mean(total_loss)

        print('Iteration: {} Loss: {}'.format(iteration,
                                              total_loss))

        # monitor training by printing the loss
        if iteration % 100 == 0:

            region = RegionFeatureInput(features=boxes_features,
                                        boxes=boxes,
                                        detections=detections)

            inputs = TransformerInput(
                queries=words[:, :-1],
                values=region,
                queries_mask=tf.greater(token_indicators[:, :-1], 0.),
                values_mask=tf.greater(image_indicators, 0.))

            # these labels correspond to left-to-right ordering
            inputs.ids = words[:, 1:]
            R = tf.eye(tf.shape(words)[1] - 1,
                       batch_shape=tf.shape(words)[:1], dtype=tf.int32)
            inputs.positions = tf.math.cumsum(
                R, axis=2, exclusive=True) - tf.math.cumsum(
                    R, axis=2, exclusive=True, reverse=True)

            cap, log_p = beam_search(inputs,
                                     model,
                                     beam_size=1,
                                     max_iterations=20)

            p = tf.math.exp(log_p)[0]

            cap = tf.strings.reduce_join(
                vocab.ids_to_words(cap)[0], axis=1, separator=' ')

            out = tf.strings.reduce_join(
                vocab.ids_to_words(words)[0], axis=0, separator=' ')

            print("\nGround Truth: {}".format(out.numpy().decode('utf8')))
            for c, cp in zip(cap, p):
                print("[p = {}] Prediction: {}".format(
                    cp.numpy(), c.numpy().decode('utf8')))
            print()

        return total_loss

    # restore an existing model if one exists and create a directory
    tf.io.gfile.makedirs(os.path.dirname(model_ckpt))
    if tf.io.gfile.exists(model_ckpt):
        model.load_weights(model_ckpt)

    # training for a pre specified number of epochs
    iteration = 0
    for epoch in range(num_epoch):

        # loop through the entire dataset precisely once
        for batch in dataset:

            optim.minimize(lambda: loss_function(iteration, batch),
                           model.trainable_variables)

            iteration += 1

        # save once at the end of every epoch
        model.save_weights(model_ckpt, save_format='tf')

        # anneal the model learning rate after an epoch
        optim.lr.assign(init_lr * (1 - epoch / num_epoch))
