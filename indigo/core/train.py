from indigo.data.load import faster_rcnn_dataset
from indigo.input import TransformerInput
from indigo.input import RegionFeatureInput
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
    dataset = faster_rcnn_dataset(tfrecord_folder, batch_size)
    optim = tf.keras.optimizers.Adam()

    def loss_function(iteration, batch):

        # select all relevant image features
        image_indicators = batch["image_indicators"]
        boxes_features = batch["boxes_features"]
        boxes = batch["boxes"]
        detections = batch["labels"]

        # select all relevant language features
        words = batch["words"]
        token_indicators = batch["token_indicators"]

        model_features = TransformerInput(
            queries=words[:, :-1],
            values=RegionFeatureInput(features=boxes_features,
                                      boxes=boxes,
                                      detections=detections),
            queries_mask=tf.greater(token_indicators[:, :-1], 0.),
            values_mask=tf.greater(image_indicators, 0.))

        # perform a forward pass using the transformer model
        logits = model(model_features)

        # calculate the loss function for training
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            words[:, 1:], logits, from_logits=True, axis=-1)
        total_loss = (tf.reduce_sum(loss * token_indicators[:, 1:]) /
                      tf.reduce_sum(token_indicators[:, 1:]))

        # monitor training by printing the loss
        if iteration % 1 == 0:
            print('Iteration: {} Loss: {}'.format(iteration,
                                                  total_loss))

        return total_loss

    # restore an existing model if one exists and create a directory
    tf.io.gfile.makedirs(os.path.dirname(model_ckpt))
    if tf.io.gfile.exists(model_ckpt):
        model.load_weights(model_ckpt)

    # training for a pre specified number of epochs
    iteration = 0
    for epoch in range(num_epoch):
        for batch in dataset:

            optim.minimize(lambda: loss_function(iteration, batch),
                           model.trainable_variables)

            iteration += 1

        # save once at the end of every epoch
        model.save_weights(model_ckpt, save_format='tf')
