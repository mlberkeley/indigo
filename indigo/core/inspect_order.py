from indigo.data.load import faster_rcnn_dataset
from indigo.nn.input import TransformerInput
from indigo.nn.input import RegionFeatureInput
from indigo.algorithms.beam_search import beam_search
import tensorflow as tf
import os
import numpy as np


def prepare_batch_for_lm(batch):
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

    # build a region feature input for the first layer of the model
    region = RegionFeatureInput(features=boxes_features,
                                boxes=boxes,
                                detections=detections)

    # build the inputs to the transformer model by left
    # shifting the target sequence
    inputs = TransformerInput(
        values=region,
        values_mask=tf.greater(image_ind, 0))

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


def inspect_order_faster_rcnn_dataset(tfrecord_folder,
                                      ref_folder,
                                      batch_size,
                                      beam_size,
                                      model,
                                      model_ckpt,
                                      order,
                                      vocab):
    """Trains a transformer based caption model using features extracted
    using a facter rcnn object detection model

    Arguments:

    tfrecord_folder: str
        the path to a folder that contains tfrecord files
        ready to be loaded from the disk
    ref_folder: str
        the path to a folder that contains ground truth sentence files
        ready to be loaded from the disk
    batch_size: int
        the maximum number of training examples in a
        single batch
    beam_size: int
        the maximum number of beams to use when decoding in a
        single batch
    model: Decoder
        the caption model to be validated; an instance of Transformer that
        returns a data class TransformerInput
    model_ckpt: str
        the path to an existing model checkpoint or the path
        to be written to when training
    order: tf.keras.Model
        the autoregressive ordering to train Transformer-InDIGO using;
        must be a keras model that returns permutations
    vocab: Vocabulary
        the model vocabulary which contains mappings
        from words to integers"""

    # create a validation pipeline
    dataset = faster_rcnn_dataset(tfrecord_folder, batch_size, shuffle=False)
    model.load_weights(model_ckpt)
    if isinstance(order, tf.keras.Model):
        order.load_weights(model_ckpt + '.order')

    ref_caps = {}
    hyp_caps = {}

    # loop through the entire dataset once (one epoch)
    for batch in dataset:

        # for every element of the batch select the path that
        # corresponds to ground truth words
        paths = [x.decode("utf-8") for x in batch["image_path"].numpy()]
        paths = [os.path.join(ref_folder,  os.path.basename(x)[:-7] + "txt")
                 for x in paths]

        # iterate through every ground truth training example and
        # select each row from the text file
        for file_path in paths:
            with tf.io.gfile.GFile(file_path, "r") as f:
                ref_caps[file_path] = [
                    x for x in f.read().strip().lower().split("\n")
                    if len(x) > 0]

        # process the dataset batch dictionary into the standard
        # model input format; perform beam search
        inputs = prepare_batch_for_lm(batch)
        cap, log_p, perm = beam_search(
            inputs, model, beam_size=beam_size, max_iterations=20)
        cap = tf.strings.reduce_join(
            vocab.ids_to_words(cap), axis=2, separator=' ').numpy()

        # calculate the order induced by the language model decoding
        pos = inputs.relative_positions
        pos = tf.argmax(pos, axis=-1, output_type=tf.int32) - 1
        pos = tf.reduce_sum(tf.nn.relu(pos[:, 1:, 1:]), axis=1)
        pos = tf.one_hot(pos, tf.shape(pos)[1], dtype=tf.int32)
        pos = tf.reshape(pos, tf.concat([[
            batch_size, beam_size], tf.shape(pos)[1:]], 0))

        # calculate the order output by the permutation transformer
        if isinstance(order, tf.keras.Model):
            perm = order(prepare_batch_for_pt(batch))

        # format the model predictions into a string; the evaluation package
        # requires input to be strings; not there will be slight
        # formatting differences between ref and hyp
        for i in range(cap.shape[0]):
            hyp_caps[paths[i]] = cap[i, 0].decode("utf-8").replace(
                "<pad>", "").replace("<start>", "").replace(
                "<end>", "").replace("  ", " ").strip()
            print("{}: [p = {}] {}".format(paths[i],
                                           np.exp(log_p[i, 0].numpy()),
                                           hyp_caps[paths[i]]))
            print("Decoder Permutation:\n", pos[i, 0].numpy())
            if isinstance(order, tf.keras.Model):
                print("Encoder Permutation:\n", perm[i, 0].numpy())
