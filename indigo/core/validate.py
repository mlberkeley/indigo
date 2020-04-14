from indigo.data.load import faster_rcnn_dataset
from indigo.nn.input import TransformerInput
from indigo.nn.input import RegionFeatureInput
from indigo.algorithms.beam_search import beam_search
import tensorflow as tf
import os
import numpy as np


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

    # build a region feature input for the first layer of the model
    region = RegionFeatureInput(features=boxes_features,
                                boxes=boxes,
                                detections=detections)

    # build the inputs to the transformer model by left
    # shifting the target sequence
    inputs = TransformerInput(
        values=region,
        values_mask=tf.greater(image_ind, 0.))

    return inputs


def validate_faster_rcnn_dataset(tfrecord_folder,
                                 ref_folder,
                                 batch_size,
                                 beam_size,
                                 model,
                                 model_ckpt,
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
    vocab: Vocabulary
        the model vocabulary which contains mappings
        from words to integers"""

    # create a validation pipeline
    dataset = faster_rcnn_dataset(tfrecord_folder, batch_size)
    model.load_weights(model_ckpt)

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
        cap, log_p = beam_search(
            prepare_batch(batch), model, beam_size=beam_size, max_iterations=20)
        cap = tf.strings.reduce_join(
            vocab.ids_to_words(cap), axis=2, separator=' ').numpy()

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

    # convert the dictionaries into lists for nlg eval input format
    ref_caps_list = []
    hyp_caps_list = []
    for key in ref_caps.keys():
        ref_caps_list.append(ref_caps[key])
        hyp_caps_list.append(hyp_caps[key])

    from nlgeval import NLGEval
    nlgeval = NLGEval()

    # compute several natural language generation metrics
    metrics = nlgeval.compute_metrics([*zip(*ref_caps_list)], hyp_caps_list)
    for key in metrics.keys():
        print("Eval/{}".format(key), metrics[key])
