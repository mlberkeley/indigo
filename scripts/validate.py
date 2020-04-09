from indigo.core.validate import validate_faster_rcnn_dataset
from indigo.nn.transformer import Transformer
from indigo.process.captions import Vocabulary
import tensorflow as tf
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--validate_folder', type=str, default='tfrecord')
    parser.add_argument(
        '--ref_folder', type=str, default='captions')
    parser.add_argument(
        '--batch_size', type=int, default=5)
    parser.add_argument(
        '--vocab_file', type=str, default='vocab.txt')
    parser.add_argument(
        '--model_ckpt', type=str, default='ckpt/decoder')
    parser.add_argument(
        '--embedding_size', type=int, default=256)
    parser.add_argument(
        '--num_layers', type=int, default=1)
    parser.add_argument(
        '--first_layer', type=str,
        default='region', choices=['region', 'discrete', 'continuous'])
    parser.add_argument(
        '--final_layer', type=str,
        default='indigo', choices=['indigo', 'logits'])
    args = parser.parse_args()

    with tf.io.gfile.GFile(args.vocab_file, "r") as f:
        vocab = Vocabulary([x.strip() for x in f.readlines()],
                           unknown_word="<unk>",
                           unknown_id=1)

    model = Transformer(vocab.size(),
                        args.embedding_size,
                        4,
                        args.num_layers,
                        queries_dropout=0.,
                        values_dropout=0.,
                        causal=True,
                        first_layer=args.first_layer,
                        final_layer=args.final_layer)

    validate_faster_rcnn_dataset(args.validate_folder,
                                 args.ref_folder,
                                 args.batch_size,
                                 model,
                                 args.model_ckpt,
                                 vocab)
