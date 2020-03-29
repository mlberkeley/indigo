from indigo.core.train import train_faster_rcnn_dataset
from indigo.models.transformer import Transformer
from indigo.process.caption import Vocabulary
import tensorflow as tf
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tfrecord_folder', type=str, default='tfrecord')
    parser.add_argument(
        '--batch_size', type=int, default=128)
    parser.add_argument(
        '--vocab_file', type=str, default='vocab.txt')
    parser.add_argument(
        '--num_epochs', type=int, default=1000)
    parser.add_argument(
        '--model_ckpt', type=str, default='ckpt/decoder')
    parser.add_argument(
        '--embedding_size', type=int, default=1024)
    parser.add_argument(
        '--num_layers', type=int, default=2)
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
                        first_layer='region',
                        final_layer='logits')

    train_faster_rcnn_dataset(args.tfrecord_folder,
                              args.batch_size,
                              args.num_epochs,
                              model,
                              args.model_ckpt,
                              vocab)
