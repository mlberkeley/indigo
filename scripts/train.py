from indigo.core.train import train_faster_rcnn_dataset
from indigo.nn.transformer import Transformer
from indigo.nn.permutation_transformer import PermutationTransformer
from indigo.process.captions import Vocabulary
import tensorflow as tf
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_folder', type=str, default='tfrecords')
    parser.add_argument(
        '--validate_folder', type=str, default='tfrecords')
    parser.add_argument(
        '--batch_size', type=int, default=16)
    parser.add_argument(
        '--beam_size', type=int, default=3)
    parser.add_argument(
        '--vocab_file', type=str, default='train2017_vocab.txt')
    parser.add_argument(
        '--num_epochs', type=int, default=10)
    parser.add_argument(
        '--model_ckpt', type=str, default='ckpt/nsds')
    parser.add_argument(
        '--embedding_size', type=int, default=256)
    parser.add_argument(
        '--heads', type=int, default=4)
    parser.add_argument(
        '--num_layers', type=int, default=2)
    parser.add_argument(
        '--queries_dropout', type=float, default=0.)
    parser.add_argument(
        '--keys_dropout', type=float, default=0.)
    parser.add_argument(
        '--values_dropout', type=float, default=0.)
    parser.add_argument(
        '--first_layer', type=str,
        default='region', choices=['region', 'discrete', 'continuous'])
    parser.add_argument(
        '--final_layer', type=str,
        default='indigo', choices=['indigo', 'logits'])
    parser.add_argument(
        '--use_permutation_generator', action='store_true')
    parser.add_argument(
        '--permutations_per_batch', type=int, default=1)
    parser.add_argument(
        '--use_policy_gradient', action='store_true')
    parser.add_argument(
        '--entropy_coeff', type=float, default=0.)
    parser.add_argument(
        '--use_birkhoff_von_neumann', action='store_true')
    parser.add_argument(
        '--baseline_order', type=str,
        default='l2r', choices=['l2r', 'r2l'])
    
    args = parser.parse_args()

    if args.use_policy_gradient or args.use_birkhoff_von_neumann:
        assert args.use_permutation_generator
    if args.permutations_per_batch > 1 or args.entropy_coeff > 0.0:
        assert args.use_policy_gradient
        
    with tf.io.gfile.GFile(args.vocab_file, "r") as f:
        vocab = Vocabulary([x.strip() for x in f.readlines()],
                           unknown_word="<unk>",
                           unknown_id=1)

    if args.use_permutation_generator:
        permutation_generator = PermutationTransformer(vocab.size(),
                        args.embedding_size,
                        args.heads,
                        args.num_layers,
                        queries_dropout=args.queries_dropout,
                        keys_dropout=args.keys_dropout,
                        values_dropout=args.values_dropout,
                        first_layer=args.first_layer)
    else:
        permutation_generator = None
        
    model = Transformer(vocab.size(),
                        args.embedding_size,
                        args.heads,
                        args.num_layers,
                        queries_dropout=args.queries_dropout,
                        keys_dropout=args.keys_dropout,
                        values_dropout=args.values_dropout,
                        causal=True,
                        first_layer=args.first_layer,
                        final_layer=args.final_layer)

    train_faster_rcnn_dataset(args.train_folder,
                              args.validate_folder,
                              args.batch_size,
                              args.beam_size,
                              args.num_epochs,
                              args.baseline_order,
                              vocab,
                              model,
                              permutation_generator,
                              args.model_ckpt,
                              args.permutations_per_batch,
                              args.use_policy_gradient,
                              args.entropy_coeff,
                              args.use_birkhoff_von_neumann)
