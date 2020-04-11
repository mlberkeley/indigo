# Transformer-InDIGO

(Unofficial) Implementation of the Transformer-InDIGO paper in TemsorFlow 2: https://arxiv.org/abs/1905.12790

How effective is the monolithic left-to-right decoding strategy employed by modern language modeling systems, and what are the alternatives? In this project, we explore the viability and implications of non-sequential and partially-autoregressive language models. This package is our research framework. Have Fun! -Brandon

## Installation

To install this package, first download the package from github, then install it using pip.

```bash
git clone git@github.com:brandontrabucco/indigo.git
pip install -e indigo
```

You must then install helper packages for word tokenization and part of speech tagging. Enter the following statements into the python interpreter where you have installed our package.

```python
import nltk
nltk.download('punkt')
nltk.download('brown')
nltk.download('universal_tagset')
```

Finally, you must install the natural language evaluation package that contains several helpful metrics.

```bash
pip install git+https://github.com/Maluuba/nlg-eval.git@master
nlg-eval --setup
```

You can now start training a non-sequential model!

## Setup

In this section, we will walk you through how to create a training dataset, using COCO 2017 as an example. In the first step, you must have downloaded COCO 2017. The annotations should be placed at `~/annotations` and the images should be placed at `~/train2017` and `~/val2017` for the training and validation set respectively.

Create a part of speech tagger first.

```bash
python scripts/create_tagger.py \
    --out_tagger_file tagger.pkl
```

Extract COCO 2017 into a format compatible with our package.

```bash
python scripts/extract_coco.py \
    --out_caption_folder ~/captions_train2017 \
    --annotations_file ~/annotations/captions_train2017.json
python scripts/extract_coco.py \
    --out_caption_folder ~/captions_val2017 \
    --annotations_file ~/annotations/captions_val2017.json
```

Process the COCO 2017 captions and extract integer features on which to train a non sequential model.

```bash
python scripts/process_captions.py \
    --out_feature_folder ~/captions_train2017_features \
    --in_folder ~/captions_train2017 \
    --tagger_file tagger.pkl \
    --vocab_file train2017_vocab.txt \
    --min_word_frequency 5 \
    --max_length 100
python scripts/process_captions.py \
    --out_feature_folder ~/captions_val2017_features \
    --in_folder ~/captions_val2017 \
    --tagger_file tagger.pkl \
    --vocab_file train2017_vocab.txt \
    --max_length 100
```

Process images from the COCO 2017 dataset and extract features using a Faster RCNN FPN backbone. Note this script will distribute inference across all visible GPUs on your system.

```bash
python scripts/process_images.py \
    --out_feature_folder ~/train2017_features \
    --in_folder ~/train2017 \
    --batch_size 4
python scripts/process_images.py \
    --out_feature_folder ~/val2017_features \
    --in_folder ~/val2017 \
    --batch_size 4
```

Finally, convert the processed features into a TFRecord format for efficient training.

```bash
python scripts/create_tfrecords.py \
    --out_tfrecord_folder ~/left_to_right \
    --caption_folder ~/captions_train2017_features \
    --image_folder ~/train2017_features \
    --samples_per_shard 4096
```

The dataset has been created, and you can start training.

## Training

You may train a model with the following command.

```bash
python scripts/train.py \
    --train_folder ~/left_to_right \
    --validate_folder ~/left_to_right \
    --batch_size 32 \
    --vocab_file train2017_vocab.txt \
    --num_epochs 10 \
    --model_ckpt ckpt/decoder
```

## Validation

You may evaluate a training model with the following command.

```bash
python scripts/validate.py \
    --validate_folder ~/left_to_right \
    --ref_folder ~/captions_train2017 \
    --batch_size 32 \
    --vocab_file train2017_vocab.txt \
    --model_ckpt ckpt/decoder
```
