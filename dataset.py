import argparse
from functools import partial
from pathlib import Path
from typing import Tuple

import tensorflow as tf
import tqdm

from prepare_tfrecords import read_tfrecord
from transforms import resize_and_crop_image


def build_dataset(tfrec_root: Path, image_size: Tuple[int, int]):
    """ image_size is height, width tuple.
    """
    options_no_order = tf.data.Options()
    options_no_order.experimental_deterministic = False
    AUTO = tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(
        [str(p) for p in tfrec_root.glob('train-*.tfrec')],
        num_parallel_reads=AUTO)
    dataset = dataset.with_options(options_no_order)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    dataset = dataset.map(
        partial(resize_and_crop_image, target_size=image_size),
        num_parallel_calls=AUTO)
    dataset = dataset.shuffle(2048)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('tfrec_root', type=Path)
    arg('--image-size',
        type=lambda x: tuple(map(int, x.split('x'))),
        default=(224, 224))
    args = parser.parse_args()

    dataset = build_dataset(args.tfrec_root, image_size=args.image_size)
    for _ in tqdm.tqdm(dataset):
        pass


if __name__ == '__main__':
    main()
