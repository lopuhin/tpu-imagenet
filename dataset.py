import argparse
from functools import partial
from typing import Optional, Tuple

import tensorflow as tf
import tqdm

from prepare_tfrecords import read_tfrecord
import transforms


def build_dataset(
        tfrec_root: str,
        image_size: Tuple[int, int],
        is_train: bool,
        batch_size: Optional[int] = None,
        cache: bool = False,
        drop_filename: bool = True,
        ):
    """ image_size is height, width tuple.
    """
    options_no_order = tf.data.Options()
    options_no_order.experimental_deterministic = False
    AUTO = tf.data.experimental.AUTOTUNE
    pattern = '/train-*.tfrec' if is_train else '/valid.tfrec'
    dataset = tf.data.TFRecordDataset(
        tf.io.gfile.glob(tfrec_root.rstrip('/') + pattern),
        num_parallel_reads=AUTO)
    dataset = dataset.with_options(options_no_order)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    dataset = dataset.map(
        partial(transforms.resize_and_crop_image, target_size=image_size),
        num_parallel_calls=AUTO)
    if drop_filename:
        dataset = dataset.map(transforms.drop_filename, num_parallel_calls=AUTO)
    if is_train:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(2048)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    if cache:
        dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('tfrec_root')
    arg('--image-size',
        type=lambda x: tuple(map(int, x.split('x'))),
        default=(224, 224))
    args = parser.parse_args()

    dataset = build_dataset(
        args.tfrec_root,
        is_train=True,
        image_size=args.image_size,
        drop_filename=False,
    )
    for image, class_num, filename in tqdm.tqdm(dataset):
        assert image.numpy().shape == args.image_size + (3,)


if __name__ == '__main__':
    main()
