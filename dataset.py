import argparse
from functools import partial
from pathlib import Path
from typing import Tuple

import tensorflow as tf
import tqdm

from prepare_tfrecords import read_tfrecord


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


def resize_and_crop_image(image, label, filename, target_size: Tuple[int, int]):
    """ Resize and crop using "fill" algorithm: make sure the resulting image
    is cut out from the source image so that it fills the target_size
    entirely with no black bars and a preserved aspect ratio.
    """
    h, w = _image_hw(image)
    th, tw = target_size
    image = tf.cond(
        (w * th) / (h * tw) < 1,
        lambda: tf.image.resize(image, [h * tw/w, w * tw/w]),
        lambda: tf.image.resize(image, [h * th/h, w * th/h])
    )
    nh, nw = _image_hw(image)
    image = tf.image.crop_to_bounding_box(
        image, (nh - th) // 2, (nw - tw) // 2, th, tw)
    return image, label, filename


def _image_hw(image):
    shape = tf.shape(image)
    return shape[0], shape[1]


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
