import argparse
from typing import Optional, Tuple, List

import tensorflow as tf
import tqdm

from prepare_tfrecords import read_tfrecord
import transforms


def build_dataset(
        tfrec_roots: List[str],
        image_size: Tuple[int, int],
        is_train: bool,
        dtype=tf.float32,
        batch_size: Optional[int] = None,
        cache: bool = False,
        drop_filename: bool = True,
        ):
    """ image_size is height, width tuple.
    """
    AUTO = tf.data.experimental.AUTOTUNE
    pattern = '/train-*.tfrec' if is_train else '/val.tfrec'
    tfrec_paths = []
    for tfrec_root in tfrec_roots:
        tfrec_paths.extend(tf.io.gfile.glob(tfrec_root.rstrip('/') + pattern))
    print('tfrec paths', tfrec_paths)
    dataset = tf.data.TFRecordDataset(tfrec_paths, num_parallel_reads=AUTO)
    options_no_order = tf.data.Options()
    options_no_order.experimental_deterministic = False
    dataset = dataset.with_options(options_no_order)

    def process(filename):
        image, label, filename = read_tfrecord(filename)
        image = transforms.resize_and_crop_image(
            image, target_size=image_size)
        image = transforms.normalize(image, dtype=dtype)
        result = (image, label)
        if not drop_filename:
            result += (filename,)
        return result

    dataset = dataset.map(process, num_parallel_calls=AUTO)
    if cache:
        dataset = dataset.cache()
    if is_train:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(4096)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
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
        [args.tfrec_root],
        is_train=True,
        image_size=args.image_size,
        drop_filename=False,
    )
    for image, class_num, filename in tqdm.tqdm(dataset):
        assert image.numpy().shape == args.image_size + (3,)


if __name__ == '__main__':
    main()
