import argparse
from pathlib import Path

import tensorflow as tf
import tqdm

from prepare_tfrecords import read_tfrecord


def build_dataset(tfrec_root: Path):
    options_no_order = tf.data.Options()
    options_no_order.experimental_deterministic = False
    AUTO = tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(
        [str(p) for p in tfrec_root.glob('*.tfrec')],
        num_parallel_reads=AUTO)
    dataset = dataset.with_options(options_no_order)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    dataset = dataset.shuffle(2048)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tfrec_root', type=Path)
    args = parser.parse_args()

    dataset = build_dataset(args.tfrec_root)
    for _ in tqdm.tqdm(dataset):
        pass


if __name__ == '__main__':
    main()
