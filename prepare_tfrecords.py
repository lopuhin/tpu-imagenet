#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict

import tensorflow as tf
import tqdm


def read_jpeg_and_label(filename):
    image_data = tf.io.read_file(filename)
    label = tf.strings.split(
        tf.expand_dims(filename, axis=-1), sep='/').values[-2]
    return image_data, label


def to_tfrecord(image_data: bytes, class_num: int) -> tf.train.Example:
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
        'class': tf.train.Feature(int64_list=tf.train.Int64List(value=[class_num])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def read_tfrecord(example):
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'class': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, features)
    image = tf.image.decode_jpeg(example['image'])
    class_num = example['class']
    return image, class_num


def get_class_map(train_root: Path) -> Dict[str, int]:
    return {path.name: idx
            for idx, path in enumerate(sorted(train_root.iterdir()))
            if path.is_dir()}


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('src', type=Path, 
        help='path to the root folder with "train" and "val" subfolders')
    arg('dst', type=Path, help='target for tfrecord files')
    arg('--n-shards', type=int, required=True)
    # arg('--max-size', type=int, default=320)
    args = parser.parse_args()

    max_shards = 1000
    if args.n_shards > max_shards:
        parser.error(f'--n-shards should be smaller than {max_shards}')

    train_root = args.src / 'train'
    class_map = get_class_map(train_root)
    args.dst.mkdir(exist_ok=True, parents=True)
    shard_writers = [
        tf.io.TFRecordWriter(str(args.dst / f'train-{shard:03d}.tfrec'))
        for shard in range(args.n_shards)]
    try:
        dataset = tf.data.Dataset.list_files(
            str(train_root / '*/*.JPEG'), shuffle=True, seed=42)
        dataset = dataset.map(
            read_jpeg_and_label,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        for i, (image_data, label) in enumerate(tqdm.tqdm(dataset)):
            shard = i % args.n_shards
            class_num = class_map[label.numpy().decode('utf8')]
            example = to_tfrecord(image_data.numpy(), class_num)
            shard_writers[shard].write(example.SerializeToString())

    finally:
        for w in shard_writers:
            w.close()
    

if __name__ == '__main__':
    main()
