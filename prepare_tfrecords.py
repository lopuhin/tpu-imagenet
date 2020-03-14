#!/usr/bin/env python3
import argparse
from functools import partial
from pathlib import Path
from typing import Dict

import tensorflow as tf
import tqdm

from transforms import resize_image


def read_jpeg_and_label(filename):
    image_data = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_data)
    image = tf.cond(
        tf.shape(image)[2] == 1,
        lambda: tf.image.grayscale_to_rgb(image),
        lambda: image[:, :, :3])
    label = tf.strings.split(
        tf.expand_dims(filename, axis=-1), sep='/').values[-2]
    return image, label, filename


def compress_image(image, label, filename):
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_jpeg(image, quality=90, optimize_size=True)
    return image, label, filename


def to_tfrecord(image_data: bytes, class_num: int, filename: bytes) -> tf.train.Example:
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
        'class': tf.train.Feature(int64_list=tf.train.Int64List(value=[class_num])),
        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def read_tfrecord(example):
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'class': tf.io.FixedLenFeature([], tf.int64),
        'filename': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, features)
    image = tf.image.decode_jpeg(example['image'])
    class_num = example['class']
    filename = example['filename']
    return image, class_num, filename


def get_class_map(train_root: Path) -> Dict[str, int]:
    return {path.name: idx
            for idx, path in enumerate(sorted(train_root.iterdir()))
            if path.is_dir()}


def prepare_dataset(root: Path, max_size: int):
    dataset = tf.data.Dataset.list_files(
        str(root / '*/*.JPEG'), shuffle=True, seed=42)
    AUTO = tf.data.experimental.AUTOTUNE
    dataset = dataset.map(read_jpeg_and_label, num_parallel_calls=AUTO)
    dataset = dataset.map(partial(resize_image, max_size=max_size),
                          num_parallel_calls=AUTO)
    dataset = dataset.map(compress_image, num_parallel_calls=AUTO)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('src', type=Path, 
        help='path to the root folder with "train" and "val" subfolders')
    arg('dst', type=Path, help='target for tfrecord files')
    arg('--n-shards', type=int, required=True)
    arg('--max-size', type=int, default=500)
    args = parser.parse_args()

    max_shards = 1000
    if args.n_shards > max_shards:
        parser.error(f'--n-shards should be smaller than {max_shards}')

    train_root = args.src / 'train'
    class_map = get_class_map(train_root)
    args.dst.mkdir(exist_ok=True, parents=True)
    train_writers = [
        tf.io.TFRecordWriter(str(args.dst / f'train-{shard:03d}.tfrec'))
        for shard in range(args.n_shards)]
    valid_writers = [tf.io.TFRecordWriter(str(args.dst / 'val.tfrec'))]
    valid_root = args.src / 'val'

    for root, writers in [(valid_root, valid_writers),
                          (train_root, train_writers)]:
        try:
            dataset = prepare_dataset(root, max_size=args.max_size)
            for i, (image_data, label, filename) in enumerate(tqdm.tqdm(dataset)):
                class_num = class_map[label.numpy().decode('utf8')]
                example = to_tfrecord(
                    image_data.numpy(), class_num, filename.numpy())
                writer = writers[i % len(writers)]
                writer.write(example.SerializeToString())

        finally:
            for w in writers:
                w.close()
    

if __name__ == '__main__':
    main()
