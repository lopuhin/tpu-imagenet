#!/usr/bin/env python3
import argparse
from typing import Tuple

import tensorflow as tf

from dataset import build_dataset


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('tfrec_roots',
        help='path(s) to folder with tfrecords '
             '(can be one or multiple gcs bucket paths as well)',
        nargs='+')
    arg('--image-size', type=int, default=224)

    arg('--batch-size', type=int, default=512, help='per device')
    arg('--lr', type=float, default=1.6)
    arg('--lr-decay', type=float, default=0.9)
    arg('--epochs', type=int, default=110)
    arg('--lr-sustain-epochs', type=int, default=20)
    arg('--lr-warmup-epochs', type=int, default=5)

    # TODO move to dataset.json created during pre-processing
    arg('--n-classes', type=int, default=1000)
    arg('--n-train-samples', type=int, default=1281167)

    arg('--xla', type=int, default=0, help='enable XLA')
    arg('--mixed', type=int, default=1, help='enable mixed precision')
    args = parser.parse_args()

    strategy, tpu = get_strategy()
    setup_policy(xla_accelerate=args.xla, mixed_precision=args.mixed, tpu=tpu)

    image_size = (args.image_size, args.image_size)
    batch_size = args.batch_size * strategy.num_replicas_in_sync
    dtype = tf.float32
    if args.mixed:
        dtype = tf.bfloat16 if tpu else tf.float16
    train_dataset, valid_dataset = [build_dataset(
        args.tfrec_roots,
        is_train=is_train,
        image_size=image_size,
        cache=not is_train,
        batch_size=batch_size,
        drop_filename=True,
        dtype=dtype,
        ) for is_train in [True, False]]

    model = build_model(
        strategy, image_size=image_size, n_classes=args.n_classes)
    lr_schedule = build_lr_schedule(
        lr_max=args.lr,
        lr_warmup_epochs=args.lr_warmup_epochs,
        lr_sustain_epochs=args.lr_sustain_epochs,
        lr_decay=args.lr_decay,
    )
    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        lr_schedule, verbose=True)

    # TODO "imagenet" preprocessing
    # TODO L2 weight decay
    model.fit(
        train_dataset,
        steps_per_epoch=args.n_train_samples // batch_size,
        epochs=args.epochs,
        callbacks=[lr_callback],
        validation_freq=4,
        validation_data=valid_dataset,
    )


def build_model(strategy, image_size: Tuple[int, int], n_classes: int):
    with strategy.scope():
        base = tf.keras.applications.ResNet50(
            input_shape=image_size + (3,),
            weights=None,
            include_top=False,
        )
        base.trainable = True
        model = tf.keras.Sequential([
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(
                n_classes, activation='softmax', dtype='float32'),
        ])
    model.compile(
        optimizer=tf.keras.optimizers.SGD(momentum=0.9),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    return model


def build_lr_schedule(
        lr_max: float,
        lr_warmup_epochs: int,
        lr_sustain_epochs: int,
        lr_decay: float,
    ):
    def get_lr(epoch: int):
        lr_min = lr_start = lr_max / 100
        if epoch < lr_warmup_epochs:
            lr = (lr_max - lr_start) / lr_warmup_epochs * epoch + lr_start
        elif epoch < lr_warmup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = ((lr_max - lr_min) *
                   lr_decay ** (epoch - lr_warmup_epochs - lr_sustain_epochs) +
                   lr_min)
        return lr
    return get_lr


def get_strategy():
    try:
        # No parameters necessary if TPU_NAME environment variable is set.
        # On Kaggle this is always the case.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy() # works on CPU and single GPU

    print(f'num replicas: {strategy.num_replicas_in_sync}, strategy {strategy}')
    return strategy, tpu


def setup_policy(mixed_precision: bool, xla_accelerate: bool, tpu: bool):
    if mixed_precision:
        policy = tf.keras.mixed_precision.experimental.Policy(
            'mixed_bfloat16' if tpu else 'mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)
        print(f'Mixed precision enabled: {policy}')

    if xla_accelerate:
        tf.config.optimizer.set_jit(True)
        print('XLA enabled')


if __name__ == '__main__':
    main()
