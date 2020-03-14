#!/usr/bin/env python3
import argparse
from functools import partial
from typing import Tuple

import tensorflow as tf

from dataset import build_dataset


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('tfrec_path', help='path to folder with tfrecords')
    arg('--image-size', type=int, default=224)

    arg('--batch-size', type=int, default=32, help='per device')
    arg('--lr', type=float, default=1e-2)
    arg('--lr-decay', type=float, default=0.8)
    arg('--epochs', type=int, default=100)
    arg('--lr-sustain-epochs', type=int, default=20)
    arg('--lr-warmup-epochs', type=int, default=5)

    arg('--xla', action='store_true', help='enable XLA')
    arg('--mixed', action='store_true', help='enable mixed precision')
    args = parser.parse_args()

    strategy, tpu = get_strategy()
    setup_policy(xla_accelerate=args.xla, mixed_precision=args.mixed, tpu=tpu)

    image_size = (args.image_size, args.image_size)
    batch_size = args.batch_size * strategy.num_replicas_in_sync
    train_dataset, valid_dataset = [build_dataset(
        args.tfrec_path,
        is_train=is_train,
        image_size=image_size,
        cache=not is_train,
        batch_size=batch_size,
        drop_filename=True,
        ) for is_train in [True, False]]

    # TODO n_classes: store it somewhere
    model = build_model(strategy, image_size=image_size, n_classes=100)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        partial(
            get_lr,
            lr_warmup_epochs=args.lr_warmup_epochs,
            lr_sustain_epochs=args.lr_sustain_epochs,
            lr_decay=args.lr_decay,
        ), verbose=True)
    # TODO preprocessing!
    # TODO L2 weight decay
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        steps_per_epoch=10000 // batch_size,  # TODO get number of examples
        epochs=args.epochs,
        callbacks=[lr_callback],
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


def get_lr(
        epoch: int,
        lr_max: float,
        lr_warmup_epochs: int,
        lr_sustain_epochs: int,
        lr_decay: float,
        ):
    lr_min = lr_start = lr_max / 10
    if epoch < lr_warmup_epochs:
        lr = (lr_max - lr_start) / lr_warmup_epochs * epoch + lr_start
    elif epoch < lr_warmup_epochs + lr_sustain_epochs:
        lr = lr_max
    else:
        lr = ((lr_max - lr_min) *
               lr_decay ** (epoch - lr_warmup_epochs - lr_sustain_epochs) +
               lr_min)
    return lr


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