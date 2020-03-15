from typing import Tuple

import tensorflow as tf


def resize_image_if_larger(image, max_size: int):
    """ Resize image to given maximal size, preserving aspect ratio,
    also casts it to float32.
    """
    h, w = _image_hw(image)
    image = tf.cast(image, tf.float32)
    image = tf.cond(
        w > max_size or h > max_size,
        lambda: tf.cond(
            w > h,
            lambda: tf.image.resize(image, [h * max_size/w, w * max_size/w]),
            lambda: tf.image.resize(image, [h * max_size/h, w * max_size/h])
        ),
        lambda: image)
    return image


def resize_and_crop_image(image, target_size: Tuple[int, int]):
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
    # explicit size needed for TPU
    image = tf.reshape(image, [*target_size, 3])
    return image


def normalize(image, dtype):
    image = tf.cast(image, dtype) / 255.0
    return image


def _image_hw(image):
    shape = tf.shape(image)
    return shape[0], shape[1]
