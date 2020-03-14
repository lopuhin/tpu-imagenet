from typing import Tuple

import tensorflow as tf


def resize_image(image, label, filename, max_size: int):
    """ Resize image to given maximal size, preserving aspect ratio.
    """
    h, w = _image_hw(image)
    image = tf.cond(
        w > h,
        lambda: tf.image.resize(image, [h * max_size/w, w * max_size/w]),
        lambda: tf.image.resize(image, [h * max_size/h, w * max_size/h])
    )
    return image, label, filename


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
