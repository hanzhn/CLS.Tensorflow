# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess images.

The preprocessing steps for VGG were introduced in the following technical
report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
"""
# ==============================================================================
# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

def _ImageDimensions(image, rank = 3):
  """Returns the dimensions of an image tensor.

  Args:
    image: A rank-D Tensor. For 3-D  of shape: `[height, width, channels]`.
    rank: The expected rank of the image

  Returns:
    A list of corresponding to the dimensions of the
    input image.  Dimensions that are statically known are python integers,
    otherwise they are integer scalar tensors.
  """
  if image.get_shape().is_fully_defined():
    return image.get_shape().as_list()
  else:
    static_shape = image.get_shape().with_rank(rank).as_list()
    dynamic_shape = tf.unstack(tf.shape(image), rank)
    return [s if s is not None else d
            for s, d in zip(static_shape, dynamic_shape)]

def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)

def unwhiten_image(image, output_rgb=True):
  means=[_R_MEAN, _G_MEAN, _B_MEAN]
  if not output_rgb:
    image_channels = tf.unstack(image, axis=-1, name='split_bgr')
    image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=-1, name='merge_rgb')
  num_channels = image.get_shape().as_list()[-1]
  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] += means[i]
  return tf.concat(axis=2, values=channels)

def random_flip_left_right(image):
  with tf.name_scope('random_flip_left_right'):
    float_width = tf.to_float(_ImageDimensions(image, rank=3)[1])

    uniform_random = tf.random_uniform([], 0, 1.0)
    mirror_cond = tf.less(uniform_random, .5)
    # Flip image.
    result = tf.cond(mirror_cond, lambda: tf.image.flip_left_right(image), lambda: image)
    return result

def preprocess_for_train(image, out_shape, data_format='channels_first', scope='ssd_preprocessing_train', output_rgb=True):
  """Preprocesses the given image for training.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    labels: A `Tensor` containing all labels for all bboxes of this image.
    bboxes: A `Tensor` containing all bboxes of this image, in range [0., 1.] with shape [num_bboxes, 4].
    out_shape: The height and width of the image after preprocessing.
    data_format: The data_format of the desired output image.
  Returns:
    A preprocessed image.
  """
  with tf.name_scope(scope, 'cls_preprocessing_train', [image]):
    if image.get_shape().ndims != 3:
      raise ValueError('Input must be of size [height, width, C>0]')
    # Convert to float scaled [0, 1].
    orig_dtype = image.dtype
    if orig_dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Randomly distort the colors. There are 4 ways to do it.
    distort_image = apply_with_random_selector(image,
                                          lambda x, ordering: distort_color(x, ordering, True),
                                          num_cases=4)

    # Randomly flip the image horizontally.
    random_sample_flip_image = random_flip_left_right(distort_image)
    # Rescale to VGG input scale.
    height, width, _ = _ImageDimensions(random_sample_flip_image, rank=3)
    float_height, float_width = tf.to_float(height), tf.to_float(width)
    target_height, target_width = tf.to_float(out_shape[0]), tf.to_float(out_shape[1])

    random_sample_flip_resized_image = tf.image.resize_images(random_sample_flip_image, out_shape, method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    random_sample_flip_resized_image.set_shape([None, None, 3])

    final_image = tf.to_float(tf.image.convert_image_dtype(random_sample_flip_resized_image, orig_dtype, saturate=True))
    final_image = _mean_image_subtraction(final_image, [_R_MEAN, _G_MEAN, _B_MEAN])

    final_image.set_shape(out_shape + [3])
    if not output_rgb:
      image_channels = tf.unstack(final_image, axis=-1, name='split_rgb')
      final_image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=-1, name='merge_bgr')
    if data_format == 'channels_first':
      final_image = tf.transpose(final_image, perm=(2, 0, 1))
    return final_image

def preprocess_for_eval(image, out_shape, data_format='channels_first', scope='ssd_preprocessing_eval', output_rgb=True):
  """Preprocesses the given image for evaluation.
  TODO @HZ not support yet
  Args:
    image: A `Tensor` representing an image of arbitrary size.
    out_shape: The height and width of the image after preprocessing.
    data_format: The data_format of the desired output image.
  Returns:
    A preprocessed image.
  """
  print(image, out_shape)
  with tf.name_scope(scope, 'cls_preprocessing_eval', [image]):
    image = tf.to_float(image)
    if out_shape is not None:
      image = tf.image.resize_images(image, out_shape, method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
      image.set_shape(out_shape + [3])

    height, width, _ = _ImageDimensions(image, rank=3)
    output_shape = tf.stack([height, width])

    image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
    if not output_rgb:
      image_channels = tf.unstack(image, axis=-1, name='split_rgb')
      image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=-1, name='merge_bgr')
    # Image data format.
    if data_format == 'channels_first':
      image = tf.transpose(image, perm=(2, 0, 1))
    return image

def preprocess_image(image, out_shape, is_training=False, data_format='channels_first', output_rgb=True):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    labels: A `Tensor` containing all labels for all bboxes of this image.
    bboxes: A `Tensor` containing all bboxes of this image, in range [0., 1.] with shape [num_bboxes, 4].
    out_shape: The height and width of the image after preprocessing.
    is_training: Wether we are in training phase.
    data_format: The data_format of the desired output image.

  Returns:
    A preprocessed image.
  """
  if is_training:
    return preprocess_for_train(image, out_shape, data_format=data_format, output_rgb=output_rgb)
  else:
    return preprocess_for_eval(image, out_shape, data_format=data_format, output_rgb=output_rgb)
