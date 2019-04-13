# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for Residual Networks.
Residual networks ('v1' ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
The full preactivation 'v2' ResNet variant was introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm(inputs, training, data_format):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


################################################################################
# ResNet block definitions.
################################################################################
def _building_block_v1(inputs, filters, training, projection_shortcut, strides,
                       data_format):
  """A single block for ResNet v1, without a bottleneck.
  Convolution then batch normalization then ReLU as described by:
    Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    shortcut = batch_norm(inputs=shortcut, training=training,
                          data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs


def _building_block_v2(inputs, filters, training, projection_shortcut, strides,
                       data_format):
  """A single block for ResNet v2, without a bottleneck.
  Batch normalization then ReLu then convolution as described by:
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/pdf/1603.05027.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)

  return inputs + shortcut


def _bottleneck_block_v1(inputs, filters, training, projection_shortcut,
                         strides, data_format):
  """A single block for ResNet v1, with a bottleneck.
  Similar to _building_block_v1(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    shortcut = batch_norm(inputs=shortcut, training=training,
                          data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs


def _bottleneck_block_v2(inputs, filters, training, projection_shortcut,
                         strides, data_format):
  """A single block for ResNet v2, with a bottleneck.
  Similar to _building_block_v2(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
  Adapted to the ordering conventions of:
    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)

  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)

  return inputs + shortcut


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format):
  """Creates one layer of blocks for the ResNet model.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block layer.
  """

  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = filters * 4 if bottleneck else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                    data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, training, None, 1, data_format)

  return tf.identity(inputs, name)

def attention_module(inputs, training, data_format):
  """Creates one self-attention module.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the self-attention module.
  """
  channel_axes = 1 if data_format == 'channels_first' else 3
  h_w_axes = [2, 3] if data_format == 'channels_first' else [1, 2]
  inputs_shape = tf.shape(inputs)
  h_times_w = inputs_shape[h_w_axes[0]] * inputs_shape[h_w_axes[1]]

  attention_map = tf.reduce_mean(inputs, channel_axes, keepdims=True)
  attention_map = tf.nn.l2_normalize(attention_map, h_w_axes) * tf.sqrt(tf.cast(h_times_w, tf.float32))
  inputs = inputs * attention_map

  return tf.identity(inputs, 'attention')

def regression_points(top_feature, lower_feature, point_num, training, data_format):
  """Creates one 2-step location regression module.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    lower_feature: Same as inputs, but from lower layer.
    point_num: Number of points to predict.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the 2-step location regression module.
  """
  # channel_axes = 1 if data_format == 'channels_first' else 3
  h_w_axes = [2, 3] if data_format == 'channels_first' else [1, 2]
  with tf.variable_scope('first_loc'):
    # The 1st step.
    # The method by learning
    inputs=tf.identity(top_feature)
    inputs=conv2d_fixed_padding(
          inputs=inputs, filters=128, 
          kernel_size=3, strides=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = tf.reduce_mean(inputs, h_w_axes, keepdims=True)
    inputs = tf.identity(inputs, 'loc_reduce_mean')
    # Output denotes [lefty, leftx, righty, rightx].
    # Crop the whole image to 7*7 grid with overlaps,
    # so these 4 numbers are in range [0, 7).
    inputs = tf.squeeze(inputs, h_w_axes)
    inputs = tf.layers.dense(inputs=inputs, units=2*point_num)
    loc_dence = tf.identity(inputs, 'loc_dense')

  with tf.variable_scope('second_loc'):
    # The 2nd step. Slice the lower feature
    # Here we get from the conv_n
    ## confirm the crop size
    with tf.variable_scope('prepare'):
      feature_map = tf.stop_gradient(lower_feature)
      # feature_map = lower_feature
      feature_map_shape = tf.shape(feature_map)
      feature_map_size = [feature_map_shape[h_w_axes[0]], feature_map_shape[h_w_axes[1]]]
      crop_size = [tf.to_int32(x*2/7) for x in feature_map_size]  # Crop the 2/7 of feature map.
      crop_centers = tf.split(tf.stop_gradient(loc_dence), point_num, axis=1) # point_num*[b*2]
      # crop_centers = tf.split(loc_dence, point_num, axis=1) # point_num*[b*2]
    ## crop and regression
    if data_format == 'channels_first':
      # Convert the feature_map from channels_first (NCHW) back to channels_last (NHWC) 
      # for tf.image.crop_and_resize.
      feature_map = tf.transpose(feature_map, [0, 2, 3, 1])
    location_list = []
    for center in crop_centers:
      begin = center - 2./7. * 0.5
      end = center + 2./7. * 0.5
      boxes = tf.concat([begin, end], axis=1)
      inputs = tf.image.crop_and_resize(
        feature_map,
        boxes=boxes,
        box_ind=tf.range(feature_map_shape[0]),
        crop_size = crop_size,
        method='bilinear',
        extrapolation_value=0,
        name=None)
      # if data_format == 'channels_first':
      #   # Convert the feature_map from to channels_first (NCHW)
      #   inputs = tf.transpose(inputs, [0, 3, 1, 2])

      with tf.variable_scope('post_crop', reuse=tf.AUTO_REUSE):
        ## get the location
        inputs=conv2d_fixed_padding(inputs, filters=64,
            kernel_size=3, strides=1, data_format='channels_last')
        inputs = batch_norm(inputs, training, 'channels_last')
        inputs = tf.nn.relu(inputs)
        inputs0 = tf.identity(inputs)
        # /2
        inputs=conv2d_fixed_padding(inputs, filters=64,
            kernel_size=3, strides=2, data_format='channels_last')
        inputs = batch_norm(inputs, training, 'channels_last')
        inputs = tf.nn.relu(inputs)
        inputs1 = tf.identity(inputs)
        # /2
        inputs=conv2d_fixed_padding(inputs, filters=64,
            kernel_size=3, strides=2, data_format='channels_last')
        inputs = batch_norm(inputs, training, 'channels_last')
        inputs = tf.nn.relu(inputs)
        inputs2 = tf.identity(inputs)
        # /2
        inputs=conv2d_fixed_padding(inputs, filters=64,
            kernel_size=3, strides=2, data_format='channels_last')
        inputs = batch_norm(inputs, training, 'channels_last')
        inputs = tf.nn.relu(inputs)
        # ==
        inputs=conv2d_fixed_padding(inputs, filters=64,
            kernel_size=3, strides=1, data_format='channels_last')
        inputs = batch_norm(inputs, training, 'channels_last')
        inputs = tf.nn.relu(inputs)
        # *2
        inputs2=conv2d_fixed_padding(inputs2, filters=64,
            kernel_size=1, strides=1, data_format='channels_last')
        inputs2 = batch_norm(inputs2, training, 'channels_last')
        inputs2 = tf.nn.relu(inputs2)
        shape = [tf.shape(inputs2)[1], tf.shape(inputs2)[2]]
        inputs = tf.image.resize_nearest_neighbor(inputs, shape)
        inputs = inputs + inputs2
        # *2
        inputs1=conv2d_fixed_padding(inputs1, filters=64,
            kernel_size=1, strides=1, data_format='channels_last')
        inputs1 = batch_norm(inputs1, training, 'channels_last')
        inputs1 = tf.nn.relu(inputs1)
        shape = [tf.shape(inputs1)[1], tf.shape(inputs1)[2]]
        inputs = tf.image.resize_nearest_neighbor(inputs, shape)
        inputs = inputs + inputs1
        # *2
        inputs0=conv2d_fixed_padding(inputs0, filters=64,
            kernel_size=1, strides=1, data_format='channels_last')
        inputs0 = batch_norm(inputs0, training, 'channels_last')
        inputs0 = tf.nn.relu(inputs0)
        shape = [tf.shape(inputs0)[1], tf.shape(inputs0)[2]]
        inputs = tf.image.resize_nearest_neighbor(inputs, shape)
        inputs = inputs + inputs0

        # full conv
        inputs=conv2d_fixed_padding(inputs, filters=64,
            kernel_size=3, strides=1, data_format='channels_last')
        inputs = batch_norm(inputs, training, 'channels_last')
        inputs = tf.nn.relu(inputs)
        inputs=conv2d_fixed_padding(inputs, filters=9,
            kernel_size=1, strides=1, data_format='channels_last')
        inputs = batch_norm(inputs, training, 'channels_last')
        inputs = tf.nn.relu(inputs)
        # we need b*4*c format, so here we trans to b*h*w*c, for h*w will be reduce to 1.
      location_list.append(tf.identity(inputs))
    location_dence = location_list 
    # location_dence = tf.concat(location_list, axis=1) 

  return loc_dence, location_dence

################################################################################
# ResNet model construction.
################################################################################
class Model(object):
  """Base class for building the Resnet Model."""

  def __init__(self, is_attention, location_feature_stage,
              resnet_size, bottleneck, num_classes, num_filters,
               kernel_size, conv_stride, first_pool_size, first_pool_stride,
               block_sizes, block_strides,
               resnet_version=DEFAULT_VERSION, data_format=None,
               dtype=DEFAULT_DTYPE):
    """Creates a model for classifying an image.
    Args:
      resnet_size: A single integer for the size of the ResNet model.
      bottleneck: Use regular blocks or bottleneck blocks.
      num_classes: The number of classes used as labels.
      num_filters: The number of filters to use for the first block layer
        of the model. This number is then doubled for each subsequent block
        layer.
      kernel_size: The kernel size to use for the initial convolution.
      conv_stride: stride size for the initial convolutional layer
      first_pool_size: Pool size to be used for the first pooling layer.
        If none, the first pooling layer is skipped.
      first_pool_stride: stride size for the first pooling layer. Not used
        if first_pool_size is None.
      block_sizes: A list containing n values, where n is the number of sets of
        block layers desired. Each value should be the number of blocks in the
        i-th set.
      block_strides: List of integers representing the desired stride size for
        each of the sets of block layers. Should be same length as block_sizes.
      resnet_version: Integer representing which version of the ResNet network
        to use. See README for details. Valid values: [1, 2]
      data_format: Input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
      dtype: The TensorFlow dtype to use for calculations. If not specified
        tf.float32 is used.
    Raises:
      ValueError: if invalid version is selected.
    """
    self.is_attention = is_attention
    self.is_regression = True if location_feature_stage is not None else False
    self.location_feature_stage = location_feature_stage

    self.resnet_size = resnet_size

    if not data_format:
      data_format = (
          'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    self.resnet_version = resnet_version
    if resnet_version not in (1, 2):
      raise ValueError(
          'Resnet version should be 1 or 2. See README for citations.')

    self.bottleneck = bottleneck
    if bottleneck:
      if resnet_version == 1:
        self.block_fn = _bottleneck_block_v1
      else:
        self.block_fn = _bottleneck_block_v2
    else:
      if resnet_version == 1:
        self.block_fn = _building_block_v1
      else:
        self.block_fn = _building_block_v2

    if dtype not in ALLOWED_TYPES:
      raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

    self.data_format = data_format
    self.num_classes = num_classes
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.conv_stride = conv_stride
    self.first_pool_size = first_pool_size
    self.first_pool_stride = first_pool_stride
    self.block_sizes = block_sizes
    self.block_strides = block_strides
    self.dtype = dtype
    self.pre_activation = resnet_version == 2


  def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE,
                           *args, **kwargs):
    """Creates variables in fp32, then casts to fp16 if necessary.
    This function is a custom getter. A custom getter is a function with the
    same signature as tf.get_variable, except it has an additional getter
    parameter. Custom getters can be passed as the `custom_getter` parameter of
    tf.variable_scope. Then, tf.get_variable will call the custom getter,
    instead of directly getting a variable itself. This can be used to change
    the types of variables that are retrieved with tf.get_variable.
    The `getter` parameter is the underlying variable getter, that would have
    been called if no custom getter was used. Custom getters typically get a
    variable with `getter`, then modify it in some way.
    This custom getter will create an fp32 variable. If a low precision
    (e.g. float16) variable was requested it will then cast the variable to the
    requested dtype. The reason we do not directly create variables in low
    precision dtypes is that applying small gradients to such variables may
    cause the variable not to change.
    Args:
      getter: The underlying variable getter, that has the same signature as
        tf.get_variable and returns a variable.
      name: The name of the variable to get.
      shape: The shape of the variable to get.
      dtype: The dtype of the variable to get. Note that if this is a low
        precision dtype, the variable will be created as a tf.float32 variable,
        then cast to the appropriate dtype
      *args: Additional arguments to pass unmodified to getter.
      **kwargs: Additional keyword arguments to pass unmodified to getter.
    Returns:
      A variable which is cast to fp16 if necessary.
    """

    if dtype in CASTABLE_TYPES:
      var = getter(name, shape, tf.float32, *args, **kwargs)
      return tf.cast(var, dtype=dtype, name=name + '_cast')
    else:
      return getter(name, shape, dtype, *args, **kwargs)

  def _model_variable_scope(self):
    """Returns a variable scope that the model should be created under.
    If self.dtype is a castable type, model variable will be created in fp32
    then cast to self.dtype before being used.
    Returns:
      A variable scope for the model.
    """

    return tf.variable_scope('resnet_model',
                             custom_getter=self._custom_dtype_getter)

  def __call__(self, inputs, training):
    """Add operations to classify a batch of input images.
    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.
    Returns:
      A logits Tensor with shape [<batch_size>, self.num_classes].
    """

    with self._model_variable_scope():
      # Tanspose has been done in preprocessing, so here we don't need to do so.
      # if self.data_format == 'channels_first':
      #   # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      #   # This provides a large performance boost on GPU. See
      #   # https://www.tensorflow.org/performance/performance_guide#data_formats
      #   inputs = tf.transpose(inputs, [0, 3, 1, 2])
      inputs = conv2d_fixed_padding(
          inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
          strides=self.conv_stride, data_format=self.data_format)
      # The first useful feature map.
      self.blocks_feature = [tf.identity(inputs, 'initial_conv')]

      # We do not include batch normalization or activation functions in V2
      # for the initial conv1 because the first ResNet unit will perform these
      # for both the shortcut and non-shortcut paths as part of the first
      # block's projection. Cf. Appendix of [2].
      if self.resnet_version == 1:
        inputs = batch_norm(inputs, training, self.data_format)
        inputs = tf.nn.relu(inputs)
        # The first useful feature map should be activated if could.
        self.blocks_feature = [tf.identity(inputs, 'initial_conv')]

      if self.first_pool_size:
        with tf.variable_scope('first_pool'):
          inputs = tf.layers.max_pooling2d(
              inputs=inputs, pool_size=self.first_pool_size,
              strides=self.first_pool_stride, padding='SAME',
              data_format=self.data_format)
          inputs = tf.identity(inputs, 'initial_max_pool')

      for i, num_blocks in enumerate(self.block_sizes):
        name = 'block_layer{}'.format(i + 1)
        with tf.variable_scope(name):
          num_filters = self.num_filters * (2**i)
          inputs = block_layer(
              inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
              block_fn=self.block_fn, blocks=num_blocks,
              strides=self.block_strides[i], training=training,
              name=name, data_format=self.data_format)
          self.blocks_feature.append(tf.identity(inputs, name))

      # Only apply the BN and ReLU for model that does pre_activation in each
      # building/bottleneck block, eg resnet V2.
      if self.pre_activation:
        inputs = batch_norm(inputs, training, self.data_format)
        inputs = tf.nn.relu(inputs)
        # The last feature map must be applied activation
        self.blocks_feature[-1]=tf.identity(inputs)

    # Attention module.
    if self.is_attention:
      with tf.variable_scope('attention'):
        inputs = attention_module(inputs, training, data_format=self.data_format)
        self.attentioned_feature = tf.identity(inputs,'attention')
        # The last feature map must be applied attention if possible.
        self.blocks_feature[-1]=self.attentioned_feature

    with tf.variable_scope('CLS'):
      # The current top layer has shape
      # `batch_size x pool_size x pool_size x final_size`.
      # ResNet does an Average Pooling layer over pool_size,
      # but that is the same as doing a reduce_mean. We do a reduce_mean
      # here because it performs better than AveragePooling2D.
      axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
      inputs = tf.reduce_mean(inputs, axes, keepdims=True)
      inputs = tf.identity(inputs, 'final_reduce_mean')

      inputs = tf.squeeze(inputs, axes)
      inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
      final_cls_dense = tf.identity(inputs, 'final_dense')

    # regression the loc
    if self.is_attention and self.is_regression:
      with tf.variable_scope('REG'):
        lower_feature = self.blocks_feature[self.location_feature_stage]
        loc, location = regression_points(self.attentioned_feature, lower_feature, 2, training, self.data_format)
      return final_cls_dense, loc, location
    # If don't regress.
    return final_cls_dense


def _get_block_sizes(resnet_size):
  """Retrieve the size of each block_layer in the ResNet model.
  The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.
  Args:
    resnet_size: The number of convolutional layers needed in the model.
  Returns:
    A list of block sizes to use in building the model.
  Raises:
    KeyError: if invalid resnet_size is received.
  """
  choices = {
      2: [],
      4: [1],
      6: [2],
      10: [2, 2],
      14: [2, 2, 2],
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)

def CLS_REG_Model(resnet_size, resnet_version, 
                  is_attention, location_feature_stage,
                  data_format):
  return Model(
          # classification configs
          is_attention=is_attention,
          # regression configs
          location_feature_stage=location_feature_stage,
          # resnet configs
          resnet_size=resnet_size,
          bottleneck=False,
          num_classes=2,
          num_filters=64,   # first block
          kernel_size=7,    # init conv
          conv_stride=2,    # init conv
          first_pool_size=3 if resnet_size!=2 else None,
          first_pool_stride=2 if resnet_size!=2 else None,
          block_sizes=_get_block_sizes(resnet_size=resnet_size),
          block_strides=[1, 2, 2, 2],   # stage 
          resnet_version=resnet_version,
          data_format=data_format,
          dtype=tf.float32)

if __name__ == '__main__':
  import numpy as np 

  image_input = tf.placeholder(tf.uint8, shape=(1, 224, 224, 3 ))
  image = tf.to_float(image_input)
  print(image)
  with tf.variable_scope('test', default_name=None, values=[image], reuse=tf.AUTO_REUSE):
    model = CLS_REG_Model(18, 2, True, 2, 'channels_last')
    out = model(image,True)

  config = tf.ConfigProto(device_count={'gpu':0})
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(out, feed_dict={image_input: np.zeros((1, 224, 224, 3))}))
    writer = tf.summary.FileWriter("./test/" ,sess.graph)