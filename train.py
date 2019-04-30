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

import os
import sys
import numpy as np

import tensorflow as tf
slim = tf.contrib.slim
import logging

from net import cls_reg_net_step2 as cls_net

from dataset import dataset_common_pupil as dataset_common
from preprocessing import cls_preprocessing
from utility import scaffolds

# hardware related configuration
tf.app.flags.DEFINE_integer(
    'num_readers', 10,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 24,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'num_cpu_threads', 0,
    'The number of cpu cores used to train.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1., 'GPU memory fraction to use.')
# scaffold related configuration
tf.app.flags.DEFINE_string(
    'data_dir', './dataset/tfrecords/pupil',
    'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'model_dir', './models/pupil',
    'The directory where the model will be stored.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are printed.')
tf.app.flags.DEFINE_integer(
    'save_summary_steps', 10,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_checkpoints_secs', 7200, # not used
    'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_checkpoints_steps', 5000,
    'The frequency with which the model is saved, in steps.')
# model related configuration
## resnet settings
tf.app.flags.DEFINE_integer(
    'resnet_version', 2,
    'The version of used resnet, default is v2.')
tf.app.flags.DEFINE_integer(
    'resnet_size', 18,
    'The layers of used resnet, [4,6,10,14,18,34,50,101,152,200].')
tf.app.flags.DEFINE_boolean(
    'attention_block', True,
    'Use attention or not. True/False')
tf.app.flags.DEFINE_boolean(
    'location_feature_stage', 0,
    'The largest feature map used to location the pupil. [0,1,2,3,4,5]or[-1,-2,-3] or None means not to reg')
## other settings
tf.app.flags.DEFINE_integer(
    'train_image_size', 64,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_integer(
    'train_epochs', None,
    'The number of epochs to use for training.')
tf.app.flags.DEFINE_integer(
    'batch_size', 32*2,
    'Batch size for training and evaluation.')
tf.app.flags.DEFINE_string(
    'data_format', 'channels_first', # 'channels_first' or 'channels_last'
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
# tf.app.flags.DEFINE_float(
#     'negative_ratio', 3., 'Negative ratio in the loss function.')
# tf.app.flags.DEFINE_float(
#     'match_threshold', 0.5, 'Matching threshold in the loss function.')
# tf.app.flags.DEFINE_float(
#     'neg_threshold', 0.5, 'Matching threshold for the negtive examples in the loss function.')
# optimizer related configuration
tf.app.flags.DEFINE_integer(
    'tf_random_seed', 20180503, 'Random seed for TensorFlow initializers.')
tf.app.flags.DEFINE_float(
    'weight_decay', 5e-5, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float(
    'learning_rate', 1e-3, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.000001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
# for learning rate piecewise_constant decay
tf.app.flags.DEFINE_string(
    'decay_boundaries', '15000, 20000, 25000, 30000, 35000',
    'Learning rate decay boundaries by global_step (comma-separated list).')
tf.app.flags.DEFINE_string(
    'lr_decay_factors', '1, 0.1, 0.01, 1, 0.1, 0.01',
    'The values of learning_rate decay factor for each segment between boundaries (comma-separated list).')
# tf.app.flags.DEFINE_string(
#     'decay_boundaries', ' 5000, 10000',
#     'Learning rate decay boundaries by global_step (comma-separated list).')
# tf.app.flags.DEFINE_string(
#     'lr_decay_factors', '1, 0.1, 0.01',
#     'The values of learning_rate decay factor for each segment between boundaries (comma-separated list).')
tf.app.flags.DEFINE_integer(
    'max_number_of_steps', 15000,
    'The max number of steps to use for training.')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', 'vgg_16',
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'model_scope', 'ssd300',
    'Model scope name used to replace the name_scope in checkpoint.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring from a checkpoint. ssd300/REG/second_loc')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', True,
    'When restoring a checkpoint would ignore missing variables.')
tf.app.flags.DEFINE_boolean(
    'multi_gpu', True,
    'Whether there is GPU to use for training.')
tf.app.flags.DEFINE_string(
    'loss_weights', '1, 2, 0',
    'training step'
)

FLAGS = tf.app.flags.FLAGS
#CUDA_VISIBLE_DEVICES
def validate_batch_size_for_multi_gpu(batch_size):
    """For multi-gpu, batch-size must be a multiple of the number of
    available GPUs.

    Note that this should eventually be handled by replicate_model_fn
    directly. Multi-GPU support is currently experimental, however,
    so doing the work here until that feature is in place.
    """
    if FLAGS.multi_gpu:
        from tensorflow.python.client import device_lib

        local_device_protos = device_lib.list_local_devices()
        num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
        if not num_gpus:
            raise ValueError('Multi-GPU mode was specified, but no GPUs '
                            'were found. To use CPU, run --multi_gpu=False.')

        remainder = batch_size % num_gpus
        if remainder:
            err = ('When running with multiple GPUs, batch size '
                    'must be a multiple of the number of available GPUs. '
                    'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
                    ).format(num_gpus, batch_size, batch_size - remainder)
            raise ValueError(err)
        return num_gpus
    return 0

def init_variables_from_checkpoint(checkpoint_exclude_scopes=None):
    """Variable initialization form a given checkpoint path.
    
    Modified from:
        https://github.com/tensorflow/models/blob/master/research/
        object_detection/model_lib.py
    
    Note that the init_fn is only run when initializing the model during the 
    very first global step.
    
    Args:
        checkpoint_exclude_scopes: Comma-separated list of scopes of variables
            to exclude when restoring from a checkpoint.
    """
    exclude_patterns = None
    if checkpoint_exclude_scopes:
        exclude_patterns = [scope.strip() for scope in 
                            checkpoint_exclude_scopes.split(',')]
    variables_to_restore = tf.global_variables()
    variables_to_restore.append(slim.get_or_create_global_step())
    variables_to_init = tf.contrib.framework.filter_variables(
        variables_to_restore, exclude_patterns=exclude_patterns)
    variables_to_init_dict = {var.op.name: var for var in variables_to_init}
    
    available_var_map = get_variables_available_in_checkpoint(
        variables_to_init_dict, FLAGS.checkpoint_path, 
        include_global_step=False)
    tf.train.init_from_checkpoint(FLAGS.checkpoint_path, available_var_map)
    
    
def get_variables_available_in_checkpoint(variables,
                                          checkpoint_path,
                                          include_global_step=True):
    """Returns the subset of variables in the checkpoint.
    
    Inspects given checkpoint and returns the subset of variables that are
    available in it.
    
    Args:
        variables: A dictionary of variables to find in checkpoint.
        checkpoint_path: Path to the checkpoint to restore variables from.
        include_global_step: Whether to include `global_step` variable, if it
            exists. Default True.
            
    Returns:
        A dictionary of variables.
        
    Raises:
        ValueError: If `variables` is not a dict.
    """
    if not isinstance(variables, dict):
        raise ValueError('`variables` is expected to be a dict.')
    
    # Available variables
    ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
    ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_shape_map()
    if not include_global_step:
        ckpt_vars_to_shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)
    vars_in_ckpt = {}
    for variable_name, variable in sorted(variables.items()):
        if variable_name in ckpt_vars_to_shape_map:
            if ckpt_vars_to_shape_map[variable_name] == variable.shape.as_list():
                vars_in_ckpt[variable_name] = variable
            else:
                logging.warning('Variable [%s] is avaible in checkpoint, but '
                                'has an incompatible shape with model '
                                'variable. Checkpoint shape: [%s], model '
                                'variable shape: [%s]. This variable will not '
                                'be initialized from the checkpoint.',
                                variable_name, 
                                ckpt_vars_to_shape_map[variable_name],
                                variable.shape.as_list())
        else:
            logging.warning('Variable [%s] is not available in checkpoint',
                            variable_name)
    return vars_in_ckpt

# def get_init_fn():
#     return scaffolds.get_init_fn_for_scaffold(FLAGS.model_dir, FLAGS.checkpoint_path,
#                                             FLAGS.model_scope, FLAGS.checkpoint_model_scope,
#                                             FLAGS.checkpoint_exclude_scopes, FLAGS.ignore_missing_vars,
#                                             name_remap={'/kernel': '/weights', '/bias': '/biases'})

def input_pipeline(dataset_pattern='train-*', is_training=True, batch_size=FLAGS.batch_size):
    def input_fn():
        target_shape = [FLAGS.train_image_size] * 2
        image_preprocessing_fn = lambda image_, label_: cls_preprocessing.preprocess_image(image_, label_, target_shape, 
                                                                    is_training=is_training, data_format=FLAGS.data_format, 
                                                                    output_rgb=False)
        _batch_size = batch_size
        image, _, points, is_reg = dataset_common.slim_get_batch(
                                                        FLAGS.num_classes,
                                                        _batch_size,
                                                        ('train' if is_training else 'val'),
                                                        os.path.join(FLAGS.data_dir, dataset_pattern),
                                                        FLAGS.num_readers,
                                                        FLAGS.num_preprocessing_threads,
                                                        image_preprocessing_fn,
                                                        num_epochs=FLAGS.train_epochs,
                                                        is_training=is_training)

        return image, {'loc_targets': points, 'is_reg': is_reg}
        # return image, {'cls_targets': cls_targets, 'loc_targets': tf.squeeze(points,axis=-1), 'is_reg': is_reg}
    return input_fn

def modified_smooth_l1(bbox_pred, bbox_targets, bbox_inside_weights=1., bbox_outside_weights=1., sigma=1.):
    """
        ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise
    """
    with tf.name_scope('smooth_l1', [bbox_pred, bbox_targets]):
        sigma2 = sigma * sigma

        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul

def ssd_model_fn(features, labels, mode, params):
    """model_fn for MODLE to be used with our Estimator."""
    # cls_targets = labels['cls_targets']
    loc_targets = labels['loc_targets']
    is_reg = labels['is_reg']

    with tf.variable_scope(params['model_scope'], default_name=None, values=[features], reuse=tf.AUTO_REUSE):
        model = cls_net.CLS_REG_Model(FLAGS.resnet_size, FLAGS.resnet_version,
                                FLAGS.attention_block, FLAGS.location_feature_stage, FLAGS.data_format)
        results = model(features, mode == tf.estimator.ModeKeys.TRAIN)
        print(results)
        if FLAGS.location_feature_stage is not None:
            logits, loc, location, location_c = results
        else:
            logits = results

    if mode == tf.estimator.ModeKeys.TRAIN:
        if FLAGS.checkpoint_path:
            init_variables_from_checkpoint(FLAGS.checkpoint_exclude_scopes)

    # This acts as a no-op if the logits are already in fp32 (provided logits are
    # not a SparseTensor). If dtype is is low precision, logits must be cast to
    # fp32 for numerical stability.
    logits = tf.cast(logits, tf.float32)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        # Return the predictions and the specification for serving a SavedModel
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    with tf.variable_scope('losses'):
        if FLAGS.location_feature_stage is not None:
            # weights = tf.concat([tf.zeros([int(FLAGS.batch_size/2)]), tf.ones(int(FLAGS.batch_size/2))], axis=0)
            # print_op2 = tf.print("reg:", loc,location,loc_targets, output_stream=sys.stdout)
            # with tf.control_dependencies([print_op2]):
            with tf.variable_scope('reg_loss'):
                coarse_label = loc_targets # b*[ly, lx, ry, rx]
                loc_loss1 = tf.losses.absolute_difference(
                    labels=loc_targets, predictions=location)
                print(loc_targets)
                loc_loss2 = tf.losses.absolute_difference(
                    labels=loc_targets[:,:,-1], predictions=location_c)
            # Reset the total_loss.
            total_loss = loc_loss1 + tf.stop_gradient(loc_loss2)
            # total_loss = tf.stop_gradient(total_loss) + tf.stop_gradient(loc_loss1) + loc_loss2
            # total_loss = total_loss + loc_loss1 + tf.stop_gradient(loc_loss2)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(loc_loss1, name='loc_loss1')
    tf.summary.scalar('loc_loss1', loc_loss1)
    tf.identity(loc_loss2, name='loc_loss2')
    tf.summary.scalar('loc_loss2', loc_loss2)
    tf.identity(total_loss, name='total_loss')
    tf.summary.scalar('total_loss', total_loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        lr_values = [params['learning_rate'] * decay for decay in params['lr_decay_factors']]
        learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32),
                                                    [int(_) for _ in params['decay_boundaries']],
                                                    lr_values)
        truncated_learning_rate = tf.maximum(learning_rate, tf.constant(params['end_learning_rate'],
                                 dtype=learning_rate.dtype), name='learning_rate')
        # Create a tensor named learning_rate for logging purposes.
        tf.summary.scalar('learning_rate', truncated_learning_rate)

        optimizer = tf.train.MomentumOptimizer(learning_rate=truncated_learning_rate,
                                                momentum=params['momentum'])
        optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        # Batch norm requires update_ops to be added as a train_op dependency.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step)
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
                              mode=mode,
                              predictions=predictions,
                              loss=total_loss,
                              train_op=train_op)
                            #   eval_metric_ops=metrics,
                            #   scaffold=tf.train.Scaffold(init_fn=get_init_fn()))

def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

def main(_):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '5'

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    gpu_options = tf.GPUOptions(allow_growth = True)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, intra_op_parallelism_threads=FLAGS.num_cpu_threads, inter_op_parallelism_threads=FLAGS.num_cpu_threads, gpu_options=gpu_options)

    num_gpus = validate_batch_size_for_multi_gpu(FLAGS.batch_size)

    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(
                                        save_checkpoints_secs=None).replace(
                                        save_checkpoints_steps=FLAGS.save_checkpoints_steps).replace(
                                        save_summary_steps=FLAGS.save_summary_steps).replace(
                                        keep_checkpoint_max=5).replace(
                                        tf_random_seed=FLAGS.tf_random_seed).replace(
                                        log_step_count_steps=FLAGS.log_every_n_steps).replace(
                                        session_config=config)

    replicate_ssd_model_fn = tf.contrib.estimator.replicate_model_fn(ssd_model_fn, loss_reduction=tf.losses.Reduction.MEAN)
    ssd_detector = tf.estimator.Estimator(
        model_fn=replicate_ssd_model_fn, model_dir=FLAGS.model_dir, config=run_config,
        params={
            'num_gpus': num_gpus,
            'data_format': FLAGS.data_format,
            'batch_size': FLAGS.batch_size,
            'model_scope': FLAGS.model_scope,
            'num_classes': FLAGS.num_classes,
            # 'negative_ratio': FLAGS.negative_ratio,
            # 'match_threshold': FLAGS.match_threshold,
            # 'neg_threshold': FLAGS.neg_threshold,
            'weight_decay': FLAGS.weight_decay,
            'momentum': FLAGS.momentum,
            'learning_rate': FLAGS.learning_rate,
            'end_learning_rate': FLAGS.end_learning_rate,
            'decay_boundaries': parse_comma_list(FLAGS.decay_boundaries),
            'lr_decay_factors': parse_comma_list(FLAGS.lr_decay_factors),
        })
    tensors_to_log = {
        'lr': 'learning_rate',
        # 'loss-ce': 'cross_entropy',
        'loss-loc1': 'loc_loss1',
        'loss-loc2': 'loc_loss2',
        'total-loss': 'total_loss'
    }
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps,
                                            formatter=lambda dicts: (', '.join(['%s=%.6f' % (k, v) for k, v in dicts.items()])))

    #hook = tf.train.ProfilerHook(save_steps=50, output_dir='.', show_memory=True)
    print('Starting a training cycle.')
    ssd_detector.train(input_fn=input_pipeline(dataset_pattern='train-*', is_training=True, batch_size=FLAGS.batch_size),
                    hooks=[logging_hook], max_steps=FLAGS.max_number_of_steps)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

