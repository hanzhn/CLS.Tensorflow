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

import tensorflow as tf

import numpy as np
from scipy.misc import imread, imsave, imshow, imresize

from net import cls_net

from dataset import dataset_common
from preprocessing import cls_preprocessing
from utility import scaffolds

# hardware related configuration
tf.app.flags.DEFINE_integer(
    'num_readers', 8,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 24,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'num_cpu_threads', 0,
    'The number of cpu cores used to train.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.1, 'GPU memory fraction to use.')
# scaffold related configuration
tf.app.flags.DEFINE_string(
    'data_dir', './dataset/tfrecords/test0.2',
    'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'model_dir', './test0.2/logs4att/',
    'The directory where the model will be stored.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are printed.')
# model related configuration
## resnet settings
tf.app.flags.DEFINE_integer(
    'resnet_version', 2,
    'The version of used resnet, default is v2.')
tf.app.flags.DEFINE_integer(
    'resnet_size', 4,
    'The layers of used resnet, [4,6,10,14,18,34,50,101,152,200].')
tf.app.flags.DEFINE_boolean(
    'attention_block', True,
    'Use attention or not. True/False')
## other settings
tf.app.flags.DEFINE_integer(
    'batch_size', 1,
    'Batch size for training and evaluation.')
tf.app.flags.DEFINE_integer(
    'train_image_size', 224,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_string(
    'data_format', 'channels_first', # 'channels_first' or 'channels_last'
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
# tf.app.flags.DEFINE_float(
#     'select_threshold', 0.01, 'Class-specific confidence score threshold for selecting a box.')
# tf.app.flags.DEFINE_float(
#     'min_size', 4., 'The min size of bboxes to keep.')
# tf.app.flags.DEFINE_float(
#     'nms_threshold', 0.45, 'Matching threshold in NMS algorithm.')
# tf.app.flags.DEFINE_integer(
#     'nms_topk', 200, 'Number of total object to keep after NMS.')
# tf.app.flags.DEFINE_integer(
#     'keep_topk', 400, 'Number of total object to keep for each image before nms.')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', './model',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'model_scope', 'ssd300',
    'Model scope name used to replace the name_scope in checkpoint.')

FLAGS = tf.app.flags.FLAGS
#CUDA_VISIBLE_DEVICES

def get_checkpoint():
    if tf.train.latest_checkpoint(FLAGS.model_dir):
        tf.logging.info('Ignoring --checkpoint_path because a checkpoint already exists in %s' % FLAGS.model_dir)
        return None

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    return checkpoint_path


def input_pipeline(dataset_pattern='val-*', is_training=False, batch_size=FLAGS.batch_size):
    def input_fn():
        assert batch_size==1, 'We only support single batch when evaluation.'
        target_shape = [FLAGS.train_image_size] * 2
        image_preprocessing_fn = lambda image_,: cls_preprocessing.preprocess_image(image_, target_shape, is_training=is_training, data_format=FLAGS.data_format, output_rgb=False)

        image, filename, label = dataset_common.slim_get_batch(FLAGS.num_classes,
                                                                            batch_size,
                                                                            ('train' if is_training else 'val'),
                                                                            os.path.join(FLAGS.data_dir, dataset_pattern),
                                                                            FLAGS.num_readers,
                                                                            FLAGS.num_preprocessing_threads,
                                                                            image_preprocessing_fn,
                                                                            num_epochs=1,
                                                                            is_training=is_training)

        return {'image': image, 'filename': filename, 'label': label}, None
    return input_fn

def ssd_model_fn(features, labels, mode, params):
    """model_fn for MODLE to be used with our Estimator."""
    filename = features['filename']
    label = features['label']
    image = features['image']

    filename = tf.identity(filename, name='filename')
    with tf.variable_scope(params['model_scope'], default_name=None, values=[image], reuse=tf.AUTO_REUSE):
        model = cls_net.CLS_Model(FLAGS.resnet_size, FLAGS.resnet_version,
                                FLAGS.attention_block, FLAGS.data_format)
        logits = model(image, mode == tf.estimator.ModeKeys.TRAIN)

    # This acts as a no-op if the logits are already in fp32 (provided logits are
    # not a SparseTensor). If dtype is is low precision, logits must be cast to
    # fp32 for numerical stability.
    logits = tf.cast(logits, tf.float32)

    # save_image_op = tf.py_func(save_image_with_label,
    #                     [ssd_preprocessing.unwhiten_image(tf.squeeze(features, axis=0), output_rgb=False),
    #                     all_labels * tf.to_int32(all_scores > 0.3),
    #                     all_scores,
    #                     all_bboxes],
    #                     tf.int64, stateful=True)
    # tf.identity(save_image_op, name='save_image_op')
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
        'label': label,
        'filename': filename
    }
    # tf.summary.image('att_map', att_map)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Return the predictions and the specification for serving a SavedModel
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })
    else:
        raise ValueError('This script only support "PREDICT" mode!')

def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

def main(_):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, intra_op_parallelism_threads=FLAGS.num_cpu_threads, inter_op_parallelism_threads=FLAGS.num_cpu_threads, gpu_options=gpu_options)

    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(
                                        save_checkpoints_secs=None).replace(
                                        save_checkpoints_steps=None).replace(
                                        save_summary_steps=50).replace(
                                        keep_checkpoint_max=5).replace(
                                        log_step_count_steps=FLAGS.log_every_n_steps).replace(
                                        session_config=config)

    summary_dir = os.path.join(FLAGS.model_dir, 'predict')
    tf.gfile.MakeDirs(summary_dir)
    ssd_detector = tf.estimator.Estimator(
        model_fn=ssd_model_fn, model_dir=FLAGS.model_dir, config=run_config,
        params={
            # 'select_threshold': FLAGS.select_threshold,
            # 'min_size': FLAGS.min_size,
            # 'nms_threshold': FLAGS.nms_threshold,
            # 'nms_topk': FLAGS.nms_topk,
            # 'keep_topk': FLAGS.keep_topk,
            'data_format': FLAGS.data_format,
            'batch_size': FLAGS.batch_size,
            'model_scope': FLAGS.model_scope,
            'num_classes': FLAGS.num_classes,
        })
    tensors_to_log = {
        'cur_image': 'filename',
    }
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps)

    print('Starting a predict cycle.')
    pred_results = ssd_detector.predict(input_fn=input_pipeline(dataset_pattern='val-*', is_training=False, batch_size=FLAGS.batch_size),
                                    hooks=[logging_hook], checkpoint_path=get_checkpoint())#, yield_single_examples=False)

    correct = []
    wrong = []
    print('________',pred_results)
    for i in pred_results:
        pred = i['classes']
        label = i['label']
        filename = i['filename']
        if pred == label:
            correct.append(filename)
        else:
            wrong.append(filename)
    print('correct:',correct,len(correct))
    print('wrong:',wrong,len(wrong))
    print('precision:',float(len(correct))/( len(correct)+len(wrong) ) )


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.gfile.MakeDirs('./debug')
  tf.app.run()
