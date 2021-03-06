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
import cv2

from net import cls_reg_net

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
    'data_dir', './dataset/tfrecords/multi-task/9p/CEW',
    'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'model_dir', './models/multitask/18_atten_9p/step4',
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
    'checkpoint_path', './models/multitask/9p/18_atten',
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
        image_preprocessing_fn = lambda image_, label_: cls_preprocessing.preprocess_image(image_, label_, target_shape, 
                                                                    is_training=is_training, data_format=FLAGS.data_format, 
                                                                    output_rgb=False)
        image, filename, label, points, is_reg = dataset_common.slim_get_batch(
                                                            FLAGS.num_classes,
                                                            batch_size,
                                                            ('train' if is_training else 'val'),
                                                            os.path.join(FLAGS.data_dir, dataset_pattern),
                                                            FLAGS.num_readers,
                                                            FLAGS.num_preprocessing_threads,
                                                            image_preprocessing_fn,
                                                            num_epochs=1,
                                                            is_training=is_training)
        return {'image': image, 'filename': filename, 'label': label, 'points':points, 'is_reg': is_reg}, None
    return input_fn

def ssd_model_fn(features, labels, mode, params):
    """model_fn for MODLE to be used with our Estimator."""
    filename = features['filename']
    label = features['label']
    image = features['image']
    points = features['points']
    is_reg = features['is_reg']
    points2 = points[:,:,-1]

    filename = tf.identity(filename, name='filename')
    with tf.variable_scope(params['model_scope'], default_name=None, values=[image], reuse=tf.AUTO_REUSE):
        model = cls_reg_net.CLS_REG_Model(FLAGS.resnet_size, FLAGS.resnet_version,
                                FLAGS.attention_block, FLAGS.location_feature_stage, FLAGS.data_format)
        results = model(image, training=(mode==tf.estimator.ModeKeys.TRAIN))
        if FLAGS.location_feature_stage is not None:
            logits, loc_ori, location = results
            loc = tf.reshape(loc_ori, [-1, 4, 1])

            with tf.variable_scope('DSNT'):
                inputs_shape_hw = int(FLAGS.train_image_size / 2 *2/7)
                kernel = np.arange(inputs_shape_hw)
                kernel = kernel / (inputs_shape_hw-1.) - 0.5
                X = np.tile(kernel, [inputs_shape_hw, 1])
                Y = np.transpose(X, [1,0])
                # channel_last format
                X = np.reshape(X, [1, -1, 1])
                Y = np.reshape(Y, [1, -1, 1])

                eyes = []
                print(location)
                for inputs in location:
                    # inputs: channel_last format
                    # print_op2 = tf.print("location:", inputs, output_stream=sys.stdout)
                    # with tf.control_dependencies([print_op2]):
                    # inputs = tf.Print(inputs, [inputs, tf.shape(inputs)])
                    print(inputs_shape_hw**2, inputs_shape_hw)
                    inputs = tf.reshape(inputs, [-1, inputs_shape_hw**2, 9])
                    inputs = tf.nn.softmax(inputs, 1)
                    # print(X, inputs)
                    xs = tf.reduce_sum(X*inputs, 1, keepdims=True) #B*1*C
                    ys = tf.reduce_sum(Y*inputs, 1, keepdims=True) #B*1*C
                    eyes.append( tf.concat([ys,xs], 1) )                #B*2*C
                location = tf.concat(eyes, 1)   # B*4*C

            location = tf.reshape(location, [-1, 4, 9])
            location = loc + location*0.25
        else:
            logits = results

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

    # tensors to print
    if FLAGS.data_format=='channels_first':
        image = tf.transpose(image, (0,2,3,1))
    image = tf.expand_dims(cls_preprocessing.unwhiten_image(tf.squeeze(image,0), output_rgb=False),0)
    feature_map_shape = tf.shape(image)
    feature_map_size = [feature_map_shape[1], feature_map_shape[2]]
    crop_size = [tf.to_int32(x*2/7) for x in feature_map_size]  # Crop the 1/4 of feature map.
    crop_centers = tf.split(loc_ori, 2, axis=1) # point_num*[b*2]
    # crop_centers = tf.split(loc_dence, point_num, axis=1) # point_num*[b*2]
    patches = []
    centers = []
    for center in crop_centers:
        begin = center - 2/7*0.5
        end = center + 2/7*0.5
        boxes = tf.concat([begin, end], axis=1)
        patch = tf.image.crop_and_resize(
            image,
            boxes=boxes,
            box_ind=tf.range(feature_map_shape[0]),
            crop_size = crop_size,
            method='bilinear', extrapolation_value=0)
        patches.append(patch)
        centers.append(center)
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
        'label': label,
        'filename': filename,
        'loc': loc,
        'location': location,
        'points': points,
        'points2': points2,
        'image': image,
        'patchl':patches[0],
        'patchr':patches[1],
        'center1':centers[0],
        'center2':centers[1],
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

def distance(a, b):
    dis = (a[0]-b[0])**2 + (a[1]-b[1])**2
    return pow(dis, 1./2.)

def d_eye(loc, points):
    dl = distance(loc[:2], points[:2])
    dr = distance(loc[2:], points[2:])
    dcc = distance(points[:2], points[2:])
    return max(dl, dr)/dcc

def main(_):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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
    deye1 = []
    deye2 = []
    # print('________',pred_results)
    for i in pred_results:
        img = i['image']
        pred = i['classes']
        label = i['label']
        filename = i['filename']
        loc = i['loc']
        location = i['location']
        points9 = i['points']
        points = i['points2']
        patchl = i['patchl']
        patchr = i['patchr']        
        center1 = i['center1']
        center2 = i['center2']
        # print(location)
        # print(points)
        # print(loc, location, points)
        if pred == label:
            correct.append(filename)
        else:
            wrong.append(filename)

        deye1.append(d_eye(loc.reshape([4]), points))
        deye2.append(d_eye(location[:,-1].reshape([4]), points))

        # img = cv2.imread(os.path.join('/data0/hz/BioID_croped/Picture_extracted', filename))
        # img = cv2.imread(os.path.join('/data0/hz/muct_croped/jpg', filename))
        

        print(img.shape)
        w, h = img.shape[1], img.shape[0]
        # img = np.transpose(img,[1,0,2])
        img = cv2.transpose(img)

        pl = location.reshape([2,2,9])
        pl = np.transpose(pl, (0,2,1))
        pl = pl.reshape([18,2])
        pl = pl*[h,w]
        pl = pl.astype(np.int64)
        for center in pl:
            center = tuple(center)
            # print(center)
            cv2.circle(img, center, radius=1, color=(255, 0, 0), thickness=1)
        # cv2.imwrite(os.path.join(summary_dir, filename), img)
        # exit()
        pl = points.reshape([2,2])
        pl = pl*[h,w]
        pl = pl.astype(np.int64)
        for center in pl:
            center = tuple(center)
            cv2.circle(img, center, radius=1, color=(0, 0, 255), thickness=1)

        pl = loc.reshape([2,2])
        pl = pl*[h,w]
        pl = pl.astype(np.int64)
        for center in pl:
            center = tuple(center)
            cv2.circle(img, center, radius=1, color=(0, 255, 0), thickness=1)
        center1=center1*[h,w]
        center2=center2*[h,w]
        cv2.imwrite(os.path.join(summary_dir, filename+'.l-{}-{}.jpg'.format(int(center1[0]),int(center1[1]))), patchl)
        cv2.imwrite(os.path.join(summary_dir, filename+'.r-{}-{}.jpg'.format(int(center2[0]),int(center2[1]))), patchr)
        img = np.transpose(img,[1,0,2])
        cv2.imwrite(os.path.join(summary_dir, filename), img)
        # exit()
        
            

    # print('correct:',correct,len(correct))
    print('wrong:',wrong,len(wrong))
    print('precision:',float(len(correct))/( len(correct)+len(wrong) ) )
    print('0.25')
    print(np.argwhere(np.array(deye1)<0.25).shape[0]/len(deye1))
    print(np.argwhere(np.array(deye2)<0.25).shape[0]/len(deye2))
    print('0.10')
    print(np.argwhere(np.array(deye1)<0.1).shape[0]/len(deye1))
    print(np.argwhere(np.array(deye2)<0.1).shape[0]/len(deye2))
    print('0.05')
    print(np.argwhere(np.array(deye1)<0.05).shape[0]/len(deye1))
    print(np.argwhere(np.array(deye2)<0.05).shape[0]/len(deye2))
    print('deye1:', np.mean(deye1))
    print('deye2:', np.mean(deye2))
    


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.gfile.MakeDirs('./debug')
  tf.app.run()
