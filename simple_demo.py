# coding=utf-8
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
# from tensorflow.python.client import timeline   
# # 可以单独用它生成 timeline，也可以使用下面两个对象生成 timeline
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

from scipy.misc import imread, imsave, imshow, imresize
import numpy as np

# from net import cls_net
from net import cls_reg_net

from dataset import dataset_common
from preprocessing import cls_preprocessing

# scaffold related configuration
tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes to use in the dataset.')
# model related configuration
## resnet settings
tf.app.flags.DEFINE_integer(
    'resnet_version', 2,
    'The version of used resnet, default is v2.')
tf.app.flags.DEFINE_integer(
    'resnet_size', 18,
    'The layers of used resnet, [4,6,10,14,18,34,50,101,152,200].')
## attention
tf.app.flags.DEFINE_boolean(
    'attention_block', True,
    'Use attention or not. True/False')
## regression
tf.app.flags.DEFINE_integer(
    'location_feature_stage', 1,
    'Regression from which stage. None/1/2/3/4')
## other settings
tf.app.flags.DEFINE_integer(
    'train_image_size', 224,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_string(
    'data_format', 'channels_first', # 'channels_first' or 'channels_last'
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
# checkpoint related configuration
if tf.app.flags.FLAGS.attention_block: Attribute = str(tf.app.flags.FLAGS.resnet_size)+'att'
else: Attribute = str(tf.app.flags.FLAGS.resnet_size)
tf.app.flags.DEFINE_string(
    'checkpoint_path', './models/multitask/18_atten_9p',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'model_scope', 'ssd300',
    'Model scope name used to replace the name_scope in checkpoint.')

FLAGS = tf.app.flags.FLAGS
#CUDA_VISIBLE_DEVICES

def get_checkpoint():
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    return checkpoint_path

def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    with tf.Graph().as_default():
        out_shape = [FLAGS.train_image_size] * 2

        image_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        shape_input = tf.placeholder(tf.int32, shape=(2,))

        features = cls_preprocessing.preprocess_for_eval(image_input, out_shape, data_format=FLAGS.data_format, output_rgb=False)
        features = tf.expand_dims(features, axis=0)

        with tf.variable_scope(FLAGS.model_scope, default_name=None, values=[features], reuse=tf.AUTO_REUSE):
            model = cls_reg_net.CLS_REG_Model(FLAGS.resnet_size, FLAGS.resnet_version,
                                    FLAGS.attention_block, FLAGS.location_feature_stage,
                                    FLAGS.data_format)

            results = model(features, training=False)
            if FLAGS.location_feature_stage:
                logits, loc, location = results
            else:
                logits = results
        # tf.summary.image('base',tf.reshape(tf.range(9, dtype=tf.float32), [1,3,3,1]))
        tf.summary.image('origin_pic',tf.transpose(features, [0, 2, 3, 1]))
        # tf.summary.image('att_map', tf.transpose(att_map, [0, 2, 3, 1]))
        # tf.summary.image('loc', tf.transpose(loc, [0, 2, 3, 1]))
        merged = tf.summary.merge_all()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # 创建 profiler 对象
            my_profiler = model_analyzer.Profiler(graph=sess.graph)
            # 创建 metadata 对象
            run_metadata = tf.RunMetadata()
            run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
            init = tf.global_variables_initializer()
            sess.run(init)
            saver.restore(sess, get_checkpoint())

            # init summary writer
            writer = tf.summary.FileWriter("./demo/test_out/" ,sess.graph)
            i = 0
            for picname in os.listdir('./demo'):
                if picname.split('.')[-1] != 'jpg':
                    print(picname)
                    continue
                np_image = imread(os.path.join('./demo',picname))

                print(type(np_image), np_image.shape)
                # exit()
                logits_, loc_, location_, summary= sess.run([logits, loc, location, merged], 
                                                        feed_dict = {image_input : np_image, shape_input : np_image.shape[:-1]},
                                                        options=run_options, run_metadata=run_metadata)
                my_profiler.add_step(step=i, run_meta=run_metadata)

                # att = att.reshape([-1])
                # ma = np.argmax(att)
                # mi = np.argmin(att)
                # print(res)
                # # print(ma/28, ma%28, mi/28, mi%28)
                # print(att[ma],att[mi])
                # print(lo)
                writer.add_summary(summary,i)
                i+=1
                # img_to_draw = draw_toolbox.bboxes_draw_on_img(np_image, labels_, scores_, bboxes_, thickness=2)
                # imsave('./demo/test_out.jpg', img_to_draw)

            #统计内容为每个graph node的运行时间和占用内存
            profile_graph_opts_builder = option_builder.ProfileOptionBuilder(
            option_builder.ProfileOptionBuilder.time_and_memory())

            #输出方式为timeline
            profile_graph_opts_builder.with_timeline_output(timeline_file='/tmp/profiler.json')
            #定义显示sess.Run() 第70步的统计数据
            profile_graph_opts_builder.with_step(3)

            #显示视图为graph view
            my_profiler.profile_graph(profile_graph_opts_builder.build())

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
