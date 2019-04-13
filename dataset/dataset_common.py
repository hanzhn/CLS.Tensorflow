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

slim = tf.contrib.slim

CEW_LABELS = {
    'OpenFace': (0, 'open'),
    'ClosedFace': (1, 'closed'),
}

# use dataset_inspect.py to get these summary
data_splits_num = {
    'train': 1939,
    'val': 484,
}

def slim_get_batch(num_classes, batch_size, split_name, file_pattern, num_readers, num_preprocessing_threads, image_preprocessing_fn, num_epochs=None, is_training=True):
    """Gets a dataset tuple with instructions for reading Pascal VOC dataset.

    Args:
      num_classes: total class numbers in dataset.
      batch_size: the size of each batch.
      split_name: 'train' of 'val'.
      file_pattern: The file pattern to use when matching the dataset sources (full path).
      num_readers: the max number of reader used for reading tfrecords.
      num_preprocessing_threads: the max number of threads used to run preprocessing function.
      image_preprocessing_fn: the function used to dataset augumentation.
      anchor_encoder: the function used to encoder all anchors.
      num_epochs: total epoches for iterate this dataset.
      is_training: whether we are in traing phase.

    Returns:
      A batch of [image, shape, loc_targets, cls_targets, match_scores].
    """
    if split_name not in data_splits_num:
        raise ValueError('split name %s was not recognized.' % split_name)

    # Features in Pascal VOC TFRecords.
    keys_to_features = {
        'image/encoded':        tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':         tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename':       tf.FixedLenFeature((), tf.string, default_value=''),
        'image/shape':          tf.FixedLenFeature([3], tf.int64),
        'label/cls/label':      tf.FixedLenFeature([1], tf.int64),
        'label/cls/label_text': tf.FixedLenFeature((), tf.string, default_value=''),
        'label/loc/leftx':      tf.FixedLenFeature([1], tf.float32),
        'label/loc/lefty':      tf.FixedLenFeature([1], tf.float32),
        'label/loc/rightx':     tf.FixedLenFeature([1], tf.float32),
        'label/loc/righty':     tf.FixedLenFeature([1], tf.float32),
        'label/task_type':      tf.FixedLenFeature((), tf.int64),
        'label/loc/left_roundx':      tf.FixedLenFeature([9], tf.float32),
        'label/loc/left_roundy':      tf.FixedLenFeature([9], tf.float32),
        'label/loc/right_roundx':     tf.FixedLenFeature([9], tf.float32),
        'label/loc/right_roundy':     tf.FixedLenFeature([9], tf.float32),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'filename': slim.tfexample_decoder.Tensor('image/filename'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'label': slim.tfexample_decoder.Tensor('label/cls/label'),
        # 'leftx': slim.tfexample_decoder.Tensor('label/loc/leftx'),
        # 'lefty': slim.tfexample_decoder.Tensor('label/loc/lefty'),
        # 'rightx': slim.tfexample_decoder.Tensor('label/loc/rightx'),
        # 'righty': slim.tfexample_decoder.Tensor('label/loc/righty'),
        'task_type': slim.tfexample_decoder.Tensor('label/task_type'),
        'left_roundx': slim.tfexample_decoder.Tensor('label/loc/left_roundx'),
        'left_roundy': slim.tfexample_decoder.Tensor('label/loc/left_roundy'),
        'right_roundx': slim.tfexample_decoder.Tensor('label/loc/right_roundx'),
        'right_roundy': slim.tfexample_decoder.Tensor('label/loc/right_roundy'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    labels_to_names = {}
    for name, pair in CEW_LABELS.items():
        labels_to_names[pair[0]] = name

    dataset = slim.dataset.Dataset(
                data_sources=file_pattern,
                reader=tf.TFRecordReader,
                decoder=decoder,
                num_samples=data_splits_num[split_name],
                items_to_descriptions=None,
                num_classes=num_classes,
                labels_to_names=labels_to_names)

    with tf.name_scope('dataset_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=num_readers,
            common_queue_capacity=32 * batch_size,
            common_queue_min=8 * batch_size,
            shuffle=is_training,
            num_epochs=num_epochs)

    org_image, filename, glabels_raw, leftx, lefty, rightx, righty, task_type = provider.get(
        ['image', 'filename', 'label', 'left_roundx','left_roundy','right_roundx','right_roundy', 'task_type'])
    image, [[leftx, lefty], [rightx, righty]]= image_preprocessing_fn(org_image, [[leftx, lefty], [rightx, righty]])
    # return image
    tensors_to_batch = [image, filename, glabels_raw, [lefty, leftx, righty, rightx], task_type]
    return tf.train.batch(tensors_to_batch,
                    dynamic_pad=(not is_training),
                    batch_size=batch_size,
                    allow_smaller_final_batch=(not is_training),
                    num_threads=num_preprocessing_threads,
                    capacity=64 * batch_size)


if __name__ == '__main__':
    num_classes = 2
    batch_size = 1
    split_name = 'train'
    file_pattern = './tfrecords/multi-task/CEW/train*'
    num_readers = 8
    num_preprocessing_threads = 24
    import cls_preprocessing
    image_preprocessing_fn = cls_preprocessing.preprocess_image
    
    image, filename, glabels_raw, leftx, lefty, rightx, righty  = slim_get_batch(
        num_classes, batch_size, split_name, file_pattern, num_readers, num_preprocessing_threads, 
        image_preprocessing_fn, num_epochs=None, is_training=True)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        # for i in range(10):
        #     [img, fn, sh, gtt, gtl, gts]= sess.run(
        #         [org_image, filename, shape, glabels_raw, gbboxes_raw, isdifficult])        
        #     img = tf.reshape(img, sh) 
        #     print(img.shape,fn, sh, gtt, gtl, gts)       

        print(sess.run([image, filename, glabels_raw, leftx, lefty, rightx, righty]))
        # print(sess.run([image]))
    
        coord.request_stop()
        coord.join(threads)  