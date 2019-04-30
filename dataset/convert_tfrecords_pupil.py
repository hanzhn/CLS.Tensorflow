import scipy.io as sio
import os
import io

import PIL.Image
import cv2
import tensorflow as tf

import contextlib2
import dataset_ultility as dataset_util

num_shards=5
part = "train"

flags = tf.app.flags
flags.DEFINE_string('dataset_root', '/hz/data/unity_croped/','Root to CEW')
flags.DEFINE_string('output_path', '/hz/eye_clossness_cls/dataset/tfrecords/pupil/'+part,
                     'Path to output TFRecord')

flags.DEFINE_string('task_type', 'reg', 'Task that the data is labeled for.')
flags.DEFINE_string('dataset_name', 'UNITY', 'what label info dict to choose')
FLAGS = flags.FLAGS

class Example():
  def __init__(self, subdir, file_name, label_list, task_type):
    # print(task_type)
    self.set_image(subdir, file_name)

    self.label, self.label_text = -1, "None" # unrelavent
    [self.roundx,self.roundy] = label_list['landmarks'] # unrelavent
    self.task_type = 1

  def set_image(self, dataset_root, file_name):
    self.file_name = file_name.encode('utf8')

    image_path = os.path.join(dataset_root, file_name)
    print( image_path)
    cv_image = cv2.imread(image_path)
    self.image = cv_image
    self.h = cv_image.shape[0]
    self.w = cv_image.shape[1]

    with tf.gfile.GFile(image_path, 'rb') as fid:
      self.encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(self.encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
      raise ValueError('Image format not JPEG')

def create_tf_example(example):
  print(example.w,example.h, example.file_name)
  
  height, width= example.h, example.w # Image width # height
  filename = example.file_name # Filename of the image. Empty if image is not from file
  encoded_image_data = example.encoded_jpg # Encoded image bytes
  image_format = b'jpeg' # b'jpeg' or b'png'

  roundx, roundy = [x/float(width) for x in example.roundx], [y/float(height) for y in example.roundy]
  task = example.task_type

  channels = 3
  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/encoded':        dataset_util.bytes_feature(encoded_image_data),
      'image/format':         dataset_util.bytes_feature(image_format),
      'image/filename':       dataset_util.bytes_feature(filename),
      'image/shape':          dataset_util.int64_feature([height, width, channels]),
      'label/task_type':      dataset_util.int64_feature(task),
      'label/loc/roundx': dataset_util.float_feature(roundx),
      'label/loc/roundy': dataset_util.float_feature(roundy)
  }))
  # exit()
  return tf_example

def create_tf_record_reg(task_type):
  UNITY_LABELS = {
    'round_shift': range(1,66,2),
    'subdir': part,
    'label_dir': 'landmark-'+part+'.csv',
  }

  if FLAGS.dataset_name == 'MUCT':
    LABELS = MUCT_LABELS
  elif FLAGS.dataset_name == 'BIOID':
    LABELS = BIOID_LABELS
  elif FLAGS.dataset_name == 'UNITY':
    LABELS = UNITY_LABELS
  else:
    exit('Unknow dataset name!')
  dataset_root = FLAGS.dataset_root

  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = dataset_util.open_sharded_output_tfrecords(
        tf_record_close_stack, FLAGS.output_path, num_shards)

    index = 0
    subdir = os.path.join(dataset_root,  LABELS['subdir'])
    labeldir = os.path.join(dataset_root, LABELS['label_dir'])

    image_labels = {}
    with open(labeldir) as f:
      for line in f:
        line = line.strip().split(',')
        # if line[0]=='name': 
        #   continue # exclude the first line as header line
        roundx = [float(line[x]) for x in LABELS['round_shift']]
        roundy = [float(line[x+1]) for x in LABELS['round_shift']]
        image_labels[line[0]+'.jpg'] = {"landmarks":[roundx, roundy]}

    for file in os.listdir(subdir):
        file_name = file
        # print(file)
        label_list = image_labels[file]
        # print(label_list)
        example = Example(subdir, file_name, label_list, task_type)

        tf_example = create_tf_example(example)
        output_shard_index = index % num_shards
        output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
        index += 1
        print (index)

def main(_):
  if FLAGS.task_type =='cls': create_tf_record_cls('cls')
  elif FLAGS.task_type =='reg': create_tf_record_reg('reg')
  else: print('unknow task type!')

if __name__ == '__main__':
  tf.app.run()
