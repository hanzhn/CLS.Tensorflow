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
flags.DEFINE_string('output_path', '/home/hz/project/eye_closeness_cls/dataset/tfrecords/multi-task/MUCT/'+part,
                     'Path to output TFRecord')
# flags.DEFINE_string('dataset_root', '/data0/hz/dataset_B_FacialImages/test0.2/'+part+'/',
#                      'Root to widerface events image')
flags.DEFINE_string('dataset_root', '/data0/hz/muct_croped',
                     'Root to widerface events image')
flags.DEFINE_string('task_type', 'reg',
                      'Task that the data is labeled for.')
FLAGS = flags.FLAGS

class Example():
  def __init__(self, subdir, file_name, label_list, task_type):
    self.set_image(subdir, file_name)
    if task_type =='cls':
      self.label, self.label_text = label_list['label'], label_list['label_text']
      self.left, self.right = [-1, -1], [-1, -1] # unrelavent
      self.task_type = 0
    elif task_type =='reg':
      self.label, self.label_text = -1, "None" # unrelavent
      self.left, self.right = label_list['left'], label_list['right']
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

  label = example.label
  label_text = example.label_text
  left = example.left
  right = example.right
  task = example.task_type

  channels = 3
  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/encoded':        dataset_util.bytes_feature(encoded_image_data),
      'image/format':         dataset_util.bytes_feature(image_format),
      'image/filename':       dataset_util.bytes_feature(filename),
      'image/shape':          dataset_util.int64_feature([height, width, channels]),
      'label/cls/label':      dataset_util.int64_feature(label),
      'label/cls/label_text': dataset_util.bytes_feature(label_text),
      'label/loc/left_eye':   dataset_util.float_feature(left),
      'label/loc/right_eye':  dataset_util.float_feature(right),
      'label/task_type':      dataset_util.int64_feature(task)
  }))
  # exit()
  return tf_example




def create_tf_record_cls(task_type):
  CEW_LABELS = {
    'OpenFace': {'label':0, 'label_text':'open'},
    'ClosedFace': {'label':1, 'label_text':'closed'},
  }
  dataset_root = FLAGS.dataset_root

  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = dataset_util.open_sharded_output_tfrecords(
        tf_record_close_stack, FLAGS.output_path, num_shards)

    index = 0
    for subkey in CEW_LABELS:
      subdir = os.path.join(dataset_root, subkey)
      for file in os.listdir(subdir):
          file_name = file
          label_list = CEW_LABELS[subkey]
          example = Example(subdir, file_name, label_list, task_type)

          tf_example = create_tf_example(example)
          output_shard_index = index % num_shards
          output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
          index += 1
          print (index)

def apply_shift(label_list, shift_list):
  shiftxy=shift_list[:2]
  cropwh=shift_list[2:]
  label_list['left']=[ (label_list['left'][i]-shiftxy[i])/cropwh[i] for i in range(2) ]
  label_list['right']=[ (label_list['right'][i]-shiftxy[i])/cropwh[i] for i in range(2) ]
  # print(label_list)
  return label_list

def create_tf_record_reg(task_type):
  MUCT_LABELS = {
    'left_idx_shift' : 31,
    'right_idx_shift' : 73
  }
  dataset_root = FLAGS.dataset_root

  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = dataset_util.open_sharded_output_tfrecords(
        tf_record_close_stack, FLAGS.output_path, num_shards)

    index = 0
    subdir = os.path.join(dataset_root, 'jpg')
    labeldir = os.path.join(dataset_root, 'muct-landmarks', 'muct76-opencv.csv')
    cropdir = os.path.join(dataset_root, 'rect_shift.csv')

    image_crop_shift = {}
    with open(cropdir) as f:
      for line in f:
        line = line.strip().split(',')
        name, xshift, yshift, width, height = line
        image_crop_shift[name] = map(float, [xshift, yshift, width, height])
        
    image_labels = {}
    with open(labeldir) as f:
      for line in f:
        line = line.strip().split(',')
        if line[0]=='name': 
          continue # exclude the first line as header line

        leftx =  float(line[ MUCT_LABELS['left_idx_shift']*2+2 ])
        lefty =  float(line[ MUCT_LABELS['left_idx_shift']*2+2+1 ])
        rightx = float(line[ MUCT_LABELS['right_idx_shift']*2+2 ])
        righty = float(line[ MUCT_LABELS['right_idx_shift']*2+2+1 ])
        if leftx+lefty==0 or rightx+righty==0:
          print('this image has eye occupied: '+line[0])
          exit()
        image_labels[line[0]+'.jpg'] = {'left':[leftx,lefty], 'right':[rightx,righty]}

    for file in os.listdir(subdir):
        file_name = file
        label_list = image_labels[file]
        shift_list = image_crop_shift[file]
        label_list = apply_shift(label_list, shift_list)
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
