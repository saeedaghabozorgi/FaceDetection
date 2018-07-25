import os
import cv2
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
FLAGS = tf.app.flags.FLAGS

# Basic model parameters.

tf.app.flags.DEFINE_string('data_dir', '/tmp/SETI_data', """Path to the SETI data directory.""")
tf.app.flags.DEFINE_string('FACE_PATH', 'Faces/positive/', """Path to the SETI data directory.""")
tf.app.flags.DEFINE_string('NON_FACE_PATH', 'Faces/negative/', """Path to the SETI data directory.""")



# Process images of this size. 
IMAGE_SIZE = 32
IN_SIZE = (IMAGE_SIZE,IMAGE_SIZE)   #Input dimensions of image for the network

# Global constants describing the Face data set.
NUM_CLASSES = 2
#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 300
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 90

      
def read_face2():
    features = []
    labels = []
    
    for dir_name in os.listdir(FLAGS.FACE_PATH):
        dname = FLAGS.FACE_PATH+dir_name
        if os.path.isdir(dname):
          print(dname+"/")
          for img_path in os.listdir(dname):
              #print(dname+"/"+img_path)
              t_img = cv2.resize(cv2.imread(dname+"/"+img_path,0),IN_SIZE)
              features.append(t_img)
              labels.append(1)
    print(len(features))
    for dir_name in os.listdir(FLAGS.NON_FACE_PATH):
        dname = FLAGS.NON_FACE_PATH+dir_name
        if os.path.isdir(dname):
          print(dname+"/")
          for img_path in os.listdir(dname):
              #print(dname+"/"+img_path)
              t_img = cv2.resize(cv2.imread(dname+"/"+img_path,0),IN_SIZE)
              features.append(t_img)
              labels.append(0)
    
    #shuffling


    print(len(features))
    features = np.array(features)
    features = np.expand_dims(features, axis=3)
    print(features.shape)
    labels = np.array(labels)
    labels = np.expand_dims(labels, axis=1)
    print(labels.shape)
    return features,labels
    #return features, labels


def read_face(filename_queue):
  """Reads and parses examples from SETI data files.
  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.
  Args:
    filename_queue: A queue of strings with the filenames to read from.
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  
  class faceRecord(object):
    pass
  result = faceRecord()

  # Dimensions of the images in the SETI dataset.
  label_bytes = 1  
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the SETI format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs2(data_dir, batch_size):
  filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
               for i in xrange(1, 5)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)


def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for SETI training using the Reader ops.
  Args:
    data_dir: Path to the SETI data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
               for i in xrange(1, 5)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  # string_input_producer creates a FIFO queue for holding the filenames until the reader needs them.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_SETI(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly crop a [height, width] section of the image.
  distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  # NOTE: since per_image_standardization zeros the mean and makes
  # the stddev unit, this likely has no effect see tensorflow#1458.
  distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(distorted_image)

  # Set the shapes of tensors.
  float_image.set_shape([height, width, 3])
  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d SETI images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)
