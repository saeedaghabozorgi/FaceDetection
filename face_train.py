
from datetime import datetime
import time
import cv2
import tensorflow as tf
import numpy as np
import face
import os
import shutil

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/face/face_train', """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,  """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,  """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10, """How often to log results to the console.""")
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
def train():
    graph1 = tf.Graph()
    with graph1.as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        features, labels = face.distorted_inputs()

        # Get images and labels.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            #Random seed for reproducibility
            tf.set_random_seed(0)
            # Placeholders
            batch_size_ph = tf.placeholder(tf.int64, name='batch_size_ph')
            features_data_ph = tf.placeholder(tf.float32, shape = [None, face.IN_SIZE[0],face.IN_SIZE[1],1] , name ='features_data_ph')
            labels_data_ph = tf.placeholder(tf.int32, shape = [None,1,1], name ='labels_data_ph')
            # Dataset
            dataset = tf.data.Dataset.from_tensor_slices((features_data_ph, labels_data_ph))
            dataset = dataset.shuffle(30000)
            #dataset = dataset.batch(batch_size_ph)
            dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size_ph))
            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
            dataset_init_op = iterator.make_initializer(dataset, name='dataset_init')
            image_tensor, labels_tensor = iterator.get_next()
            # dataset = face.distorted_inputs()
            # print(dataset.output_types)  # ==> "tf.float32"
            # print(dataset.output_shapes)  # ==> "(10,)"
            # dataset = dataset.repeat().apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
            # iterator = dataset.make_one_shot_iterator()
            # next_images, next_labels = iterator.get_next()
            labels_tensor = tf.reshape(labels_tensor,[-1])
            # with tf.Session() as sess:
            #     for i in range(5):
            #         value = sess.run(next_element)
            #         print(value[0].shape)
            print('-------------')
            print('image shape:')
            print(image_tensor.shape)
            print('label shape:')
            print(labels_tensor.shape)
            print('-------------')
            # Build a Graph that computes the logits predictions from the
            # inference model.
            logits = face.inference(image_tensor)
            print("about logit")
            print(logits.dtype)
            print(logits.shape)
            print(logits)
            # Calculate loss.
            
            loss = face.loss(logits, labels_tensor)
            #Create a saver object which will save all the variables
            #saver = tf.train.Saver()
            # # Build a Graph that trains the model with one batch of examples and
            # # updates the model parameters.
            train_op = face.train(loss, global_step)
            with tf.Session(graph=graph1) as sess:
                # Initialize variables
                tf.global_variables_initializer().run(session=sess)
                for epoch in range(1):
                    batch = 0
                    # Initialize dataset (could feed epochs in Dataset.repeat(epochs))
                    sess.run(
                        dataset_init_op,
                        feed_dict={
                            features_data_ph: features,
                            labels_data_ph: labels,
                            batch_size_ph: FLAGS.batch_size
                        })
                    values = []
                    while True:
                        try:
                            if epoch < 2:
                                # Training
                                #sample_label = sess.run([labels_tensor])
                                #print(sample_label)
                                
                                _, loss_value, sample_label, pred_value = sess.run([train_op, loss, labels_tensor, tf.nn.softmax(logits)])
                                # print(np.argmax(pred_value,1))
                                # print(sample_label)
                                
                                accuracy = np.mean(sample_label==np.argmax(pred_value,1))
                                print('Epoch {}, batch {}, loss {:.2f}, acc {:.2f} | Sample: {}, Pred value: {}'.format(epoch,  batch, loss_value, accuracy, sample_label[0], pred_value[0]))
                                
                                batch += 1
                            else:
                                # Final inference
                                values.append(sess.run(tf.nn.softmax(logits)))
                                print('Epoch {}, batch {} | Final inference | Sample value: {}'.format(epoch, batch, values[-1][0]))
                                batch += 1
                        except tf.errors.OutOfRangeError:
                            break
                # Save model state
                print('\nSaving...')
                cwd = os.getcwd()
                path = os.path.join(cwd, 'export_model')
                shutil.rmtree(path, ignore_errors=True)
                # Saving
                
                inputs = {
                    "batch_size_placeholder": batch_size_ph,
                    "features_placeholder": features_data_ph,
                    "labels_placeholder": labels_data_ph
                }
                outputs = {"logits": logits}
                tf.saved_model.simple_save( sess, path, inputs, outputs)

            print('Ok')                   



def main(argv=None):  # pylint: disable=unused-argument
    # TO DO: face.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    # FACE_PATH = "Faces/positive/faces/Aaron_Guiel_0001.pgm"
    # t_img = cv2.resize(cv2.imread(FACE_PATH,0),(32,32))
    # tensor_img = tf.convert_to_tensor(t_img, np.float32)

    # #print(type(t_img))
    # dataset = tf.data.Dataset.from_tensor_slices(tensor_img) 
    # print(dataset.output_types)  # ==> "tf.float32"
    # print(dataset.output_shapes)  # ==> "(10,)"    

    train()


if __name__ == '__main__':
  tf.app.run()