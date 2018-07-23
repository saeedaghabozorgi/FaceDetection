
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


        # get data
        print('-------------')
        print("reading data ....")
        features, labels = face.distorted_inputs()


        # split dataset in train and test
        print('-------------')
        print("spliting data ....")
        data_size = features.shape[0]
        indices = np.random.permutation(data_size)
        training_idx, test_idx = indices[:int(data_size*0.8)], indices[int(data_size*0.8):]
        print(training_idx)
        features_train_set, features_test_set = features[training_idx,:], features[test_idx,:]
        labels_train_set, labels_test_set = labels[training_idx], labels[test_idx]
        print(features_train_set.shape)

        
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            #Random seed for reproducibility
            tf.set_random_seed(0)
            # Placeholders
            batch_size_ph = tf.placeholder(tf.int64, name='batch_size_ph')
            features_data_ph = tf.placeholder(tf.float32, shape = [None, face.IN_SIZE[0],face.IN_SIZE[1],1] , name ='features_data_ph')
            labels_data_ph = tf.placeholder(tf.int32, shape = [None,1], name ='labels_data_ph')
            
            # Dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((features_data_ph, labels_data_ph)).shuffle(30000).batch(batch_size_ph)
            test_dataset = tf.data.Dataset.from_tensor_slices((features_data_ph, labels_data_ph)).shuffle(30000).batch(batch_size_ph)
             # To create an Iterator that will extract data from this dataset. This type of initializers, can be re-initialized
            iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
            # create the initialisation operations
            train_dataset_init_op = iterator.make_initializer(train_dataset, name='train_dataset_init')
            test_dataset_init_op = iterator.make_initializer(test_dataset, name='test_dataset_init')
            
            
            print('-------------')
            print('image shape:')
            image_tensor, labels_tensor = iterator.get_next()
            print(image_tensor.shape)
            print('label shape:')
            labels_tensor = tf.reshape(labels_tensor,[-1])
            print(labels_tensor.shape)
            print('-------------')
            # Build a Graph that computes the logits predictions from the inference model.
            logits = face.inference(image_tensor)
            print("Building logit")
            print(logits.dtype)
            print(logits.shape)
            print(logits)

            # # Calculate loss.       
            loss = face.loss(logits, labels_tensor)


            # # Build a Graph that trains the model with one batch of examples and updates the model parameters.
            train_op = face.train(loss, global_step)

            with tf.Session(graph=graph1) as sess:
                # Initialize variables
                tf.global_variables_initializer().run(session=sess)
                for epoch in range(5):
                    batch = 0
                    train_pred_values = []
                    # Initialize training dataset (could feed epochs in Dataset.repeat(epochs))
                    sess.run(
                        train_dataset_init_op,
                        feed_dict={
                            features_data_ph: features_train_set,
                            labels_data_ph: labels_train_set,
                            batch_size_ph: FLAGS.batch_size
                        })
                    
                    while True:
                        try:
                            # Training
                            _, loss_value, sample_label, pred_value = sess.run([train_op, loss, labels_tensor, tf.nn.softmax(logits)])
                            train_pred_values.append(pred_value)
                            accuracy = np.mean(sample_label==np.argmax(pred_value,1))
                            print('Epoch {}, batch {}, loss {:.2f}, acc {:.2f} | Sample: {}, Pred value: {}'.format(epoch,  batch, loss_value, accuracy, sample_label[0], pred_value[0]))
                            batch += 1
                        except tf.errors.OutOfRangeError:
                            break    
                    
                    # calculate Train accuracy
                    # print("train accuracy evaluatio ...")
                    # train_pred_values = np.vstack(train_pred_values)
                    # y = labels_test_set.reshape(1,-1)[0]
                    # print(y)
                    # accuracy = np.mean(y==np.argmax(valutrain_pred_valueses,1) )
                    # print(accuracy)


                    # Test accuracy
                    # Initialize test dataset (could feed epochs in Dataset.repeat(epochs))
                    print("calculating accuracy on test set ...")
                    sess.run(
                        test_dataset_init_op,
                        feed_dict={
                            features_data_ph: features_test_set,
                            labels_data_ph: labels_test_set,
                            batch_size_ph: FLAGS.batch_size
                        })
                    test_pred_values = []
                    test_sample_labels = []
                    batch = 0
                    while True:
                        try:
                            # Testing
                            sample_label, pred_value = sess.run([labels_tensor, tf.nn.softmax(logits)])
                            test_pred_values.append(pred_value)
                            test_sample_labels.append(sample_label)
                            print('Epoch {}, batch {} | Final inference | Sample value: {}'.format(epoch, batch, test_pred_values[-1][0]))
                            batch += 1
                        except tf.errors.OutOfRangeError:
                            break
                    


                    

                    print("evaluatio ...")
                    test_pred_values = np.vstack(test_pred_values)
                    y_hat = np.argmax(test_pred_values,1) 
                    print(y_hat)
                    y = np.hstack(test_sample_labels)
                    print(y)
                    accuracy = np.mean(y==y_hat)
                    print(accuracy)
 
                
                
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