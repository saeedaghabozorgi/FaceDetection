import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import face
import cv2

#Localization parameters
DET_SIZE = (300,300)    #Run all localization at a standard size
BLUR_DIM = (50,50)      #Dimension for blurring the face location mask
CONF_THRESH = 0.99      #Confidence threshold to mark a window as a face

X_STEP = 10     #Horizontal slide for the sliding window
Y_STEP = 10     #Vertical stride for the sliding window
WIN_MIN = 40    #Minimum sliding window size
WIN_MAX = 60   #Maximum sliding window size
WIN_STRIDE = 10   #Stride to increase the sliding window

cwd = os.getcwd()
path = os.path.join(cwd, 'export_model')
# Restoring
graph2 = tf.Graph()
with graph2.as_default():
    with tf.Session(graph=graph2) as sess:
        # Restore saved values
        print('\nRestoring...')
        tf.saved_model.loader.load(
            sess,
            [tag_constants.SERVING],
            path
        )
        print('Ok')
        # for n in tf.get_default_graph().as_graph_def().node:
        #     print(n.name)
        # Get restored placeholders
        labels_data_ph = graph2.get_tensor_by_name('labels_data_ph:0')
        features_data_ph = graph2.get_tensor_by_name('features_data_ph:0')
        print(features_data_ph.shape)
        batch_size_ph = graph2.get_tensor_by_name('batch_size_ph:0')
        # Get restored model output
        restored_logits = graph2.get_tensor_by_name('softmax_linear/softmax_linear:0')
        # Get dataset initializing operation
        dataset_init_op = graph2.get_operation_by_name('dataset_init')

        #dataset_init_op.
        #tf.global_variables_initializer().run(session=sess)
        # features, labels = face.distorted_inputs()
        # count = 1
        # # features= features[0:count,:,:,:].reshape(count,32,32,1)
        # labels = labels[0:count,:,:].reshape(count,1,1)
        # print(labels)
        # sess.run(  dataset_init_op,  feed_dict={ features_data_ph: features,labels_data_ph: labels, batch_size_ph: 1 })
        print("initializaiton ok")       
        # pred_value = sess.run(restored_logits, feed_dict={features_data_ph: features, batch_size_ph:1})
        # print(features.shape)
        # print(pred_value)
        # print("second")
        # sess.run(  dataset_init_op,  feed_dict={ features_data_ph: features,labels_data_ph: labels, batch_size_ph: 1 })
        # pred_value = sess.run(restored_logits, feed_dict={features_data_ph: features, batch_size_ph:1})
        # print(features.shape)
        # print(pred_value)
        # print ("finish test")
        # Initialize restored dataset
        # sess.run(
        #     dataset_init_op,
        #     feed_dict={
        #         features_data_ph: features,
        #         labels_data_ph: labels,
        #         batch_size_ph: 32
        #     }

        #)
        # Compute inference for both batches in dataset
        # restored_values = []
        # for i in range(3):
        #     restored_values.append(sess.run(restored_logits))
        #     print('Restored values: ', restored_values[i][0])
            #Run all detection at a fixed size
        
        img = cv2.imread("demo.jpg",0)
        img = cv2.resize(img,DET_SIZE)
        mask = np.zeros(img.shape)
        #Run sliding windows of different sizes
        print(WIN_MIN)
        print(WIN_MAX)
        for bx in range(WIN_MIN,WIN_MAX,WIN_STRIDE):
            by = bx
            print(by)
            for i in range(0, img.shape[1]-bx, X_STEP):
                for j in range(0, img.shape[0]-by, Y_STEP):
                    sub_img = cv2.resize(img[i:i+bx,j:j+by],face.IN_SIZE)
                    features = []
                    #print(sub_img.shape)
                    #X = sub_img.reshape((1,face.dim_prod(face.IN_SIZE)))
                    features.append(sub_img)
                    features = np.dstack(features)
                    features = np.rollaxis(features,-1)
                    features = np.reshape(features,(1,32,32,1))
                    #print(features.shape)
                    sess.run(  dataset_init_op,  feed_dict={ features_data_ph: features,labels_data_ph: np.array([[[1]]]), batch_size_ph: 1 })
                    pred_value = sess.run(tf.nn.softmax(restored_logits), feed_dict={features_data_ph: features, batch_size_ph:1})
                    #print(np.argmax(pred_value,1))
                    out = pred_value
                    if out[0][1] >= 0.8:
                        mask[i:i+bx,j:j+by] = mask[i:i+bx,j:j+by]+1

        sess.close()
        mask = np.uint8(255*mask/np.max(mask))
        faces_x = img*(cv2.threshold(cv2.blur(mask,BLUR_DIM),0,255,cv2.THRESH_OTSU)[1]/255)
        cv2.imshow("faces",faces_x)
        cv2.imshow("sliding window mask",mask)
        cv2.imshow("input image",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # return (faces_x,mask)
# Check if original inference and restored inference are equal
# valid = all((v == rv).all() for v, rv in zip(values, restored_values))
# print('\nInferences match: ', valid)