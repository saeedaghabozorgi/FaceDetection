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

X_STEP = 5     #Horizontal slide for the sliding window
Y_STEP = 5     #Vertical stride for the sliding window
WIN_MIN = 50    #Minimum sliding window size
WIN_MAX = 80   #Maximum sliding window size
WIN_STRIDE = 10   #Stride to increase the sliding window

cwd = os.getcwd()
path = os.path.join(cwd, 'export_model')


def localize(img):
    features = []
    for bx in range(WIN_MIN,WIN_MAX,WIN_STRIDE):
        print(bx)
        by = bx
        print("window size:" + str(by))
        for i in range(0, img.shape[1]-bx, X_STEP):
            for j in range(0, img.shape[0]-by, Y_STEP):
                sub_img = cv2.resize(img[i:i+bx,j:j+by],face.IN_SIZE)
                features.append(np.array(sub_img))
    features = np.array(features)
    features = np.expand_dims(features, axis=3)
    print(features.shape)
    return features


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
        dataset_init_op = graph2.get_operation_by_name('train_dataset_init')

        print("initializaiton ok")       

        
        img = cv2.imread("demo3.jpg",0)
        img = cv2.resize(img,DET_SIZE)
        mask = np.zeros(img.shape)
        #Run sliding windows of different sizes
        # print(WIN_MIN)
        # print(WIN_MAX)
        features = localize(img)
        batch_size = features.shape[0]
        sess.run(dataset_init_op,  feed_dict={ features_data_ph: features,labels_data_ph: np.ones(batch_size).reshape(batch_size,1), batch_size_ph: batch_size })
        pred_value = sess.run(tf.nn.softmax(restored_logits), feed_dict={features_data_ph: features, batch_size_ph:batch_size})
        out = np.argmax(pred_value,1)
        print(out)

        # --- test ---
        #cv2.imshow("input image 1",img)
        for i in range(len(pred_value)):
            if pred_value[i][1]>0.7:
                print (i)
                print(pred_value[i])
                sam_image = features[i].reshape((32,32))
                cv2.imshow("input image 2",sam_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                

        idx = 0
        for bx in range(WIN_MIN,WIN_MAX,WIN_STRIDE):
            by = bx
            print("window size:" + str(by))
            for i in range(0, img.shape[1]-bx, X_STEP):
                for j in range(0, img.shape[0]-by, Y_STEP):
                    if pred_value[idx][1] > 0.5:
                        print(idx)
                        mask[i:i+bx,j:j+by] = mask[i:i+bx,j:j+by]+1
                    idx+=1

        sess.close()
        mask = np.uint8(255*mask/np.max(mask))
        faces_x = img*(cv2.threshold(cv2.blur(mask,BLUR_DIM),0,255,cv2.THRESH_OTSU)[1]/255)
        cv2.imshow("faces",faces_x)
        cv2.imshow("sliding window mask",mask)
        cv2.imshow("input image",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



# Check if original inference and restored inference are equal
# valid = all((v == rv).all() for v, rv in zip(values, restored_values))
# print('\nInferences match: ', valid)

