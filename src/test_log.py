import numpy as np 
import sys
import tensorflow as tf 

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
# set keras run on gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  #设置需要使用的GPU的编号
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4 #设置使用GPU容量占GPU总容量的比例
sess = tf.Session(config=config)
KTF.set_session(sess)

#check gpu avaliable
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
#check gpu avaliable
import tensorflow
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def set-anaconda_keras_backend():
    '''
    There is a file keras_activate.sh in the path "/anaconda2/envs/py2/etc/conda/activate.d".
    Then editing it, delete the content :
        "if [ "$(uname)" == "Darwin" ] 
        then
        # for Mac OSX
            export KERAS_BACKEND=tensorflow
        elif [ "$(uname)" == "Linux" ] 
        then
        # for Linux
            export KERAS_BACKEND=theano
        fi". 
    after that, add the line :
    set "KERAS_BACKEND=tensorflow"
    '''


def log_(in_data):
    return np.log(in_data)/np.log(2.0)

def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.log(x) / tf.log(2.0)

class test_tf(object):
    '''
    boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
    '''
    def __init__(self,boxes,feature_maps):
        self.boxes = boxes
        self.feature_maps = feature_maps
        self.pool_shape = tf.constant([7,7])

    def call(self):
        boxes = self.boxes
        feature_maps = self.feature_maps
        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        image_shape = [640,640]
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.cast(tf.sqrt(h*w),tf.float32)/ tf.cast(64.0 / tf.sqrt(image_area),tf.float32))
        #roi_level = log2_graph(tf.sqrt(h*w)/ 64.0 )
        roi_level = tf.minimum(5,tf.maximum(1,2+tf.cast(tf.round(roi_level),tf.int32)))
        level_botom = 1
        level_top = 6
        #print("in poolalign ",roi_level)
        roi_level = tf.squeeze(roi_level, 2)
        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        boxid = []
        boxes_ = []
        for i, level in enumerate(range(level_botom, level_top)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)
            box_indices = tf.cast(ix[:, 0], tf.int32)
            #print(i,level,box_indices_)
            #box_indices = tf.range(tf.shape(ix)[0])
            # Keep track of which box is mapped to which level
            box_to_level.append(ix)
            boxid.append(box_indices)
            boxes_.append(level_boxes)
            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))
        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],axis=1)
        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)
        # Re-add the batch dimension
        n_shape = tf.concat([tf.shape(boxes)[:2],tf.shape(pooled)[1:]],axis=0)
        pooled = tf.reshape(pooled,n_shape)
        return pooled,ix,sorting_tensor

def test():
    '''
    a = float(sys.argv[1])
    b = log_(a/64.0)
    print(b)
    '''
    graph = tf.Graph()
    with graph.as_default():
        '''
        p_s = tf.constant([7,7])
        box = tf.constant([[
            [10,10,42,42],[20,20,60,60],
            [50,50,110,110],[60,60,130,130],
            [100,100,230,230],[150,150,290,290],
            [200,200,450,450],[300,300,550,550],
            [30,30,550,550],[40,40,600,600]
        ]])
        print(tf.shape(box))
        box = tf.cast(box,tf.float32)
        fs1 = tf.ones([1,320,320,5],dtype=tf.float32)
        fs2 = tf.ones([1,160,160,5],dtype=tf.float32)
        fs3 = tf.ones([1,80,80,5],dtype=tf.float32)
        fs4 = tf.ones([1,40,40,5],dtype=tf.float32)
        fs5 = tf.ones([1,20,20,5],dtype=tf.float32)
        '''
        #p_s = [7,7]
        box = np.array([[
            [10,10,42,42],[20,20,60,60],
            [50,50,110,110],[60,60,130,130],
            [100,100,230,230],[150,150,290,290],
            [200,200,450,450],[300,300,550,550],
            [30,30,550,550],[40,40,600,600]
            #[60,60,120,120],[70,70,130,130]
        ]],dtype=np.float32)
        print('input box shape',box.shape)
        fs1 = np.ones([2,320,320,5],dtype=np.float32)
        fs2 = np.ones([2,160,160,5],dtype=np.float32)
        fs3 = np.ones([2,80,80,5],dtype=np.float32)
        fs4 = np.ones([2,40,40,5],dtype=np.float32)
        fs5 = np.ones([2,20,20,5],dtype=np.float32)
        ps = [fs1,fs2,fs3,fs4,fs5]
        box = np.float32(box / 639.0)
        boxes = np.zeros([2,10,4],dtype=np.float32)
        boxes[0] = box
        boxes[1] = box 
        c_t = test_tf(boxes,ps)
        p = c_t.call()
        sess = tf.Session()
        out_t = sess.run(p)
        print('out shape',np.shape(out_t[0]),out_t[1].shape,out_t[2].shape)
        print(np.array(out_t[1]))
        print(np.array(out_t[2]))

def test3():
    a = tf.constant([1,1])
    b = tf.constant([2,2])
    c = a+b 
    print(c)
    return c 
'''
if __name__ == '__main__':
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        t = test3()
        t2 = sess.run(t)
    print(t2)
'''
import cv2 

def findcontours():
    img = cv2.imread('../th.jpeg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    img_out,contours,herity = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print(np.shape(img))
    print(np.shape(contours))
    print(herity)
    print(np.shape(contours[1]))
    print(contours[1])
    #print(contours[])
    cv2.drawContours(img,contours[1],-1,(0,255,0),20)
    cv2.imshow("img", img)  
    cv2.waitKey(0)

if __name__ == '__main__':
    findcontours()