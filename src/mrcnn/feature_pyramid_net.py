# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/10/16 10:09
#project: Face recognize
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  papers:
####################################################
from keras import backend as K
from keras import layers as KL
from keras.models import Model
from keras.utils import plot_model
from mask_rcnn_config import Config as config

def fpn_block(P_top,P_bottom,name):
    P_up = KL.UpSampling2D(size=(2, 2), name="fpn_upsampled_%d" % name)(P_top)
    P_b = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c%d' % name)(P_bottom)
    P_add = KL.Add(name="fpn_add_%d" % name)([P_up,P_b])
    P_out = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p%d" % name)(P_add)
    return P_add,P_out

def get_fpn(feature_map_list):
    '''
    func: gnerate feature pyramid maps
    feature_map_list: conlution features maps, form bottom to down, [C2, C3, C4, C5]
    '''
    if not isinstance(feature_map_list,list):
        feature_map_list = [feature_map_list]
    feature_map_list = feature_map_list[::-1]
    top_num = len(feature_map_list)
    fpn_out = []
    for i, p in enumerate(feature_map_list):
        name = int(top_num-i)
        if i == 0:
            f_add = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c%d' % name)(p)
            f_out = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p%d" % name)(f_add)
            fpn_out.append(f_out)
        else:
            f_add,f_out = fpn_block(f_add,p,name)
            fpn_out.append(f_out)
    return fpn_out

def get_fpn_old(feature_map_list,config):
    if not len(feature_map_list):
        print("raise error: no feature maps")
        return 0
    #for i, p in enumerate(feature_map_list):
    C2, C3, C4, C5 = feature_map_list
    P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
    P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
    P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
    P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
    return [P2,P3,P4,P5]

def vis_net(model):
    plot_model(model,show_shapes=True,show_layer_names=True,to_file='model_fpn2.png')

if __name__ == '__main__':
    c5 = K.ones(shape=[1,32,32,1024])
    c4 = K.ones(shape=[1,64,64,1024])
    c3 = K.ones(shape=[1,128,128,512])
    c2 = K.ones(shape=[1,256,256,256])
    C2 = KL.Input(tensor=c2)
    C3 = KL.Input(tensor=c3)
    C4 = KL.Input(tensor=c4)
    C5 = KL.Input(tensor=c5)
    f=[C2,C3,C4,C5]
    #m = get_fpn_old(f,config)
    m = get_fpn(f)
    net = Model(input=f,output=m)
    vis_net(net)