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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np 
import cv2
import argparse
import sys 
import keras 
import tensorflow as tf 
from keras.utils import plot_model
from imgaug import augmenters as iaa
sys.path.append("../test")
from evaluate import evaluate_coco
sys.path.append("../mrcnn")
from mask_rcnn_config import Config as CocoConfig
from mask_rcnn_model import MaskRCNN 
sys.path.append("../data")
from load_dataset import CocoDataset


def params():
    parser = argparse.ArgumentParser(description='Mask RCNN Train')
    parser.add_argument('--command',type=str,default='inference',\
            help='run training or inference, evalute')
    parser.add_argument('--model-path',dest='model_path',type=str,default='../../models',\
            help='load which model or save model path')
    parser.add_argument('--dataset-path',dest='dataset_path',type=str,\
            default='/home/lxy/Downloads/DataSet/COCO-2017', help='coco dataset path ')
    parser.add_argument('--coco-year',dest='coco_year',type=str,\
            default='2017',help='load coco dataset year')
    parser.add_argument('--download',type=bool,default=False,\
            help='if not exist download the data')
    parser.add_argument('--log-dir',dest='log_dir',type=str,\
            default='../../logs',help='save logs dir')
    parser.add_argument('--load-epoch',dest='load_epoch',type=str,\
            default=None,help='which model to load')
    parser.add_argument('--epochs',type=int,default=100000,\
            help='how much num to train')
    parser.add_argument('--learning-rate',dest='learning_rate',type=float,\
            default=0.2,help='trianing learn rate')
    parser.add_argument('--class-names',dest='class_names',type=str,\
            default='person',help='classes to load to train')
    parser.add_argument('--lr-patience',dest='lr_patience',type=int,\
            default=2,help='how many epochs to reduce lr')
    parser.add_argument('--gpu-list',dest='gpu_list',type=str,\
            default='0',help='how many gpu to run')
    parser.add_argument('--train-stage',dest='train_stage',type=str,\
            default='heads',help='trian stage: heads,3+,4+,5+,all')
    return parser.parse_args()


def display_model(model):
    plot_model(model,show_shapes=True,show_layer_names=True,to_file='mrcnn.png')

def set_gpu(gpu_list):
    #config = tf.ConfigProto( device_count = {'GPU': 0 } ) 
    #sess = tf.Session(config=config) 
    #keras.backend.set_session(sess)
    #if not isinstance(gpu_list,list):
     #   gpu_list = [gpu_list]
    #gpu_list = map(str,gpu_list)
    #gpu_num = ','.join(gpu_list)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

def train(args):
    '''
    train mrcnn models on coco dataset
    '''
    command = args.command
    year = args.coco_year
    model_path = args.model_path
    load_epoch = args.load_epoch
    log_dir = args.log_dir
    dataset_path = args.dataset_path
    download_fg = args.download
    class_names = args.class_names.split(',')
    LEARNING_RATE = args.learning_rate
    epoch_num = args.epochs
    lr_patience = args.lr_patience
    print("here")
    set_gpu(args.gpu_list)
    if command == "train":
        config = CocoConfig()
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.5
        config = InferenceConfig()
    # Create model
    if command == "train":
        print("build net")
        model = MaskRCNN(mode="training", config=config,
                                  model_dir=model_path,train_stage=args.train_stage)
    elif command == 'inference':
        model = MaskRCNN(mode="inference", config=config,
                                  model_dir=model_path,train_stage=args.train_stage)
    else:
        print("No commond")
    if config.displaymodel:
        display_model(model.keras_model)
    # Load weights
    if load_epoch is not None:
        print("Loading weights ", load_epoch)
        model.load_weights(load_epoch, by_name=True)
    # Train or evaluate
    if command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CocoDataset()
        print("begin to load data")
        dataset_train.load_coco(dataset_path, "train", year=year, auto_download=download_fg,class_names=config.ObjectNames)
        if year in '2014':
            dataset_train.load_coco(dataset_path, "valminusminival", year=year, auto_download=download_fg)
        dataset_train.prepare()
        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val" if year in '2017' else "minival"
        dataset_val.load_coco(dataset_path, val_type, year=year, auto_download=download_fg,class_names=config.ObjectNames)
        dataset_val.prepare()
        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = iaa.Fliplr(0.5)
        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        '''
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=LEARNING_RATE,
                    epochs=epoch_num,
                    patience = lr_patience,
                    augmentation=augmentation)
        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=LEARNING_RATE,
                    epochs=epoch_num,
                    patience = lr_patience,
                    augmentation=augmentation)
        '''
        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=LEARNING_RATE,
                    epochs=epoch_num,
                    patience = lr_patience,
                    augmentation=augmentation)

    elif command == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val" if year in '2017' else "minival"
        coco = dataset_val.load_coco(dataset_path, val_type, year=year, return_coco=True, auto_download=download_fg)
        dataset_val.prepare()
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(500))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(command))

if __name__ == '__main__':
    args = params()
    train(args)