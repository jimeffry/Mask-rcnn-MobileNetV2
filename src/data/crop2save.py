# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/10/12 16:09
#project: Face recognize
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  papers:
####################################################
import numpy as np 
import sys
import os
import cv2
from load_coco import CocoDataset
import argparse
from tqdm import tqdm

BoxWiden =1

def params():
    parser = argparse.ArgumentParser(description='coco data load')
    parser.add_argument('--dataset-path',dest='dataset_path',type=str,\
            default='/home/lxy/Downloads/DataSet/COCO-2017', help='coco dataset path ')
    parser.add_argument('--coco-year',dest='coco_year',type=str,\
            default='2017',help='load coco dataset year')
    parser.add_argument('--save-dir',dest='save_dir',type=str,\
            default='/home/lxy/Downloads/DataSet/COCO_Crop',help='save  dir')
    parser.add_argument('--load-name',dest='load_name',type=str,\
            default='train',help='train or test')
    parser.add_argument('--objectnames',dest='objectnames',type=list,\
            default=['person'],help='class names to load')
    parser.add_argument('--area-low',dest='area_low',type=int,default=1e4,\
            help='filter area low value')
    parser.add_argument('--area-high',dest='area_high',type=int,\
            default=1e6,help='filter area high value')
    return parser.parse_args()

def mk_dirs(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

def img_crops(img,bboxes):
    imgh,imgw = img.shape[:2]
    crop_out = []
    for bbox in bboxes:
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        if BoxWiden:
            boxw = x2-x1
            boxh = y2-y1
            x1 = int(max(0,int(x1-0.3*boxw)))
            y1 = int(max(0,int(y1-0.3*boxh)))
            x2 = int(min(imgw,int(x2+0.3*boxw)))
            y2 = int(min(imgh,int(y2+0.3*boxh)))
        cropimg = img[y1:y2,x1:x2,:]
        crop_out.append(cropimg)
    return crop_out

def img_crop(img,bbox):
    imgh,imgw = img.shape[:2]
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    if BoxWiden:
        boxw = x2-x1
        boxh = y2-y1
        x1 = int(x1-0.3*boxw)
        y1 = int(y1-0.3*boxh)
        x2 = int(x2+0.3*boxw)
        y2 = int(y2+0.3*boxh)
    x1 = int(max(0,x1))
    y1 = int(max(0,y1))
    x2 = int(min(imgw,x2))
    y2 = int(min(imgh,y2))
    cropimg = img[y1:y2,x1:x2,:]
    return cropimg

def label_show(img,rectangles,names):
    for i, rectangle in enumerate(rectangles):
        score_label = names[i]
        cv2.putText(img,score_label,(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        cv2.rectangle(img,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
    cv2.imshow('load',img)
    cv2.waitKey(0)

def main(args):
    year = args.coco_year
    dataset_path = args.dataset_path
    ObjectNames = args.objectnames
    save_dir = args.save_dir
    load_name = args.load_name
    area_range = [args.area_low,args.area_high]
    assert load_name in ['train','val','minival','valminusminival']
    mk_dirs(save_dir)
    train_dir = os.path.join(save_dir,load_name)
    mk_dirs(train_dir)
    #keep input name dirs
    train_dir_dict = dict()
    #load coco dataset
    print("begin to load data")
    dataset_train = CocoDataset()
    dataset_train.load_coco(dataset_path, load_name, year=year,class_names=ObjectNames,areaRng=area_range)
    #make dir for names
    print("valid names",dataset_train.ValidNames)
    for tmp_name in ObjectNames:
        if tmp_name in dataset_train.ValidNames:
            train_name_dir = os.path.join(train_dir,tmp_name)
            mk_dirs(train_name_dir)
            train_dir_dict[tmp_name] = train_name_dir
        else:
            print('%s is not in cocodataset' % tmp_name)
    #load image data and bbox
    train_img_ids = dataset_train.img_ids
    #statistics imgs for every class
    saveimg_dict = dict()
    print("total imgs",len(train_img_ids))
    for tmp_idx in tqdm(range(len(train_img_ids))):
        tmp_bboxes,tmp_train_id = dataset_train.load_bbox(tmp_idx)
        if tmp_bboxes is None:
            continue
        name_list = [dataset_train.trainId2Names[j] for j in tmp_train_id]
        img = dataset_train.load_image(tmp_idx)
        #label_show(img,tmp_bboxes,name_list)
        for i,bbox in enumerate(tmp_bboxes):
            tmp_crop = img_crop(img,bbox)
            key_name = name_list[i]
            tmp_cnt = saveimg_dict.setdefault(key_name,0)
            img_path = os.path.join(train_dir_dict[key_name],key_name+'_'+load_name+'_'+str(tmp_cnt)+'.jpg')
            saveimg_dict[key_name] = tmp_cnt+1
            cv2.imwrite(img_path,tmp_crop)


if __name__=='__main__':
    args = params()
    args.objectnames = ['tv','laptop','cell phone','remote']
    if args.load_name=='val':
        args.load_name = "val" if args.coco_year in '2017' else "minival"
    main(args)
