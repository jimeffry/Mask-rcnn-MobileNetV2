# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/11/22 16:09
#project: Face recognize
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  papers:
####################################################
import numpy as np 
import sys
import os 
import shutil
import cv2
import argparse

def parm():
    parser = argparse.ArgumentParser(description='mkdir')
    parser.add_argument('--img-dir',dest='img_dir',type=str,default='./',\
                        help='input images saved dir')
    parser.add_argument('--save-dir',dest='save_dir',type=str,default='./',
                        help='mkdir saved imgs')
    parser.add_argument('--command',type=str,default='dir',\
                        help='dir')
    return parser.parse_args()

def mdir_fromimg(base_dir,save_dir):
    '''
    base_dir: image paths saved
    return: make dirs according to the image name
    '''
    #file_p = open(file_in,'r')
    file_contents = os.listdir(base_dir)
    total_num = len(file_contents)
    idx_ = 0
    for file_tmp in file_contents:
        idx_+=1
        sys.stdout.write("\r>> deal with %d/%d" % (idx_,total_num))
        sys.stdout.flush()
        file_tmp = file_tmp.strip()
        org_path = os.path.join(base_dir,file_tmp)
        try:
            img = cv2.imread(org_path)
            if img is None:
                print('read failed:',org_path)
                continue
        except "read failed" :
            continue
        file_splits = file_tmp.split('_')
        dist_name = 'HC' +file_splits[0]
        dist_dir = os.path.join(save_dir,dist_name)
        if not os.path.exists(dist_dir):
            os.makedirs(dist_dir)
        save_name = file_splits[0] +'.jpg'
        save_path = os.path.join(dist_dir,save_name)
        shutil.copyfile(org_path,save_path)


if __name__ == '__main__':
    args = parm()
    base_dir = args.img_dir
    saved_dir = args.save_dir
    mdir_fromimg(base_dir,saved_dir)
