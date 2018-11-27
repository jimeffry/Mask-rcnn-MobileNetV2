# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/11/06 10:09
#project: Face recognize
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  histogram
####################################################
import numpy as np 
from matplotlib import pyplot as plt 
import csv 
import json
import string
import argparse
import shutil
from tqdm import tqdm
import os


def params():
    parser = argparse.ArgumentParser(description='histgram')
    parser.add_argument('--path1',type=str,default='/home/lxy/Downloads/result.csv',\
            help='load csv file')
    parser.add_argument('--path2',type=str,default="/home/lxy/Downloads/distance_top2.csv",\
            help='load top2 csv')
    parser.add_argument('--path3',type=str,default="/home/lxy/Downloads/distance_top1.csv",\
             help='load top1 csv')
    parser.add_argument('--key-name',dest='key_name',type=str,\
            default='blur',help='load csv data')
    parser.add_argument('--threshold',dest='threshold_value',type=float,\
            default=250.0,help='threshold value of l1_regulars')
    parser.add_argument('--save-dir',dest='save_dir',type=str,\
            default='./save_imgs/',help='paths of images saved')
    parser.add_argument('--base-dir',dest='base_dir',type=str,\
            default='./base_imgs/',help='paths of images saved')
    parser.add_argument('--out-file',dest='out_file',type=str,\
            default='record_out.csv',help='paths to save')
    parser.add_argument('--command',type=str,default='static3',help='run fun: static2 static3 copyfile compare')
    return parser.parse_args()


def read_data(f_path,dict_keys):
    '''
    dataformat: filename, blur , left_right, up_down, rotate, cast, da, bbox_face.width, bbox_face.height , distance
    return: data_dict
    '''
    list_out = []
    for name in dict_keys:
        list_out.append([])
    data_dict = dict(zip(dict_keys,list_out))
    f_in = open(f_path,'rb')
    reader = csv.DictReader(f_in)
    for f_item in reader:
        #print(f_item['filename'])
        for cur_key in dict_keys:
            if cur_key == 'filename':
                data_dict[cur_key].append(f_item[cur_key])
            else:
                data_dict[cur_key].append(string.atof(f_item[cur_key]))
    f_in.close()
    return data_dict

def write_data(out_path,data_dict):
    '''
    out_path: saved csv file path
    data_dict: out data dic()
    '''
    csv_file = open(out_path,'w')
    fieldnames = data_dict.keys()
    print('keys',fieldnames)
    key1 = fieldnames[0]
    data1 = data_dict[key1]
    key2 = fieldnames[1]
    data2 = data_dict[key2]
    dict_list = [{key1:tp1,key2:tp2} for tp1,tp2 in zip(data1,data2)]
    writer = csv.DictWriter(csv_file,fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(dict_list)
    csv_file.close()

def statistic_data(datas,data_2,data_3,key_name):
    '''
    datas: input dict ,include 10 keys
    data_2: input dict 2 include 2keys: filename, confidence
    cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]

    '''
    ay_data = data_2['confidence']
    #ax_data = datas[dict_keys[-1]]
    ax_data = data_3["distance"]
    #ay_data = data_2['confidence2']
    #print('reg_fg',fg_data[:10])
    #print('ay',ay_data[:10])
    ay_data = np.array([round(tp,3) for tp in ay_data[:]])
    #test_data = datas[key_name]
    test_data = data_2[key_name]
    #
    color_data = test_data
    color_data = np.array([round(tp,2) for tp in color_data[:]])
    ax_data = np.array([round(tp,2) for tp in ax_data[:]])
    #print("ax",ax_data[:10])
    #get min-max
    plot_continue_color(ax_data,ay_data,color_data,key_name)
    
def plot_continue_color(ax_data,ay_data,color_data,key_name):
    '''
    color_data: continue data
    '''
    min_c = np.min(color_data)
    max_c = np.max(color_data)
    min_txt = "color_min:"+str(min_c)
    max_txt = "color_max:"+str(max_c)
    #get the unique number
    col_set = list(set(color_data))
    ay_set = list(set(ay_data))
    ax_set = list(set(ax_data))
    total_len = len(color_data)
    col_len = len(col_set)
    ay_len = len(ay_set)
    ax_len = len(ax_set)
    set_txt = "unique_color:" + str(col_len)
    total_txt = "total_color:"+str(total_len)
    ax_txt = "unique_ax:"+str(ax_len)
    ay_txt = "unique_ay:"+str(ay_len)
    #sorft data
    indx_ax = np.argsort(ax_data)
    #sorft
    ax_data = ax_data[indx_ax[:]]
    ay_data = ay_data[indx_ax[:]]
    print("ax_f",ax_data[:10])
    print("ay_f",ay_data[:10])
    color_data = color_data[indx_ax[:]]
    print("color",color_data[:10])
    fig = plt.figure(num=0,figsize=(20,10))
    ax1 = fig.add_subplot(111)
    #plt.scatter(ax_data,ay_data,alpha=0.6,c=color_data,cmap=plt.get_cmap('Paired'))
    plt.scatter(ax_data,ay_data,alpha=0.5,c=color_data,cmap=plt.get_cmap('viridis'))
    plt.colorbar()
    plt.xlabel('distance')
    plt.ylabel('confidence')
    plt.title('distance Vs confidence')
    ax_show1 = int(ax_len/2)
    ax_x1 = ax_set[ax_show1]
    ax_xend = ax_set[-5]
    ay_end = ay_set[-1]
    print('ay',ay_end)
    plt.text(ax_x1,ay_end,min_txt)
    plt.text(ax_x1,ay_end-0.02,max_txt)
    plt.text(ax_xend,ay_end,"color:%s" % key_name)
    plt.text(ax_x1,ay_end-0.05,total_txt)
    plt.text(ax_x1,ay_end-0.08,set_txt)
    plt.text(ax_x1,ay_end-0.09,ay_txt)
    plt.text(ax_x1,ay_end-0.11,ax_txt)
    #plt.ylim(0.0,0.5)
    #plt.setp(ax1.get_xticklabels(), fontsize=6)
    # share x only
    plt.savefig("./output/%s.png" % key_name,format='png')
    plt.show()
    
def test():
    a = [1,2,3,3,4,5]
    b = [0.3,0.7,0.3,0.4,0.3,0.5]
    fig = plt.figure(num=0,figsize=(10,10))
    ax = fig.add_subplot(111)
    plt.scatter(a,b,alpha=0.5)
    plt.show()

def plot3_colors(ax_data,ay_data,color_data,key_name):
    '''
    ax_data: plot ax
    ay_data: plot ay
    color_data: display color for point
    '''
    min_x = np.min(ax_data)
    max_x = np.max(ax_data)
    min_txt = "ax_min:"+str(min_x)
    max_txt = "ax_max:"+str(max_x)
    #get the unique number
    ax_set = list(set(ax_data))
    ay_set = list(set(ay_data))
    ax_set.sort()
    total_len = len(ax_data)
    ax_len = len(ax_set)
    ay_len = len(ay_set)
    set_txt = "unique_ax:" + str(ax_len)
    total_txt = "total_ax:"+str(total_len)
    ay_txt = "unique_ay:"+str(ay_len)
    #sorft data
    indx_ax = np.argsort(ax_data)
    #sorft
    ax_data = ax_data[indx_ax[:]]
    ay_data = ay_data[indx_ax[:]]
    print("ax_f",ax_data[:10])
    print("ay_f",ay_data[:10])
    color_data = color_data[indx_ax[:]]
    print("color",color_data[:10])
    fig = plt.figure(num=0,figsize=(20,10))
    ax1 = fig.add_subplot(111)
    plt.scatter(ax_data,ay_data,alpha=0.6,c=color_data,cmap=plt.get_cmap('Paired'))
    #plt.scatter(ax_data,ay_data,alpha=0.6,c=color_data,cmap=plt.get_cmap('viridis'))
    plt.colorbar()
    plt.xlabel('%s' % key_name)
    plt.ylabel('confidence')
    plt.title('%s Vs confidence' % key_name)
    ax_show1 = int(ax_len/2)
    ax_x1 = ax_set[ax_show1]
    ax_xend = ax_set[-20]
    ay_end = ay_set[-2]
    plt.text(ax_x1,ay_end,min_txt)
    plt.text(ax_x1,ay_end-0.02,max_txt)
    plt.text(ax_xend,ay_end,"color: tpr-%d fpr-%d no-reg-%d" % (20,60,1))
    plt.text(ax_x1,ay_end-0.05,total_txt)
    plt.text(ax_x1,ay_end-0.07,set_txt)
    plt.text(ax_x1,ay_end-0.09,ay_txt)
    #plt.ylim(0.0,0.5)
    #plt.setp(ax1.get_xticklabels(), fontsize=6)
    # share x only
    plt.savefig("./output/%s.png" % key_name,format='png')
    plt.show()

def statistic_fpr(datas,data_2,key_name):
    '''
    datas: input dict ,include 10 keys
    data_2: input dict 2 include 2keys: filename, confidence
    cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]

    '''
    ay_data = data_2['confidence']
    #ay_data = datas['distance']
    #ax_data = datas[key_name]
    ax_data = data_2[key_name]
    fg_data = data_2['reg_fg']
    fg_data = np.array([round(tp,1) for tp in fg_data[:]])
    #print('reg_fg',fg_data[:10])
    #print('ay',ay_data[:10])
    ay_data = np.array([round(tp,3) for tp in ay_data[:]])
    color_data = np.ones(np.shape(fg_data))
    #color_data *=5
    tpr_indx = np.where(fg_data==1.0)
    fpr_indx = np.where(fg_data== -1.0)
    print("tpr,fpr",np.shape(tpr_indx),np.shape(fpr_indx))
    color_data[tpr_indx] = 20
    color_data[fpr_indx] = 60
    #print('color_fpr',color_data[fpr_indx])
    #
    ax_data = np.array([round(tp,2) for tp in ax_data[:]])
    #print("ax",ax_data[:10])
    #get min-max
    plot3_colors(ax_data,ay_data,color_data,key_name)

def get_path_by_l1(data_dict,threshold):
    '''
    data_dict: keys: filename,l1_regular,confidence,reg_fg
    threshold: select output according thresholdvalue of l1_regular
    return: out_dict
    '''
    out_dict = dict()
    img_paths = list(data_dict['filename'])
    l1_dis = list(data_dict['l1_regular'])
    confidences = list(data_dict['confidence'])
    reg_fg = list(data_dict['reg_fg'])
    l1_dis = np.array([np.round(tp,3) for tp in l1_dis])
    idx = np.where(l1_dis < threshold)
    print('idx shape',np.shape(idx),idx[0][2])
    print("path",img_paths[2])
    out_dict['img_paths'] = [img_paths[i] for i in idx[0]]
    out_dict['l1_regular'] = [l1_dis[i] for i in idx[0]]
    out_dict['confidence'] = [confidences[i] for i in idx[0]]
    out_dict['reg_fg'] = [reg_fg[i] for i in idx[0]]
    return out_dict

def copy_save_imgs(data_dict,base_dir,save_dir):
    '''
    data_dict: keys: filename,l1_regular,confidence,reg_fg
    save_dir: images saved path
    '''
    img_paths = data_dict['img_paths']
    l1_dis = data_dict['l1_regular']
    confidences = data_dict['confidence']
    reg_fgs = data_dict['reg_fg']
    assert np.shape(img_paths) == np.shape(l1_dis)==np.shape(confidences)==np.shape(reg_fgs),'input shape is not same'
    if not os.path.exists(save_dir):
        print("the %s is not exist" % save_dir)
    else:
        tpr_idx = 0
        fpr_idx = 0
        noreg_idx = 0
        tpr_dir = os.path.join(save_dir,'tpr')
        fpr_dir = os.path.join(save_dir,'fpr')
        noreg_dir = os.path.join(save_dir,'noreg')
        if not os.path.exists(tpr_dir):
            os.makedirs(tpr_dir)
        if not os.path.exists(fpr_dir):
            os.makedirs(fpr_dir)
        if not os.path.exists(noreg_dir):
            os.makedirs(noreg_dir)
        for i in tqdm(range(len(img_paths))):
            img_splits = img_paths[i].strip().split('/')
            org_name = img_splits[-2]+'/'+img_splits[-1]
            org_path = os.path.join(base_dir,org_name)
            img_name = 'img_'+'%.3f' % l1_dis[i]+'_'+'%.3f' % confidences[i]
            if int(reg_fgs[i]) == 1:
                tpr_idx +=1
                img_name = img_name +'_'+'%d' % tpr_idx
                save_path = os.path.join(tpr_dir,img_name)
            elif int(reg_fgs[i])==-1:
                fpr_idx +=1
                img_name = img_name +'_'+'%d' % fpr_idx
                save_path = os.path.join(fpr_dir,img_name)
            elif int(reg_fgs[i])==0:
                noreg_idx +=1
                img_name = img_name +'_'+'%d' % noreg_idx
                save_path = os.path.join(noreg_dir,img_name)
            shutil.copyfile(org_path,save_path)

def statics_top3data(data1,data2,out_file):
    '''
    data1: top1-top2
    data2: top2-top3
    '''
    dis1 = data1['confidence']
    #dis2 = data2['confidence']
    dis2 = data1['confidence2']
    reg_fg = data1['reg_fg']
    color = data1['l1_regular']
    out_dict = dict()
    #convert to array
    color_data = np.array([round(tp,3) for tp in color])
    reg_data = np.array([int(tp) for tp in reg_fg])
    dis1_data = np.array([round(tp,3) for tp in dis1])
    dis2_data = np.array([round(tp,3) for tp in dis2])
    #select top2-top1 >=0.1
    '''
    top3_idx = np.where(dis2_data >=0.1)
    print("top3 indx",np.shape(top3_idx))
    reg_data = reg_data[top3_idx]
    dis1_data = dis1_data[top3_idx]
    dis2_data = dis2_data[top3_idx]
    color_data = color_data[top3_idx]
    '''
    #select fpr tpr
    tpr_inx = np.where(reg_data==1)
    fpr_inx = np.where(reg_data==-1)
    print("tpr, fpr shape:",np.shape(tpr_inx),np.shape(fpr_inx))
    if np.shape(tpr_inx)[1] >0:
        tpr_data = dis1_data[tpr_inx]
        dis2_tpr = dis2_data[tpr_inx]
        color_tpr = color_data[tpr_inx]
        dis3_tpr = tpr_data+dis2_tpr
    if np.shape(fpr_inx)[1] >0:
        fpr_data = dis1_data[fpr_inx]
        dis2_fpr = dis2_data[fpr_inx]
        color_fpr = color_data[fpr_inx]
        dis3_fpr = fpr_data+dis2_fpr
    '''
    out_dict['top12'] = tpr_data
    out_dict['top23'] = dis2_tpr
    out_dict['top12'] = fpr_data
    out_dict['top23'] = dis_fpr
    '''
    #write_data(out_file,out_dict)
    threshold = 0.12
    threshold_idx = np.where(dis3_fpr<=threshold) #tpr:338
    print("below threshold ",threshold)
    print(np.shape(threshold_idx))
    plot_continue_color(dis3_fpr,fpr_data,color_fpr,'top13_fpr_t')

def test():
    a = [1,2,3,3,4,5]
    b = [0.3,0.7,0.3,0.4,0.3,0.5]
    fig = plt.figure(num=0,figsize=(10,10))
    ax = fig.add_subplot(111)
    plt.scatter(a,b,alpha=0.5)
    plt.show()

if __name__=='__main__':
    args = params()
    file_path = args.path1
    out_file = args.out_file
    dict_keys = ['filename','blur','left_right','up_down','rotate','cast','da','bbox_face.width','bbox_face.height','distance']
    #file_path = '/home/lxy/Downloads/DataSet/annotations/V2/v2.csv'
    d2_keys = ['filename','confidence','reg_fg','l1_regular','confidence2']
    f2_path = args.path2
    d3_keys = ['filename','distance','reg_fg','l1_regular']
    f3_path = args.path3
    #
    dd2 = read_data(f2_path,d2_keys)
    dd3 = read_data(f3_path,d3_keys)
    #test = dd['filename']
    #print(test[:3])
    key_ = args.key_name
    threshold = args.threshold_value
    save_dir = args.save_dir
    base_dir = args.base_dir
    if args.command == 'static3':
        dd = read_data(file_path,dict_keys)
        statistic_data(dd,dd2,dd3,key_)
    elif args.command == 'static2':
        statistic_fpr(dd3,dd2,key_)
    elif args.command == 'copyfile':
        data_dicts = get_path_by_l1(dd2,threshold)
        copy_save_imgs(data_dicts,base_dir,save_dir)
    elif args.command == 'compare':
        #top32_data = read_data(file_path,d2_keys)
        statics_top3data(dd2,dd2,out_file)
    #test()