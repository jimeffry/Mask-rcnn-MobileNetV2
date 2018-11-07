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
    return data_dict

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
    fg_data = data_2['reg_fg']
    #fg_data = np.array([round(tp,1) for tp in fg_data[:]])
    print('reg_fg',fg_data[:10])
    print('ay',ay_data[:10])
    ay_data = np.array([round(tp,3) for tp in ay_data[:]])
    test_data = datas[key_name]
    '''
    #color data
    '''
    '''
    color_data = np.ones(np.shape(fg_data))
    #color_data *=5
    tpr_indx = np.where(fg_data==1.0)
    fpr_indx = np.where(fg_data==2.0)
    print("tpr,fpr",np.shape(tpr_indx),np.shape(fpr_indx))
    color_data[tpr_indx] = 20
    color_data[fpr_indx] = 60
    print('color_fpr',color_data[fpr_indx])
    '''
    #
    #ax_data = blur_data
    color_data = test_data
    #ax_data = da_data
    color_data = np.array([round(tp,2) for tp in color_data[:]])
    ax_data = np.array([round(tp,2) for tp in ax_data[:]])
    print("ax",ax_data[:10])
    #get min-max
    min_c = np.min(color_data)
    max_c = np.max(color_data)
    min_txt = "color_min:"+str(min_c)
    max_txt = "color_max:"+str(max_c)
    #get the unique number
    col_set = list(set(color_data))
    total_len = len(color_data)
    col_len = len(col_set)
    set_txt = "unique_color:" + str(col_len)
    total_txt = "total_color:"+str(total_len)
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
    plt.text(0.8,0.8,min_txt)
    plt.text(0.8,0.78,max_txt)
    plt.text(1.2,0.8,"color:%s" % key_name)
    plt.text(0.8,0.75,total_txt)
    plt.text(0.8,0.73,set_txt)
    #plt.ylim(0.0,0.5)
    #plt.setp(ax1.get_xticklabels(), fontsize=6)
    # share x only
    plt.savefig("%s.png" %key_name,format='png')
    plt.show()
    
def test():
    a = [1,2,3,3,4,5]
    b = [0.3,0.7,0.3,0.4,0.3,0.5]
    fig = plt.figure(num=0,figsize=(10,10))
    ax = fig.add_subplot(111)
    plt.scatter(a,b,alpha=0.5)
    plt.show()

def statistic_fpr(datas,data_2,data_3,key_name):
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
    ax_data = datas[key_name]
    fg_data = data_2['reg_fg']
    #fg_data = np.array([round(tp,1) for tp in fg_data[:]])
    print('reg_fg',fg_data[:10])
    print('ay',ay_data[:10])
    ay_data = np.array([round(tp,3) for tp in ay_data[:]])
    color_data = np.ones(np.shape(fg_data))
    #color_data *=5
    tpr_indx = np.where(fg_data==1.0)
    fpr_indx = np.where(fg_data==2.0)
    print("tpr,fpr",np.shape(tpr_indx),np.shape(fpr_indx))
    color_data[tpr_indx] = 20
    color_data[fpr_indx] = 60
    print('color_fpr',color_data[fpr_indx])
    #
    ax_data = np.array([round(tp,2) for tp in ax_data[:]])
    print("ax",ax_data[:10])
    #get min-max
    min_x = np.min(ax_data)
    max_x = np.max(ax_data)
    min_txt = "ax_min:"+str(min_x)
    max_txt = "ax_max:"+str(max_x)
    #get the unique number
    ax_set = list(set(ax_data))
    total_len = len(ax_data)
    ax_len = len(ax_set)
    set_txt = "unique_ax:" + str(ax_len)
    total_txt = "total_ax:"+str(total_len)
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
    plt.text(0.8,0.8,min_txt)
    plt.text(0.8,0.78,max_txt)
    plt.text(1.2,0.8,"color: tpr-%f fpr-%f no-reg-%f" % (20,60,1))
    plt.text(0.8,0.75,total_txt)
    plt.text(0.8,0.73,set_txt)
    #plt.ylim(0.0,0.5)
    #plt.setp(ax1.get_xticklabels(), fontsize=6)
    # share x only
    plt.savefig("%s.png" %key_name,format='png')
    plt.show()
    
def test():
    a = [1,2,3,3,4,5]
    b = [0.3,0.7,0.3,0.4,0.3,0.5]
    fig = plt.figure(num=0,figsize=(10,10))
    ax = fig.add_subplot(111)
    plt.scatter(a,b,alpha=0.5)
    plt.show()

if __name__=='__main__':
    
    file_path = '/home/lxy/Downloads/result.csv'
    dict_keys = ['filename','blur','left_right','up_down','rotate','cast','da','bbox_face.width','bbox_face.height','distance']
    #file_path = '/home/lxy/Downloads/DataSet/annotations/V2/v2.csv'
    d2_keys = ['filename','confidence','reg_fg']
    f2_path = "/home/lxy/Downloads/distance_top2.csv"
    d3_keys = ['filename','distance','reg_fg']
    f3_path = "/home/lxy/Downloads/distance_top1.csv"
    #
    dd = read_data(file_path,dict_keys)
    dd2 = read_data(f2_path,d2_keys)
    dd3 = read_data(f3_path,d3_keys)
    #test = dd['filename']
    #print(test[:3])
    statistic_data(dd,dd2,dd3,'rotate')
    
    #test()