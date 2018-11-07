# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/11/06 10:09
#project: Face recognize
#company: Senscape
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

def statistic_data(datas,data_2,dict_keys):
    '''
    datas: input dict ,include 10 keys
    data_2: input dict 2 include 2keys: filename, confidence
    '''
    ay_data = data_2['confidence']
    #ax_data = datas[dict_keys[-1]]
    print('ay',ay_data[:10])
    ay_data = np.array([round(tp,3) for tp in ay_data[:]])
    blur_data = datas[dict_keys[1]]
    blur_data = np.array([round(tp,2) for tp in blur_data[:]])
    lr_data = datas[dict_keys[2]]
    lr_data = np.array([round(tp,2) for tp in lr_data[:]])
    ud_data = datas[dict_keys[3]]
    ud_data = np.array([round(tp,2) for tp in ud_data[:]])
    rt_data = datas[dict_keys[4]]
    rt_data = np.array([round(tp,2) for tp in rt_data[:]])
    ct_data = datas[dict_keys[5]]
    ct_data = np.array([round(tp,2) for tp in ct_data[:]])
    da_data = datas[dict_keys[6]]
    da_data = np.array([round(tp,2) for tp in da_data[:]])
    #
    ax_data = da_data
    print("ax",ax_data[:10])
    #sorft data
    indx_ay = np.argsort(ax_data)
    print(ax_data[indx_ay[:10]])
    print(ay_data[indx_ay[:10]])
    #sorft
    ax_data = ax_data[indx_ay[:]]
    ay_data = ay_data[indx_ay[:]]
    fig = plt.figure(num=0,figsize=(20,10))
    ax1 = fig.add_subplot(111)
    plt.scatter(ax_data,ay_data,alpha=0.5)
    plt.xlabel('confidence')
    plt.ylabel('da')
    plt.title('da Vs confidence')
    #plt.ylim(0.0,0.5)
    #plt.setp(ax1.get_xticklabels(), fontsize=6)
    # share x only
    '''
    ax2 = plt.subplot(612, sharex=ax1)
    plt.plot(ax_data, lr_data,label='left_right')
    # make these tick labels invisible
    plt.setp(ax2.get_xticklabels(), visible=False)
    # share x and y
    #ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)
    
    ax3 = plt.subplot(613,sharex=ax1)
    plt.plot(ax_data,ud_data,label='up_down')
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax4 = plt.subplot(614,sharex=ax1)
    plt.plot(ax_data,rt_data,label='rotate')
    plt.setp(ax4.get_xticklabels(),visible=False)
    ax5 = plt.subplot(615,sharex=ax1)
    plt.plot(ax_data,ct_data,label='cast')
    plt.setp(ax5.get_xticklabels(),visible=False)
    ax6 = plt.subplot(616,sharex=ax1)
    plt.plot(ax_data,da_data,label='da')
    plt.setp(ax6.get_xticklabels(),visible=False)
    '''
    plt.savefig("da_confidence.png",format='png')
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
    d2_keys = ['filename','confidence']
    f2_path = "/home/lxy/Downloads/distance_top2.csv"
    #
    dd = read_data(file_path,dict_keys)
    dd2 = read_data(f2_path,d2_keys)
    #test = dd['filename']
    #print(test[:3])
    statistic_data(dd,dd2,dict_keys)
    
    #test()