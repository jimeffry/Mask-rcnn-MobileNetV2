import numpy as np 
import string
import matplotlib.pyplot as plt

def plt_histgram(file_in,file_out,distance,num_bins=20):
    '''
    file_in: saved distances txt file
    file_out: output bins and 
    '''
    out_name = file_out
    input_file=open(file_in,'r')
    file_1 = 'conf_'+file_out
    file_2 = 'dis_'+file_out
    out_file1=open(file_1,'w')
    out_file2=open(file_2,'w')
    data_arr_conf=[]
    data_arr_dist=[]
    print(out_name)
    out_list = out_name.strip()
    out_list = out_list.split('/')
    out_name1 = out_list[-1][:-4]+"conf.png"
    out_name2 = out_list[-1][:-4]+"dist.png"
    print(out_name)
    for line in input_file.readlines():
        line = line.strip()
        line_s = line.split(',')
        temp_0=string.atof(line_s[0])
        temp_1=string.atof(line_s[1])
        data_arr_conf.append(temp_1)
        data_arr_dist.append(temp_0)
    data_in_conf=np.asarray(data_arr_conf)
    data_in_dist=np.asarray(data_arr_dist)
    if distance is None:
        max_bin = np.max(data_in_conf)
        print("confdence max:",max_bin)
    else:
        max_bin = distance
    plt.subplot(211)
    conf_datas,conf_bins,conf_c=plt.hist(data_in_conf,num_bins,range=(0.0,max_bin),normed=0,color='blue',cumulative=0)
    plt.title('histogram_conf')
    plt.savefig(out_name1, format='png')
    #a,b,c=plt.hist(data_in,num_bins,normed=1,color='blue',cumulative=1)
    if distance is None:
        max_bin = np.max(data_in_dist)
        print("max distance:",max_bin)
    else:
        max_bin = distance
    plt.subplot(212)
    datas,bins,c=plt.hist(data_in_dist,num_bins,range=(0.0,max_bin),normed=0,color='blue',cumulative=0)
    plt.title('histogram_dis')
    plt.savefig(out_name2, format='png')
    plt.show()
    for i in range(num_bins):
        out_file2.write(str(datas[i])+'\t'+str(bins[i])+'\n')
        out_file1.write(str(conf_datas[i])+'\t'+str(conf_bins[i])+'\n')
    input_file.close()
    out_file1.close()
    out_file2.close()

if __name__ == '__main__':
    file_in = "/home/lxy/Downloads/disconf.txt"
    file_out = 'hist.txt'
    plt_histgram(file_in,file_out,None)