#!/bin/bash
#['filename','blur','left_right','up_down','rotate','cast','da','bbox_face.width','bbox_face.height','distance']
'''
python histogram.py --key-name  blur
python histogram.py --key-name  left_right
python histogram.py --key-name  up_down
python histogram.py --key-name  rotate
python histogram.py --key-name  cast
python histogram.py --key-name  da
'''
'''
python histogram.py --key-name  blur        --command static2
python histogram.py --key-name  left_right  --command static2
python histogram.py --key-name  up_down     --command static2
python histogram.py --key-name  rotate      --command static2
python histogram.py --key-name  cast        --command static2
python histogram.py --key-name  da          --command static2
'''
#python histogram.py --path2 ./distance_top2.csv --path3 ./distance_top1.csv --key-name l1_regular  --save-dir /home/lxy/Downloads/DataSet/Face_reg/prison_result/test_l1reg \
 #       --base-dir /home/lxy/Downloads/DataSet/Face_reg/model_test_data/ --threshold 500  --command copyfile
# compare top21 and top31 top23
# python histogram.py --path2 ./distance_top2.csv --path3 ./distance_top1.csv   --path1 ./distance_top3.csv    --command compare
python histogram.py --path2 ./output/dis_top2.csv --path3 ./output/dis_top1.csv      --command compare
#test prison
#python histogram.py --key-name  l1_regular --path2 ./output/dis_top2.csv --path3 ./output/dis_top1.csv --command static3
#python histogram.py --key-name  l1_regular --path2 ./output/dis_top2.csv --path3 ./output/dis_top1.csv  --command static2