
from test import Test
from Utils.color_lib import RGBmean,RGBstdv
from Utils.Utils import data_dict_reader,mkdir_if_missing, wjx_data_dict_reader
import os
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    Data= 'CUB'
    dst = './wjxTest/CUB/resnet50_proxyanchor+softmax_orth_0.1_batchsize60_SGD0.0001_M3/'
    phase = 'test'
    mkdir_if_missing(dst)

    cur_root = os.getcwd()

    data_dict = wjx_data_dict_reader(Data,phase,cur_root)
    model_path = './wjxTest/CUB/resnet50_proxyanchor+softmax_orth_0.1_batchsize60_SGD0.0001_M3/67_model.pth'#
    Test.eval(dst, Data, data_dict, model_path, feature_save_name='wjx-selected_feature_67.npz', salency='scda',pool_type='max_avg_V',phase=phase).run()#



