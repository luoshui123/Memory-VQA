# -*- coding: utf-8 -*-
import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from utils import performance_fit, train_test_split
from utils import L1RankLoss
import torch.nn as nn

from data_loader import VideoDataset_images_with_motion_features, \
    VideoDataset_LSVQ_Swin_features, VideoDataset_VQA_Swin_features
from final_fusion_model import swin_small_patch4_window7_224 as create_model
from final_fusion_model_v2 import swinv2_small_patch4_window8_256 as create_model_v2

from torchvision import transforms
import time
import os.path as osp
import random



def main(args):
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model_v2()


    # weights = 'ckpts/.pth'
    # weights_dict = torch.load(weights, map_location=device)
    # print(model.load_state_dict(weights_dict))
    model = model.to(device)
    # Multi
    # new_state_dict = OrderedDict()
    # for k, v in weights_dict.items():
    #     name = k[7:]  # module字段在最前面，从第7个字符开始就可以去掉module
    #     new_state_dict[name] = v  # 新字典的key值对应的value一一对应
    #
    # print(model.load_state_dict(new_state_dict))
    # model = torch.nn.DataParallel(model, device_ids=[0,1,2]).to(device)


    # LSVQ
    videos_dir = 'LSVQ_image'
    feature_dir = 'LSVQ_SlowFast_feature'
    datainfo_train = 'data/LSVQ_whole_train.csv'
    datainfo_test = 'data/LSVQ_whole_test.csv'
    datainfo_test_1080p = 'data/LSVQ_whole_test_1080p.csv'

    all_src, all_plc, all_krc, all_rms = 0,0,0,0

    transformations_test = transforms.Compose(
        [transforms.Resize((512, 480)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    testset = VideoDataset_LSVQ_Swin_features(videos_dir, feature_dir, datainfo_test, transformations_test,
                                                       'LSVQ_test', 'SlowFast')
    testset_1080p = VideoDataset_LSVQ_Swin_features(videos_dir, feature_dir, datainfo_test_1080p,
                                                             transformations_test, 'LSVQ_test_1080p',
                                                             'SlowFast')
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=args.num_workers)
    test_loader_1080p = torch.utils.data.DataLoader(testset_1080p, batch_size=1,
                                                    shuffle=False, num_workers=args.num_workers)
    # do validation after each epoch
    with torch.no_grad():
        model.eval()
        label = np.zeros([len(testset)])
        y_output = np.zeros([len(testset)])
        for i, (video, tem_f, mos, _) in enumerate(test_loader):
            video = video.to(device)
            tem_f = tem_f.to(device)
            video = torch.reshape(video, [video.shape[0] * video.shape[1], 3, 512, 480])
            tem_f = torch.reshape(tem_f, [tem_f.shape[0] * tem_f.shape[1], 2304])
            label[i] = mos.item()
            outputs = model(video, tem_f)
            y_output[i] = torch.mean(outputs).item()

        test_PLCC, test_SRCC, test_KRCC, test_RMSE = performance_fit(label, y_output)

        print(
            'The result on the test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                test_SRCC, test_KRCC, test_PLCC, test_RMSE))

        label_1080p = np.zeros([len(testset_1080p)])
        y_output_1080p = np.zeros([len(testset_1080p)])
        for i, (video, tem_f, mos, _) in enumerate(test_loader_1080p):
            video = video.to(device)
            tem_f = tem_f.to(device)
            video = torch.reshape(video, [video.shape[0] * video.shape[1], 3, 512, 480])
            tem_f = torch.reshape(tem_f, [tem_f.shape[0] * tem_f.shape[1], 2304])
            label_1080p[i] = mos.item()
            outputs = model(video, tem_f)
            y_output_1080p[i] = torch.mean(outputs).item()

        test_PLCC_1080p, test_SRCC_1080p, test_KRCC_1080p, test_RMSE_1080p = performance_fit(label_1080p,
                                                                                             y_output_1080p)

        print(
            ' The result on the test_1080p databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                test_SRCC_1080p, test_KRCC_1080p, test_PLCC_1080p, test_RMSE_1080p))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--database', type=str)
    parser.add_argument('--model_name', type=str)
    # training parameters
    parser.add_argument('--conv_base_lr', type=float, default=1e-5)
    parser.add_argument('--decay_ratio', type=float, default=0.95)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--n_trial', type=int, default=0)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--exp_version', type=int)
    parser.add_argument('--print_samples', type=int, default=1000)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    # misc
    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--loss_type', type=str, default='L1RankLoss')

    parser.add_argument('--weights', type=str, default='../Light-VQA/swin_small_patch4_window7_224.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=True)

    args = parser.parse_args()
    main(args)
