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

from data_loader import  VideoDataset_images_with_motion_features, \
    VideoDataset_LSVQ_Swin_features, VideoDataset_VQA_Swin_features
from final_fusion_model import swin_small_patch4_window7_224 as create_model
from final_fusion_model_v2 import swinv2_small_patch4_window8_256 as create_model_v2

from torchvision import transforms


def main(args):
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model_v2()


    weights = 'ckpts/StorageUseLiner_epoch_94_SRCC_0.896898.pth'
    weights_dict = torch.load(weights, map_location=device)
    print(weights)
    print(model.load_state_dict(weights_dict))

    model = model.to(device)
    # Multi
    # new_state_dict = OrderedDict()
    # for k, v in weights_dict.items():
    #     name = k[7:]
    #     new_state_dict[name] = v
    #
    # print(model.load_state_dict(new_state_dict))
    # model = torch.nn.DataParallel(model, device_ids=[0,1,2]).to(device)

    videos_dir = 'konvid1k_image'
    data_dir_3D = 'KoNViD_SlowFast_feature'

    #datainfo = 'data/KoNViD-1k_data_modified.mat'

    datainfo = 'data/Konvid_1k.txt'
    for i in range(90,100):
        print(i)
        train_infos, val_infos = train_test_split(datainfo,0,i)

        transformations_test = transforms.Compose(
            [transforms.Resize((512, 480)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        #testset = VideoDataset_VQA_Swin_features(videos_dir, data_dir_3D, datainfo, transformations_test, 'KoNViD-1k', 'SlowFast', 'val')
        testset = VideoDataset_VQA_Swin_features(videos_dir, data_dir_3D, val_infos, transformations_test, 'other', 'SlowFast', 'val')
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                  shuffle=False, num_workers=args.num_workers)

        # do validation after each epoch
        with torch.no_grad():
            model.eval()
            label = np.zeros([len(testset)])
            y_output = np.zeros([len(testset)])
            videos_name = []
            for i, (video, tem_f, mos, _) in enumerate(test_loader):
                video = video.to(device)
                tem_f = tem_f.to(device)
                video = torch.reshape(video, [video.shape[0] * video.shape[1], 3, 512, 480])
                tem_f = torch.reshape(tem_f, [tem_f.shape[0] * tem_f.shape[1], 2304])
                label[i] = mos.item()
                outputs = model(video, tem_f)
                # video_name = _[0][:-4]
                # img_path = "./konvid1k_image/" + video_name + "/000.png"
                # save_path = "test"
                # visualize_grid_attention_v2(video_name,
                #                             img_path,
                #                             save_path=save_path,
                #                             attention_mask=avg_attns[0][0] + avg_attns[0][1] + avg_attns[1][0] + +
                #                             avg_attns[1][1],
                #                             save_image=True,
                #                             save_original_image=False,
                #                             quality=300)
                y_output[i] = torch.mean(outputs).item()

            val_PLCC, val_SRCC, val_KRCC, val_RMSE = performance_fit(label, y_output)
            print('The result on the databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(\
                val_SRCC, val_KRCC, val_PLCC, val_RMSE))

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
    parser.add_argument('--train_batch_size', type=int, default=1)
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
