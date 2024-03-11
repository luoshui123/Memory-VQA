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
    # model = torch.nn.DataParallel(model,device_ids=[0,1,2]).to(device)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name or "head_att" in name or "head_att_z" in name or "head_Line" in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=0.00002, weight_decay=0.0000001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)

    if args.loss_type == 'L1RankLoss':
        criterion = L1RankLoss(batchsize=args.train_batch_size)

    param_num = 0
    for param in pg:
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    # # Konvid1K
    # finetune_name = "Konvid1K"
    # videos_dir = 'konvid1k_image'
    # data_dir_3D = 'KoNViD_SlowFast_feature'
    # datainfo = 'data/Konvid_1k.txt'
    # train_infos, val_infos = train_test_split(datainfo, 0.8, 99)
    #
    # Youtube
    # finetune_name = "Youtube"
    # videos_dir = 'youtube_ugc_image'
    # data_dir_3D = 'youtube_ugc_SlowFast_feature'
    # datainfo = 'data/Youtube_ugc.txt'
    # train_infos, val_infos = train_test_split(datainfo, 0.8, 90)
    #
    # # LiveVQC
    finetune_name = "LiveVQC"
    videos_dir = 'Live_VQC_image'
    data_dir_3D = 'Live_VQC_SlowFast_feature'
    datainfo = 'data/LIVE_VQC_data.txt'
    train_infos, val_infos = train_test_split(datainfo, 0.8, 93)

    # # Divide3k
    # finetune_name = "Divide3k"
    # videos_dir = 'Divide_3k_image'
    # data_dir_3D = 'Divide_3k_SlowFast_feature'
    # test_datainfo = 'data/Divide_3k_test.txt'
    # train_datainfo = 'data/Divide_3k_train.txt'
    # train_infos, _ = train_test_split(train_datainfo, 1)
    # _, val_infos = train_test_split(test_datainfo, 0)


    transformations_train = transforms.Compose(
        [transforms.Resize((512, 480)),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transformations_test = transforms.Compose(
        [transforms.Resize((512, 480)),transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainset = VideoDataset_VQA_Swin_features(videos_dir, data_dir_3D, train_infos, transformations_train, 'Live_VQC', 'SlowFast', 'train')
    testset = VideoDataset_VQA_Swin_features(videos_dir, data_dir_3D, val_infos, transformations_test, 'Live_VQC', 'SlowFast', 'val')

    ## dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size,
                                               shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=args.num_workers)


    best_test_criterion = -1  # SROCC min
    best_test = []

    print('Starting training:')

    old_save_name = None

    for epoch in range(args.epochs):
        model.train()
        batch_losses = []
        batch_losses_each_disp = []
        session_start_time = time.time()
        for i, (video, tem_f, mos, _) in enumerate(train_loader):

            video = video.to(device)
            tem_f = tem_f.to(device)
            video = torch.reshape(video, [video.shape[0] * video.shape[1], 3, 512, 480])
            tem_f = torch.reshape(tem_f, [tem_f.shape[0] * tem_f.shape[1], 2304])
            labels = mos.to(device).float()
            outputs = model(video, tem_f)
            optimizer.zero_grad()

            loss = criterion(outputs, labels)
            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())
            loss.backward()

            optimizer.step()

            if (i + 1) % (args.print_samples // args.train_batch_size) == 0:
                session_end_time = time.time()
                avg_loss_epoch = sum(batch_losses_each_disp) / (args.print_samples // args.train_batch_size)
                print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' % \
                      (epoch + 1, args.epochs, i + 1, len(trainset) // args.train_batch_size, \
                       avg_loss_epoch))
                batch_losses_each_disp = []
                print('CostTime: {:.4f}'.format(session_end_time - session_start_time))
                session_start_time = time.time()

        avg_loss = sum(batch_losses) / (len(trainset) // args.train_batch_size)
        print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

        scheduler.step()
        lr = scheduler.get_last_lr()
        print('The current learning rate is {:.06f}'.format(lr[0]))

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
                'Epoch {} completed. The result on the test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                    epoch + 1, \
                    test_SRCC, test_KRCC, test_PLCC, test_RMSE))

            if test_SRCC > best_test_criterion:
                print("Update best model using best_test_criterion in epoch {}".format(epoch + 1))
                best_test_criterion = test_SRCC
                best_test = [test_SRCC, test_KRCC, test_PLCC, test_RMSE]
                print('Saving model...')
                if not os.path.exists(args.ckpt_path):
                    os.makedirs(args.ckpt_path)

                if epoch > 0:
                    if os.path.exists(old_save_name):
                        os.remove(old_save_name)

                save_model_name = args.ckpt_path + '/' + 'Finetune_%s_epoch_%d_SRCC_%f.pth' % (finetune_name ,epoch + 1, test_SRCC)
                torch.save(model.state_dict(), save_model_name)
                old_save_name = save_model_name

    print('Training completed.')
    print(
        'The best training result on the test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
            best_test[0], best_test[1], best_test[2], best_test[3]))


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

    parser.add_argument('--weights', type=str, default='../SwinCKPT/swinv2_small_patch4_window8_256.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=True)

    args = parser.parse_args()
    main(args)