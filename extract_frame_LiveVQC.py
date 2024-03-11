
import numpy as np
import os
import pandas as pd
import cv2
import scipy.io as scio
import os.path as osp
import random

from utils import train_test_split


def extract_frame(videos_dir, video_name, save_folder):
    filename = os.path.join(videos_dir, video_name)
    video_name_str = video_name[:-4]
    video_capture = cv2.VideoCapture()
    video_capture.open(filename)

    if not video_capture.isOpened():
        print(f"Error: Could not open video {filename}")
        return

    video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(video_capture.get(cv2.CAP_PROP_FPS)))

    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    if video_height > 0 and video_width > 0:
        if video_height > video_width:
            video_width_resize = 520
            video_height_resize = int(video_width_resize / video_width * video_height)
        else:
            video_height_resize = 520
            video_width_resize = int(video_height_resize / video_height * video_width)

        dim = (video_width_resize, video_height_resize)

        video_read_index = 0
        frame_idx = 0
        video_length_min = 8

        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                # key frame
                if (video_read_index < video_length) and (frame_idx % video_frame_rate == 0):
                    read_frame = cv2.resize(frame, dim)
                    exit_folder(os.path.join(save_folder, video_name_str))
                    cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                             '{:03d}'.format(video_read_index) + '.png'), read_frame)
                    video_read_index += 1
                frame_idx += 1

        if video_read_index < video_length_min:
            for i in range(video_read_index, video_length_min):
                cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                         '{:03d}'.format(i) + '.png'), read_frame)
    else:
        print(f"Error: Invalid video dimensions for {filename}")

    return


def exit_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    return


if __name__ == '__main__':
    videos_dir = '../DataSet/Video'
    filename_path = 'data/LIVE_VQC_data.txt'
    save_folder = 'Live_VQC_image'

    #extract_frame(videos_dir, 'B166.mp4', save_folder)
    train_infos, val_infos = train_test_split(filename_path)
    # n_video = len(train_infos)
    # video_names = []
    # for i in range(n_video):
    #      video_names.append(train_infos[i].get('filename'))
    #
    # for i in range(n_video):
    #     video_name = video_names[i].strip()
    #     print(f'start extract {i}th video: {video_name}')
    #     extract_frame(videos_dir, video_name, save_folder)
    print("val start:")
    n_video = len(val_infos)
    video_names = []
    for i in range(n_video):
         video_names.append(val_infos[i].get('filename'))

    for i in range(n_video):
        video_name = video_names[i].strip()
        print(f'start extract {i}th video: {video_name}')
        extract_frame(videos_dir, video_name, save_folder)