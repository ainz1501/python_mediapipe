import cv2 
import numpy as np
import matplotlib.pyplot as plt

# 使用ビデオ番号 20,21番の動画は使用できない
VIDEO_NUM = 0
# 使用データセット
DATASET_NAME = "171204_pose3"
IMG_FOLDER_PATH = "./input_images/hd00_frames/"

hd00_list = [1400, 6200]

VIDEO = cv2.VideoCapture("./panoptic-toolbox/"+DATASET_NAME+"/hdVideos/hd_00_"+str(VIDEO_NUM).zfill(2)+".mp4")
frame_num = 1
while True:
    print("now frame:",frame_num)
    img_path = IMG_FOLDER_PATH+"hd00_"+str(frame_num*100).zfill(4)+"frame.png"
    # キャプチャー処理
    for i in range(100):
        # ビデオキャプチャー
        ret1, frame = VIDEO.read()
    if not ret1:
        print("breaked frame:", frame_num)
        break

    cv2.imwrite(img_path, frame)
    frame_num += 1


