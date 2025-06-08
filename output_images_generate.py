import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import glob

def concatenation_images():
    idx = 1
    while True:
        left_annotation_image = cv2.imread("./output_images/images_temporary_storage/annotated_L.png")
        right_annotation_image = cv2.imread("./output_images/images_temporary_storage/annotated_R.png")
        plot3d_image = cv2.imread("/Users/tokudataichi/Documents/python_mediapipe/output_3dplot/3dplot_"+str(idx)+".png")
        HEIGHT, WIDTH, _ = left_annotation_image.shape # HEIGHT=1080, WIDTH=1920
        plot3d_image_resize = cv2.resize(plot3d_image, (HEIGHT, HEIGHT), interpolation=cv2.INTER_CUBIC)
        concatenation_image = cv2.hconcat([left_annotation_image, right_annotation_image])
        cv2.imshow("concatenation_image", concatenation_image)
        cv2.imwrite("./corresponding_landmarks_image.png", concatenation_image)
        break

def create_video_from_images():
    images = []
    for name in sorted(glob.glob('/Users/tokudataichi/Documents/python_mediapipe/output_images/hdvideo_frames_00&01/*.jpg')):
        img = cv2.imread(name)
        if img is None:
            break
        print(name)
        images.append(img)
    HEIGHT, WIDTH, _ = images[0].shape
    out_video = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (WIDTH, HEIGHT)) 
    
    for i in range(len(images)):
        out_video.write(images[i])

concatenation_images()