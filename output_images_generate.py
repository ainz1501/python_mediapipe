import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import glob

def concatenation_images():
    idx = 1
    while True:
        non_annotation_image = cv2.imread("/Users/tokudataichi/Documents/python_mediapipe/output_images/hd00_"+str(idx)+".png")
        annotation_image = cv2.imread("/Users/tokudataichi/Documents/python_mediapipe/output_images/hd00_"+str(idx)+"_annotated.png")
        plot3d_image = cv2.imread("/Users/tokudataichi/Documents/python_mediapipe/output_3dplot/3dplot_"+str(idx)+".png")
        if (non_annotation_image is None) or (annotation_image is None) or (plot3d_image is None):
            break
        HEIGHT, WIDTH, _ = non_annotation_image.shape # HEIGHT=1080, WIDTH=1920
        plot3d_image_resize = cv2.resize(plot3d_image, (HEIGHT, HEIGHT), interpolation=cv2.INTER_CUBIC)
        concatenation_image = cv2.hconcat([non_annotation_image, annotation_image, plot3d_image_resize])
        cv2.imshow("concatenation_image", concatenation_image)
        cv2.imwrite("/Users/tokudataichi/Documents/python_mediapipe/output_images/hdvideo_frames_00&01/frame_"+str(idx).zfill(4)+".jpg", concatenation_image)
        idx += 1

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
create_video_from_images()