from argparse import ArgumentParser
import json
import os
import math
import cv2
import numpy as np

from modules.input_reader import VideoReader, ImageReader
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses

from pathlib import Path

kpt_names = ['neck', 'nose',
                'l_sho', 'l_elb', 'l_wri', 'l_hip', 'l_knee', 'l_ank',
                'r_sho', 'r_elb', 'r_wri', 'r_hip', 'r_knee', 'r_ank',
                'r_eye', 'l_eye',
                'r_ear', 'l_ear']

NECK = 0
NOSE = 1
PELVIS = 2 
L_SHO = 3
L_ELB = 4
L_WRI = 5
L_HIP = 6
L_KNEE = 7
L_ANK = 8
R_SHO = 9
R_ELB = 10
R_WRI = 11
R_HIP = 12
R_KNEE = 13
R_ANK = 14
R_EYE = 15
L_EYE = 16
R_EAR = 17
L_EAR = 18




fileName = 'sample3'
Videofile = Path(fileName).with_suffix('.mp4')
posesFile = Path(fileName).with_suffix('.npy')

if __name__ == '__main__':
    with open(posesFile, 'rb') as f:
        all_poses = np.load(f)
    canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
    plotter = Plotter3d(canvas_3d.shape[:2])
    canvas_3d_window_name = 'Canvas 3D'
    cv2.namedWindow(canvas_3d_window_name)
    cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)

    file_path = None #args.extrinsics_path TODO correct this to extrinistics
    if file_path is None:
        file_path = os.path.join('data', 'extrinsics.json')
    with open(file_path, 'r') as f:
        extrinsics = json.load(f)
    R = np.array(extrinsics['R'], dtype=np.float32)
    t = np.array(extrinsics['t'], dtype=np.float32)

 

    delay = 100
    esc_code = 27
    p_code = 112
    space_code = 32
    mean_time = 0
    limitter = 0
    for frame in all_poses:
        current_time = cv2.getTickCount()
        if frame is None:
            break
  
        poses_3d = frame
        edges = []
        

        edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
        plotter.plot(canvas_3d, poses_3d, edges)


        for pose in poses_3d:
            #
            # 
            # Pose and action estimation area.
            
             

            #If distance between wrist and ear smaller than distance between should and ear 
            # AND wrist is above shoulder
            # ear_to_shoulder_distance can be replaced by a certain heurisitic.
            #ear_to_shoulder_distance = math.sqrt(pow(pose[R_SHO][0]-pose[R_EAR][0],2)+pow(pose[R_SHO][1]-pose[R_EAR][1],2)+pow(pose[R_SHO][2]-pose[R_EAR][2],2))
            ear_to_wrist_distance = math.sqrt(pow(pose[R_WRI][0]-pose[R_EAR][0],2)+pow(pose[R_WRI][1]-pose[R_EAR][1],2)+pow(pose[R_WRI][2]-pose[R_EAR][2],2))
            if 50>ear_to_wrist_distance and pose[R_WRI][2]>pose[R_SHO][2]:
                print("talking on the phone")
            #l_ear_to_shoulder_distance = math.sqrt(pow(pose[L_SHO][0]-pose[L_EAR][0],2)+pow(pose[L_SHO][1]-pose[L_EAR][1],2)+pow(pose[L_SHO][2]-pose[L_EAR][2],2))
            l_ear_to_wrist_distance = math.sqrt(pow(pose[L_WRI][0]-pose[L_EAR][0],2)+pow(pose[L_WRI][1]-pose[L_EAR][1],2)+pow(pose[L_WRI][2]-pose[L_EAR][2],2))
            print(l_ear_to_wrist_distance)
            if  50 > l_ear_to_wrist_distance and pose[L_WRI][2]>pose[L_SHO][2]:
                cv2.putText(canvas_3d, "A terrible leftie is talking on the phone" ,
                    (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                print("A leftie is talking on the phone")
            print(pose)
                        

        
        
        
       

        #draw_poses(frame, poses_2d)
        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        if mean_time == 0:
            mean_time = current_time
        else:
            mean_time = mean_time * 0.95 + current_time * 0.05
       # cv2.putText(canvas_3d, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
       #             (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        #cv2.imshow('ICV 3D Human Pose Estimation', frame)
        cv2.imshow(canvas_3d_window_name, canvas_3d)
        key = cv2.waitKey(delay)
        if key == esc_code:
            break
        if key == p_code:
            if delay == 1:
                delay = 0
            else:
                delay = 1
        if delay == 0:  # allow to rotate 3D canvas while on pause
            key = 0
            while (key != p_code
                   and key != esc_code
                   and key != space_code):
                plotter.plot(canvas_3d, poses_3d, edges)
                cv2.imshow(canvas_3d_window_name, canvas_3d)
                key = cv2.waitKey(33)
            if key == esc_code:
                break
            else:
                delay = 1000

        