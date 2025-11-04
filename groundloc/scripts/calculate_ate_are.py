# Copyright 2025 Dahlem Center for Machine Learning and Robotics, Freie Universität Berlin
# CC BY-NC-SA 4.0
#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import csv
import math
import tf_transformations
from kiss_icp.metrics import absolute_trajectory_error
from argparse import ArgumentParser
from ament_index_python.packages import get_package_share_directory


parser = ArgumentParser(prog="calculate_ate_are", description="Calculate ATE and ARE metrics")
parser.add_argument("-p", "--poses", help="path to the pose file")
parser.add_argument("-gt", "--ground-truth-poses", help="path to the ground-truth pose file. Use this parameter OR the sequence and lidar parameters")
parser.add_argument("-s", "--sequence", help="sequence of the dataset to analyze (KITTI: 00, 01, ..., HeLiPR: roundabout, bridge, town)")
parser.add_argument("-l", "--lidar", help="lidar sensor poses to be used (HeLiPR only: aeva, avia, ouster)")
parser.add_argument("-d", "--dataset", help="name of the dataset (kitti, helipr)")
args = parser.parse_args()
pose_file = args.poses
if pose_file == None:
    pose_file = "result.csv"
gt_poses_file = args.ground_truth_poses
dataset = args.dataset
sequence_value = args.sequence
sensor_value = args.lidar


poses = []  # gt poses

if gt_poses_file == None:
    if sequence_value == None:
        print("Error: No ground truth pose file or sequence given")
        exit()
    
    # add leading zero
    if len(sequence_value) == 1:
        sequence_value = '0' + sequence_value

    if dataset.lower() == 'kitti':
        gt_poses_file = os.path.join(
            get_package_share_directory('groundloc'),
            'res',
            'poses',
            'Kitti',
            args.sequence,
            args.sequence+'_gt_kitti.txt'
        )

    if dataset.lower() == "helipr":
        if sensor_value == None:
            print("Error: No ground truth pose file or LiDAR given")
            exit()

        if sequence_value.lower() == "bridge":
        	sequence_value = "Bridge01"
        if sequence_value.lower() == "roundabout":
        	sequence_value = "Roundabout03"
        if sequence_value.lower() == "town":
        	sequence_value = "Town03"
        # select correct ground truth poses file
        gt_poses_file = os.path.join(
            get_package_share_directory('groundloc'),
            'res',
            'poses',
            'HeLiPR',
            sequence_value,
            sensor_value+'_gt_kitti.txt'
        )

calibstring = "4.276802385584e-04 -9.999672484946e-01 -8.084491683471e-03 -1.198459927713e-02 -7.210626507497e-03 8.081198471645e-03 -9.999413164504e-01 -5.403984729748e-02 9.999738645903e-01 4.859485810390e-04 -7.206933692422e-03 -2.921968648686e-01"
calib = np.fromstring(calibstring, dtype=float, sep=' ')
calib = calib.reshape(3, 4)
calib = np.vstack((calib, [0, 0, 0, 1]))
calib_inv = np.linalg.inv(calib)
with open(gt_poses_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        pose = np.fromstring(line, dtype=float, sep=' ')  # gt pose
        pose = pose.reshape(3, 4)
        pose = np.vstack((pose, [0, 0, 0, 1]))
        if dataset.lower() == "kitti": # apply kitti calibration
            pose = np.matmul(calib_inv, np.matmul(pose, calib))
        q = tf_transformations.quaternion_from_matrix(pose)
        r, p, y = tf_transformations.euler_from_quaternion(q)
        pose = tf_transformations.compose_matrix(angles=[0.0,0.0,y], translate=[pose[0][3], pose[1][3], 0.0])
        poses.append(pose)


count = 0
results = []  # my results
gt_poses = [] # matched gt poses

with open(pose_file) as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        count += 1

        gt_poses.append(poses[count-1])
        pose = tf_transformations.compose_matrix(angles=[0.0,0.0,float(row['yaw'])], translate=[float(row['x']), float(row['y']), 0.0])
        results.append(pose)

        posx = poses[count-1][0][3]  # posx_inter
        posy = poses[count-1][1][3]  # posy_inter
        R = poses[count-1]
        q = tf_transformations.quaternion_from_matrix(R)
        r, p, y = tf_transformations.euler_from_quaternion(q)

are, ate = absolute_trajectory_error(gt_poses, results)
deg = are * 180.0/math.pi
print(f"ARE: {are:.4}  (deg: {deg:.4}°), ATE: {ate:.4} m")
