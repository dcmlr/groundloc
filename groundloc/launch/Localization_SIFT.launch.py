# Copyright 2025 Dahlem Center for Machine Learning and Robotics, Freie Universität Berlin
# CC BY-NC-SA 4.0
import os

import numpy as np
import math
import tf_transformations
import launch
import launch_ros
from ament_index_python.packages import get_package_share_directory

def setup(context):
	dataset_value = launch.substitutions.LaunchConfiguration('dataset_name').perform(context).lower()
	dataset_path = launch.substitutions.LaunchConfiguration('dataset_path').perform(context).lower()
	poses_file = launch.substitutions.LaunchConfiguration('poses_file').perform(context)
	gt_poses_file = launch.substitutions.LaunchConfiguration('gt_poses_file').perform(context)
	sequence_value = launch.substitutions.LaunchConfiguration('sequence').perform(context)
	sensor_value = launch.substitutions.LaunchConfiguration('sensor').perform(context)
	project_2d = launch.substitutions.LaunchConfiguration('project_2d').perform(context).lower() == "true"
	map_file = launch.substitutions.LaunchConfiguration('map_file').perform(context)
	config_file = launch.substitutions.LaunchConfiguration('config_file').perform(context)

	if dataset_value == "kitti":
		# add leading zero
		if len(sequence_value) == 1:
			sequence_value = '0' + sequence_value
		
		# select correct kiss-icp poses file
		if poses_file == '':
			poses_file = os.path.join(
		        get_package_share_directory('groundloc'),
		        'res',
		        'poses',
		        'Kitti',
		        sequence_value,
		        sequence_value+'_poses_kitti.txt'
		    )

		# select correct map file for sequence
		if map_file == '':
			map_file = os.path.join(
		        get_package_share_directory('groundloc'),
		        'res',
		        'maps',
		        'Kitti',
		        sequence_value+'.tif'
		    )

		# select correct ground truth poses file
		if gt_poses_file == '':
			gt_poses_file = os.path.join(
		        get_package_share_directory('groundloc'),
		        'res',
		        'poses',
		        'Kitti',
		        sequence_value,
		        sequence_value+'_gt_kitti.txt'
		    )

		# fix for ros2 typecasting bug
		if sequence_value == '08' or sequence_value == '09':
			sequence_value = sequence_value[1:]

	if dataset_value == "helipr":
		if sequence_value.lower() == "bridge":
			sequence_value = "Bridge01"
		if sequence_value.lower() == "roundabout":
			sequence_value = "Roundabout03"
		if sequence_value.lower() == "town":
			sequence_value = "Town03"

		if poses_file == '':
			poses_file = os.path.join(
		        get_package_share_directory('groundloc'),
		        'res',
		        'poses',
		        'HeLiPR',
		        sequence_value,
		        sensor_value+'_poses_kitti.txt'
		    )

		# select correct map file for sequence
		if map_file == '':
			if sequence_value.lower() == "bridge01":
				filename = "bridge04.tif"
			elif sequence_value.lower() == "roundabout03":
				filename = "roundabout0102.tif"
			elif sequence_value.lower() == "town03":
				filename = "town0201.tif"
			else:
				print("No map file available, please provide map_file parameter")
			map_file = os.path.join(
		        get_package_share_directory('groundloc'),
		        'res',
		        'maps',
		        'HeLiPR',
		        sensor_value,
		        sensor_value.lower()+'_'+filename
		    )

		# select correct ground truth poses file
		if gt_poses_file == '':
			gt_poses_file = os.path.join(
		        get_package_share_directory('groundloc'),
		        'res',
		        'poses',
		        'HeLiPR',
		        sequence_value,
		        sensor_value+'_gt_kitti.txt'
		    )

	if dataset_value == "kitti" or dataset_value == "kitti360":
		config_file = 'kitti.yaml'
	elif dataset_value == "mulran" or dataset_value == "mulran_gt":
		config_file = 'mulran.yaml'
	elif dataset_value == "helipr":
		if sensor_value.lower() == "ouster":
			config_file = 'helipr_ouster.yaml'
		if sensor_value.lower() == "aeva":
			config_file = 'helipr_aeva.yaml'
			project_2d = True
		if sensor_value.lower() == "avia":
			config_file = 'helipr_avia.yaml'
			project_2d = True
		if sensor_value.lower() == "velodyne":
			config_file = 'helipr_velodyne.yaml'
			project_2d = True

	# GroundGrid parameters for the LiDAR
	# Select parameter file according to lidar sensor 
	config = os.path.join(
        get_package_share_directory('groundgrid'),
        'param',
        config_file
    )

	# check parameters
	if not os.path.isfile(config):
		print(f"ERROR: Could not find groundgrid config file: {config}, using defaults")
	if not os.path.isfile(map_file):
		print(f"ERROR: Could not find map file: {map_file}")
	if dataset_value != "live":
		if not os.path.isfile(poses_file):
			print(f"ERROR: Could not find poses file: {poses_file}")

	# starting position depends on sequence/dataset selected
	x = 0
	y = 0
	z = 0
	yaw = 0
	if gt_poses_file != "" and dataset_value == "helipr":
		with open(gt_poses_file) as file:
			line = file.readline()
			pose = np.fromstring(line, dtype=float, sep=' ')  # gt pose
			pose = pose.reshape(3, 4)
			pose = np.vstack((pose, [0, 0, 0, 1]))
			q = tf_transformations.quaternion_from_matrix(pose)
			r, p, yaw = tf_transformations.euler_from_quaternion(q)
			x = pose[0][3]
			y = pose[1][3]
			z = pose[2][3]

	map_utm_tf = launch_ros.actions.Node(
		package='tf2_ros',
		executable='static_transform_publisher',
		name='map_utm_tf_pub',
		arguments=[str(x), str(y), str(z), str(yaw), '0', '0', 'map', 'utm'])

	# Setup of the extrinsic sensor calibration transform
	if dataset_value == "helipr":
		if sensor_value.lower() == "ouster":
		    base_link_velodyne_tf = launch_ros.actions.Node(
		        package='tf2_ros',
		        executable='static_transform_publisher',
		        name='base_link_velodyne_tf_pub',
		        arguments=['0.0', '0.0', '2.0', '0.0', '0.0', '0.0', #TODO: this is a rough guess
		                   'base_link',
		                   'velodyne'])
		elif sensor_value.lower()  == "aeva":
		    base_link_velodyne_tf = launch_ros.actions.Node(
		        package='tf2_ros',
		        executable='static_transform_publisher',
		        name='base_link_velodyne_tf_pub',
		        arguments=['0.0', '0.0', '2.0', '0.0', '-0.015595725680495854', '0.007271127000748835', # Aeva -> Ouster
		                   'base_link',
		                   'velodyne'])
		elif sensor_value.lower() == "avia":
		    base_link_velodyne_tf = launch_ros.actions.Node(
		        package='tf2_ros',
		        executable='static_transform_publisher',
		        name='base_link_velodyne_tf_pub',
		        arguments=['0.0', '0.0', '2.0', '0.0', '0.009958177597463428', '-0.017660507983823313', # Avia -> Ouster
		                   'base_link',
		                   'velodyne'])
		elif sensor_value.lower() == "velodyne":
		    base_link_velodyne_tf = launch_ros.actions.Node(
		        package='tf2_ros',
		        executable='static_transform_publisher',
		        name='base_link_velodyne_tf_pub',
		        arguments=['0.0', '0.0', '2.0', '0.0', '0.005542258713330074', '0.0056861114771429465', # Velodyne -> Ouster
		                   'base_link',
		                   'velodyne'])
	elif dataset_value == "kitti":
	    base_link_velodyne_tf = launch_ros.actions.Node(
	        package='tf2_ros',
	        executable='static_transform_publisher',
	        name='base_link_velodyne_tf_pub',
	        arguments=['0.0', '0.0', '1.733', '0.0', '0.0', '0.0', 
	                   'base_link',
	                   'velodyne'])
	elif dataset_value == "kitti360":
	    base_link_velodyne_tf = launch_ros.actions.Node(
	        package='tf2_ros',
	        executable='static_transform_publisher',
	        name='base_link_velodyne_tf_pub',
	        arguments=['0.771049336280387', '0.29854143649499193', '-0.8362802189143268', '0.005805702483432155', '-0.010400477715954315', '3.1385789123483367', 
	                   'base_link',
	                   'velodyne'])
	elif dataset_value == "mulran":
	    base_link_velodyne_tf = launch_ros.actions.Node(
	        package='tf2_ros',
	        executable='static_transform_publisher',
	        name='base_link_velodyne_tf_pub',
	        arguments=['0', '0.0', '0', '0', '-0.0222359877559', '0.000001745329',
	                   'base_link',
	                   'velodyne'])
	else:
	    base_link_velodyne_tf = launch_ros.actions.Node(
	        package='tf2_ros',
	        executable='static_transform_publisher',
	        name='base_link_velodyne_tf_pub',
	        arguments=['0.0', '0.0', '2.0', '0.0', '0.0', '0.0', # default
	                   'base_link',
	                   'velodyne'])

	return [map_utm_tf,
		base_link_velodyne_tf,
		launch_ros.actions.Node(
                                package='quatro',
								executable='quatro_node_sift',
								name='quatro_node_sift',
								output='screen',
								parameters=[
									{'bev_registration/geotiff_path': map_file},
									{'use_sim_time': launch.substitutions.LaunchConfiguration('simtime')},
									{'bev_registration/debug_imgs': launch.substitutions.LaunchConfiguration('debug_imgs')},
									{'bev_registration/visualize': launch.substitutions.LaunchConfiguration('visualize')},
									{'bev_registration/log_all_matchings': launch.substitutions.LaunchConfiguration('log_all_matchings')},
									{'bev_registration/correction_factor': launch.substitutions.LaunchConfiguration('correction_factor')},
								]),
		launch_ros.actions.Node(
                                package='groundgrid',
								executable='groundgrid_node',
								name='groundgrid_node',
								output='screen',
								parameters=[
									{'groundloc/dataset_name': launch.substitutions.LaunchConfiguration('dataset_name')},
									{'use_sim_time': launch.substitutions.LaunchConfiguration('simtime')},
									{'groundloc/dataset_gen': launch.substitutions.LaunchConfiguration('dataset_gen')},
									{'groundloc/dataset_path': launch.substitutions.LaunchConfiguration('dataset_path')},
									{'groundloc/sensor': launch.substitutions.LaunchConfiguration('sensor')},
									{'groundloc/sequence': sequence_value},
									{'groundloc/poses_file': poses_file},
									{'groundloc/gt_poses_file': gt_poses_file},
									{'groundloc/poses_delimiter': launch.substitutions.LaunchConfiguration('poses_file_delimiter')},
									{'groundloc/project_2d': project_2d},
									{'groundloc/frame_rate_cap': launch.substitutions.LaunchConfiguration('frame_rate_cap')},
									{'groundloc/visualize': launch.substitutions.LaunchConfiguration('visualize')},
									config
								],
								remappings=[
									('/pointcloud', launch.substitutions.LaunchConfiguration('point_cloud_topic')),
								]
	)]


def generate_launch_description():
	# use sim time (for rosbag playbag)
	sim_time_arg = launch.actions.DeclareLaunchArgument(
		name='simtime',
		default_value='False',
		description='Whether to use simulated time or not'
	)

		# dataset name (kitti/helipr)
	dataset_name = launch.actions.DeclareLaunchArgument(
		name='dataset_name',
		default_value="kitti",
		description='Name of the dataset (KITTI, HeLIPR, ...). Use "live" for ros node mode'
	)

		# point cloud topic (only needed for ros node mode)
	point_cloud_topic = launch.actions.DeclareLaunchArgument(
		name="point_cloud_topic",
		default_value="/point_cloud",
		description="Point cloud topic to subscribe to"
	)

	# set to True if you want to create your own training dataset
	dataset_gen = launch.actions.DeclareLaunchArgument(
		name="dataset_gen",
		default_value='False',
		description="Generate image localization dataset",
	)

	# path to the dataset
	dataset_path = launch.actions.DeclareLaunchArgument(
		name="dataset_path",
		default_value='',
		description="Path to the dataset",
	)

	# dataset sequence (00, 01, etc for Kitti, bridge, town, roundabout for HeLiPR)
	sequence = launch.actions.DeclareLaunchArgument(
		name="sequence",
		default_value="00",
		description="Selected sequence of the dataset",
	)

	# dataset sensor (LiDAR type for HeLIPR: "Ouster", "Aeva", "Avia")
	sensor = launch.actions.DeclareLaunchArgument(
		name="sensor",
		default_value="velodyne",
		description="Selected sensor of the dataset",
	)

	# groundgrid config file
	config_file = launch.actions.DeclareLaunchArgument(
		name="config_file",
		default_value="kitti.yaml",
		description="Groundgrid sensor specific config file",
	)

	# odometry poses file: Txt file with kiss-icp poses in Kitti format
	odom_poses =launch.actions.DeclareLaunchArgument(
		name="poses_file",
		default_value='',
		description="Odometry poses file",
	)

	# ground truth poses file: Txt file with ground truth poses in Kitti format (optional)
	gt_poses = launch.actions.DeclareLaunchArgument(
		name="gt_poses_file",
		default_value='',
		description="Ground truth poses file",
	)

	# delimiter for the pose files 
	delim = launch.actions.DeclareLaunchArgument(
		name="poses_file_delimiter",
		default_value="' '",
		description="Odometry poses file delimiter",
	)

	# write debug images to /tmp/quatro/
	debug_imgs = launch.actions.DeclareLaunchArgument(
		name="debug_imgs",
		default_value='False',
		description="Write debug query and map images to /tmp/quatro",
	)

	# visualize matchings
	visualize = launch.actions.DeclareLaunchArgument(
		name="visualize",
		default_value='False',
		description="Visualize matchings using OpenCV",
	)

	# cap the processing frame rate to the specified frequency (0 = unlimited)
	frame_rate_cap = launch.actions.DeclareLaunchArgument(
		name="frame_rate_cap",
		default_value='0.0',
		description="Cap the processing frame rate to the specified frequency (0 = unlimited)",
	)

	# text file logging of all matchings
	log_matchings = launch.actions.DeclareLaunchArgument(
		name="log_all_matchings",
		default_value='False',
		description="Log all matchings to a text file",
	)
	
	# 2d projection of odometry poses
	project_2d = launch.actions.DeclareLaunchArgument(
		name="project_2d",
		default_value='False',
		description="Project odometry poses onto the xy-plane",
	)

	# inlier confidence factor
	correction_factor = launch.actions.DeclareLaunchArgument(
		name="correction_factor",
		default_value='15.0',
		description="How many inliers correspond to full matching confidence",
	)

	# geotiff map path
	map_file = launch.actions.DeclareLaunchArgument(
		name="map_file",
		default_value='',
		description="Path to the geotiff prior map file",
	)

	# map odom transform (identity transform)
	map_odom_tf = launch_ros.actions.Node(
		package='tf2_ros',
		executable='static_transform_publisher',
		name='map_odom_tf_pub',
		arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'])

	opfunc = launch.actions.OpaqueFunction(function = setup)
	launch_description = launch.LaunchDescription([
		sim_time_arg,
		dataset_name,
		point_cloud_topic,
		dataset_gen,
		dataset_path,
		sequence,
		sensor,
		config_file,
		odom_poses,
		gt_poses,
		delim,
		debug_imgs,
		visualize,
		frame_rate_cap,
		log_matchings,
		project_2d,
		correction_factor,
		map_file,
		map_odom_tf,
	])

	launch_description.add_action(opfunc)
	return launch_description

if __name__ == '__main__':
	generate_launch_description()
