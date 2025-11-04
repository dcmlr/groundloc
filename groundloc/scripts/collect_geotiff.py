# Copyright 2025 Dahlem Center for Machine Learning and Robotics, Freie Universität Berlin
# CC BY-NC-SA 4.0
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
import math
import time
import signal
import sys

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, PointStamped
from nav_msgs.msg import Odometry
import tf2_geometry_msgs
import tf2_ros
#import tf
from tf_transformations import quaternion_from_euler, euler_from_quaternion
import numpy as np, cv2
from cv_bridge import CvBridge, CvBridgeError
import rasterio
from rasterio.transform import Affine


class CollectGeoTiff(Node):
    def __init__(self):
        super().__init__('collect_geotiff')
        self.minx = 11111110
        self.miny = 11111110
        self.maxx = 0
        self.maxy = 0
        self.offset_x = 0
        self.offset_y = 0
        self.lastPos = PointStamped()
        self.resolution = 0.33
        self.bridge = CvBridge()
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer, self)
        self.publisher = self.create_publisher(Odometry, '/groundloc/offset', qos_profile_system_default)
        self.subscription = self.create_subscription(Image,
                                 '/groundloc/groundgrid/grid_map_cv_normals_z',
                                 self.callback_intensitymap,
                                 qos_profile_system_default) # use reliable for offline processing, since best effort leads to data loss
        
        self.geotiff_path = 'geotiff.tiff' 
        self.intensity = np.zeros((45100,45100), dtype=np.float32) # 10km^2 with 0.3cm^2 per cell resolution
        self.slope = np.zeros(self.intensity.shape, dtype=np.float32) 
        self.variance = np.zeros(self.intensity.shape, dtype=np.float32) # for averaging
        self.occupied_int = np.ones(self.intensity.shape, dtype=np.float32) # for averaging
        self.occupied_slope = np.ones(self.intensity.shape, dtype=np.float32) # for averaging
        self.occupied_var = np.ones(self.intensity.shape, dtype=np.float32) # for averaging

        print('init finished')
        self.publisher.publish(Odometry()) # let groundgrid know that we're ready


    def callback_intensitymap(self, img):
        start = time.time()
        center_frame_id = 'base_link'

        im = self.bridge.imgmsg_to_cv2(img, desired_encoding='32FC3')

        pos_x = float(img.header.frame_id[:img.header.frame_id.index('_')])
        pos_y = float(img.header.frame_id[img.header.frame_id.index('_')+1:])

        now = collect_geotiff.get_clock().now()
        utm_pos = collect_geotiff.tfBuffer.lookup_transform('odom', 'utm', now)
        header = Header()
        header.stamp.sec = now.seconds_nanoseconds()[0]
        header.stamp.nanosec = now.seconds_nanoseconds()[1]
        ps = PointStamped()
        ps.header = header
        ps.point.x = pos_x
        ps.point.y = pos_y 
        ps.header.frame_id = 'odom'
        utmps = tf2_geometry_msgs.do_transform_point(ps, utm_pos)

        if (pos_x-self.lastPos.point.x)**2 + (pos_y-self.lastPos.point.y)**2 < 0.33: # we have moved less than 33cm -> don't update
            self.publisher.publish(Odometry()) # let groundgrid know that we're ready
            return
        
        self.lastPos.point.x = pos_x
        self.lastPos.point.y = pos_y

        im = cv2.flip(im, 0) 
        im = np.swapaxes(im, 0, 1)
        im = cv2.flip(im, 0)

        x = pos_x/self.resolution + self.intensity.shape[0]/2
        y = pos_y/self.resolution + self.intensity.shape[1]/2

        if self.offset_x == 0 and self.offset_y == 0:
            self.offset_x = math.floor(x - self.intensity.shape[0]/2)
            self.offset_y = math.floor(y - self.intensity.shape[1]/2)

        x -= self.offset_x
        y -= self.offset_y

        self.minx = np.int64((min(self.minx, round(x-im.shape[0]/2))))
        self.maxx = np.int64((max(self.maxx, round(x+im.shape[0]/2))))
        self.miny = np.int64((min(self.miny, round(y-im.shape[1]/2))))
        self.maxy = np.int64((max(self.maxy, round(y+im.shape[1]/2))))

        if self.minx < 0 or self.maxx > self.intensity.shape[0] or self.miny < 0 or self.maxy > self.intensity.shape[1]:
            print(f"out of bounds: minx {self.minx} maxx {self.maxx} miny {self.miny} maxy {self.maxy} data {self.intensity.shape}")

        x1_idx = np.int32((y-im.shape[0]/2.0) )
        x2_idx = np.int32((y+im.shape[0]/2.0) )
        y1_idx = np.int32((x-im.shape[1]/2.0) )
        y2_idx = np.int32((x+im.shape[1]/2.0) )
        w_chan = 2 # image channel used for weights
        intensity_update = im[0:364,0:364,0]
        intensity_update = np.maximum(np.zeros(intensity_update.shape), intensity_update)
        intensity_update = np.minimum(np.ones(intensity_update.shape), intensity_update)
        normals_update = im[:,:,1]
        normals_update = np.maximum(np.zeros(normals_update.shape), normals_update) 
        normals_update  = np.minimum(np.ones(normals_update.shape), normals_update)
        variance_update = im[:,:,2]
        variance_update = np.maximum(np.zeros(variance_update.shape), variance_update) 
        variance_update = np.minimum(np.ones(variance_update.shape), variance_update)
        var_update = np.nan_to_num(1.0/np.maximum(variance_update, 0.001))
        self.occupied_int[x1_idx:x2_idx,y1_idx:y2_idx] += np.ceil(intensity_update) * var_update
        self.occupied_slope[x1_idx:x2_idx,y1_idx:y2_idx] += np.ceil(normals_update)
        self.occupied_var[x1_idx:x2_idx,y1_idx:y2_idx] += np.ceil(variance_update) # binary threshholding
        self.intensity[x1_idx:x2_idx, y1_idx:y2_idx] += intensity_update * var_update
        self.slope[x1_idx:x2_idx,y1_idx:y2_idx] += normals_update # slope
        self.variance[x1_idx:x2_idx,y1_idx:y2_idx] += variance_update

        self.publisher.publish(Odometry()) # let groundgrid know that we're ready
        return

def shutdown_handler(sig, frame):
    minxm = (collect_geotiff.minx - collect_geotiff.intensity.shape[0]/2) * collect_geotiff.resolution
    minym = (collect_geotiff.maxy - collect_geotiff.intensity.shape[1]/2) * collect_geotiff.resolution

    now = collect_geotiff.get_clock().now()
    utm_pos = collect_geotiff.tfBuffer.lookup_transform('odom', 'utm', now)
    header = Header()
    header.stamp.sec = now.seconds_nanoseconds()[0]
    header.stamp.nanosec = now.seconds_nanoseconds()[1]
    ps = PointStamped()
    ps.header = header
    ps.point.x = minxm + collect_geotiff.offset_x * collect_geotiff.resolution
    ps.point.y = minym + collect_geotiff.offset_y * collect_geotiff.resolution
    ps.header.frame_id = 'odom'
    utmps = tf2_geometry_msgs.do_transform_point(ps, utm_pos)
    rotx = utmps.point.x
    roty = utmps.point.y

    transform = Affine.translation(rotx + collect_geotiff.resolution,roty + collect_geotiff.resolution) * Affine.scale(collect_geotiff.resolution, -collect_geotiff.resolution)

    collect_geotiff.intensity /= collect_geotiff.occupied_int # average
    collect_geotiff.slope /= collect_geotiff.occupied_slope# average
    collect_geotiff.variance /= collect_geotiff.occupied_var# average

    collect_geotiff.occupied_int = None
    collect_geotiff.occupied_slope = None
    collect_geotiff.occupied_var = None

    data = [np.flipud(collect_geotiff.intensity[collect_geotiff.miny:collect_geotiff.maxy,collect_geotiff.minx:collect_geotiff.maxx] * 255),
            np.flipud(collect_geotiff.slope[collect_geotiff.miny:collect_geotiff.maxy,collect_geotiff.minx:collect_geotiff.maxx] * 255),
            np.flipud(collect_geotiff.variance[collect_geotiff.miny:collect_geotiff.maxy,collect_geotiff.minx:collect_geotiff.maxx] * 255)]
    data = np.ceil(data)

    # hardcoded utm zone
    print(f"writing geotiff of size {collect_geotiff.maxx-collect_geotiff.minx} x {collect_geotiff.maxy-collect_geotiff.miny} to disk")
    with rasterio.open(
            "/tmp/map.tif",
            mode="w",
            driver="GTiff",
            height=collect_geotiff.maxy-collect_geotiff.miny,
            width=collect_geotiff.maxx-collect_geotiff.minx,
            count=3,
            dtype=np.uint8,
            crs=rasterio.crs.CRS.from_epsg(32652),
            transform=transform,
            nodata=0,
            compress="ZSTD"
    ) as new_dataset:
            new_dataset.write(data[0].astype(np.uint8), 1)
            new_dataset.write(data[1].astype(np.uint8), 2)
            new_dataset.write(data[2].astype(np.uint8), 3)

    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, shutdown_handler)
    rclpy.init(args=None)
    collect_geotiff = CollectGeoTiff()
    try:
        rclpy.spin(collect_geotiff)
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
