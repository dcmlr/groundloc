// Copyright 2025 Dahlem Center for Machine Learning and Robotics, Freie Universität Berlin
// CC BY-NC-SA 4.0
#ifndef QUATRO_NODE_H
#define QUATRO_NODE_H

// torch
#include <torch/csrc/api/include/torch/nn.h>
#include <torch/script.h> // One-stop header.

// gdal
#include <gdal/gdal_priv.h>

#include <image_transport/publisher.hpp>
#include <rclcpp/node.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/point_types.h>
#include "quatro.hpp"

// ros tf
#include <tf2_ros/transform_listener.h>
#include <tf2/convert.h>
#include <tf2_ros/buffer.h>
#include <tf2/transform_datatypes.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using PointType = pcl::PointXYZ;

namespace quatro_node{

class QuatroNode : public rclcpp::Node{
public:
  QuatroNode(const rclcpp::NodeOptions & node_options = rclcpp::NodeOptions());
  virtual ~QuatroNode();
  rcl_interfaces::msg::SetParametersResult parametersCallback(const std::vector<rclcpp::Parameter> &parameters);
  void setParams(
            double noise_bound_of_each_measurement, double square_of_the_ratio_btw_noise_and_noise_bound,
            double estimating_scale, int num_max_iter, double control_parameter_for_gnc,
            double rot_cost_thr, const std::string& reg_type_name, Quatro<PointType, PointType>::Params &params);
  void callbackBEV(const sensor_msgs::msg::Image::ConstSharedPtr& image);

protected:
  at::Tensor infer_descriptors(const cv::Mat& img);
  std::vector<std::tuple<int, int> > get_keypoints(const at::Tensor& result, const cv::Mat& mask, const std::string& debug_image_postfix, 
                                                   const int border = 12, const int add_keypoints_area = 0);
  void publish_debug_descriptor_image(const std::vector<float>& data, const std::string& path) const;

  image_transport::Subscriber imageSubscriber;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr positionDifferencePublisher;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr posCorrectionPublisher;

  OnSetParametersCallbackHandle::SharedPtr mCallbackHandle;

  // TensorScript stuff
  torch::jit::script::Module module;

  // GDAL
  GDALDatasetUniquePtr dataset;
  GDALRasterBand* band1, *band2, *band3;
  std::vector<float*> band1_data;
  std::vector<float*> band2_data;
  std::vector<float*> band3_data;
  double inv_transform[6];
  std::string filename;

  // tf stuff
  tf2_ros::Buffer mTfBuffer;
  tf2_ros::TransformListener mTfListener;

  // ros publishers
  image_transport::Publisher query_rela_pub, query_repe_pub, query_bev_pub,
                            tgt_rela_pub, tgt_repe_pub, tgt_bev_pub, matchings_img_pub;

  const size_t size = 192; // image crop size
  const float reliability_thresh = 0.001f;
  const float repeatability_thresh = 0.001f;
  size_t max_keypoints = 2000;
  float max_feat_dist = 0.6f;
  size_t border = 12;
  const float resolution = 0.33f;
  double inlier_fac = 25.0; 
  double speed_fac = 57.295779;

  // Quatro parameters
  bool estimating_scale = false;
  double noise_bound = 0.5;
  double noise_bound_coeff = 0.99;
  double gnc_factor = 1.39;
  double rot_cost_diff_thr = 0.0001;
  size_t num_max_iter = 100;

  // keep track of offsets
  double offset_x = 0.0;
  double offset_y = 0.0;
  double offset_yaw = 0.0;
  size_t search_radius = size;

  // Parameters
  bool writeDebugImgs = false;
  bool use_cuda = true;
  bool visualize = false;
  bool log_matchings = false;
  std::string model_file;

  // debug logging csv file for logging of matchings
  std::ofstream log_file;
};
}

#endif
