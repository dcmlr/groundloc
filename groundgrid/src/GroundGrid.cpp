/*
Copyright 2025 Dahlem Center for Machine Learning and Robotics, Freie Universität Berlin

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions
and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or other materials provided
with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to
endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <groundgrid/GroundGrid.hpp>

#include <chrono>
#include <sstream> // stringstream

// Ros package for package path resolution
//#include <ros/package.h>
#include <ament_index_cpp/get_package_prefix.hpp>

// Grid map
#include <grid_map_cv/GridMapCvConverter.hpp>
#include <grid_map_core/GridMapMath.hpp>

// OpenCv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>

// Tf
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

using namespace groundgrid;

GroundGrid::GroundGrid(rclcpp::Clock::SharedPtr clock) : mTfBuffer(clock), mTf2_listener(mTfBuffer)
{}

GroundGrid::~GroundGrid() {
}

void GroundGrid::onInit()
{
}


void GroundGrid::init(const nav_msgs::msg::Odometry::ConstSharedPtr &inOdom)
{
    geometry_msgs::msg::PoseWithCovarianceStamped odomPose, utmPose;

    mMap_ptr = std::make_shared<grid_map::GridMap, const std::vector< std::string >>({"points", "ground", "intensity", "groundpointcount", 
        "groundpatch", "minGroundHeight", "maxGroundHeight", "groundpatch_avg", "variance_avg", "points_accumulated", "normals_avg"}); //TODO: consolidate layers
    grid_map::GridMap& map = *mMap_ptr;
    map.setFrameId("odom");
    map.setGeometry(grid_map::Length(mDimension, mDimension), mResolution, grid_map::Position(inOdom->pose.pose.position.x,inOdom->pose.pose.position.y));
    RCLCPP_INFO(mLogger, "Created map with size %f x %f m (%i x %i cells).",
             map.getLength().x(), map.getLength().y(),
             map.getSize()(0), map.getSize()(1));

    map["points_accumulated"].setConstant(0.0); // not deleted every frame
    map["intensity"].setConstant(0.0);
    map["groundpatch_avg"].setConstant(0.0);
    map["variance_avg"].setConstant(0.0);
    map["normals_avg"].setConstant(0.0);
    map["groundpointcount"].setConstant(0.0);
    map["minGroundHeight"].setConstant(500.0);
    map["maxGroundHeight"].setConstant(-500.0);

    odomPose.pose = inOdom->pose;
    odomPose.header = inOdom->header;

    std::vector<grid_map::BufferRegion> damage;
    map.move(grid_map::Position(odomPose.pose.pose.position.x, odomPose.pose.pose.position.y), damage);
    grid_map::BufferRegion region(grid_map::Index(0,0), map.getSize(), grid_map::BufferRegion::Quadrant(0));

    map["points"].setZero();
    map["ground"].setConstant(inOdom->pose.pose.position.z);
    map["groundpatch"].setConstant(0.0001);

    mLastPose = odomPose;
}


std::shared_ptr<grid_map::GridMap> GroundGrid::update(const nav_msgs::msg::Odometry::ConstSharedPtr &inOdom)
{
    static unsigned int seq = 0;

    if(!mMap_ptr){
        init(inOdom);
        return mMap_ptr;
    }

    auto start = std::chrono::steady_clock::now();
    geometry_msgs::msg::PoseWithCovarianceStamped odomPose, utmPose;
    grid_map::GridMap& map = *mMap_ptr;

    geometry_msgs::msg::PoseWithCovarianceStamped poseDiff;
    poseDiff.pose.pose.position.x = inOdom->pose.pose.position.x - mLastPose.pose.pose.position.x;
    poseDiff.pose.pose.position.y = inOdom->pose.pose.position.y - mLastPose.pose.pose.position.y;
    std::vector<grid_map::BufferRegion> damage;
    map.move(grid_map::Position(inOdom->pose.pose.position.x, inOdom->pose.pose.position.y), damage);

    // static so if the new transform is not yet available, we can use the last one
    static geometry_msgs::msg::TransformStamped base_to_odom;

    base_to_odom.header = inOdom->header;
    base_to_odom.child_frame_id = "base_link";
    base_to_odom.transform.translation.x = inOdom->pose.pose.position.x;
    base_to_odom.transform.translation.y = inOdom->pose.pose.position.y;
    base_to_odom.transform.translation.z = inOdom->pose.pose.position.z;
    base_to_odom.transform.rotation = inOdom->pose.pose.orientation;

    geometry_msgs::msg::PointStamped ps;
    ps.header = inOdom->header;
    ps.header.frame_id = "odom";
    grid_map::Position pos;

    for(auto region : damage){
        for(auto it = grid_map::SubmapIterator(map, region); !it.isPastEnd(); ++it){
            auto idx = *it;

    	      map.getPosition(idx, pos);
        	  ps.point.x = pos(0);
    	      ps.point.y = pos(1);
    	      ps.point.z = 0;
            //tf2::doTransform(ps, ps, base_to_odom);
            //map.at("ground", idx) = ps.point.z; //TODO: this might be problematic in situations with high roll/pitch drift
            map.at("ground", idx) = inOdom->pose.pose.position.z;//ps.point.z; // set it very low so it will be corrected by interpolation
            map.at("groundpatch", idx) = 0.0;
            map.at("groundpatch_avg", idx) = 0.0;
            map.at("variance_avg", idx) = 0.0;
            map.at("normals_avg", idx) = 0.0;
            map.at("points_accumulated", idx) = 0.0;
            map.at("groundpointcount", idx) = 0.0;
            map.at("intensity", idx) = 0.0;
        }
    }

    // We havent moved so we have nothing to do
    if(damage.empty())
        return mMap_ptr;

    mLastPose.pose = inOdom->pose;
    mLastPose.header = inOdom->header;

    map.convertToDefaultStartIndex();
    auto end = std::chrono::steady_clock::now();
    RCLCPP_DEBUG_STREAM(mLogger, "total " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms");
    return mMap_ptr;
}

