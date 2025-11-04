// Copyright 2025 Dahlem Center for Machine Learning and Robotics, Freie Universität Berlin
// CC BY-NC-SA 4.0

#include <rmw/qos_profiles.h>
#include <cmath>
#include <image_transport/image_transport.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <quatro_node_cuda.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/features2d.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>

#include <rclcpp/qos.hpp>


namespace pcl{
    struct PointXYZDesc{
        PCL_ADD_POINT4D;
        float descriptor[128];
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };//EIGEN_ALIGN16;
}
#include <pcl/impl/point_types.hpp>  // Include struct definitions
POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::PointXYZDesc,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float[128], descriptor, descriptor)
)


namespace quatro_node {

    QuatroNode::QuatroNode(const rclcpp::NodeOptions & node_options):
        rclcpp::Node("quatro_node", node_options),
        mTfBuffer(get_clock()),
        mTfListener(mTfBuffer)
        {
            auto param_use_cuda = rcl_interfaces::msg::ParameterDescriptor{};
            param_use_cuda.description = "Use CUDA to run the model on the graphics hardware instead of the CPU";
            use_cuda = declare_parameter<bool>("bev_registration/use_cuda", true, param_use_cuda);
            RCLCPP_INFO_STREAM(get_logger(), "Using " << (use_cuda ? "CUDA " : "CPU ") << "for inference");
            int gpu_id = 0;
            auto param_debug_imgs = rcl_interfaces::msg::ParameterDescriptor{};
            param_debug_imgs.description = "Write debug images to the disk (impacts run-time performance)";
            writeDebugImgs = declare_parameter<bool>("bev_registration/debug_imgs", false, param_debug_imgs);
            auto param_visualize = rcl_interfaces::msg::ParameterDescriptor{};
            param_visualize.description = "Real-time visualization (impacts run-time performance)";
            visualize = declare_parameter<bool>("bev_registration/visualize", false, param_visualize);
            auto param_log_matchings = rcl_interfaces::msg::ParameterDescriptor{};
            param_log_matchings.description = "Write all keypoint matchings to csv file (impacts run-time performance)";
            log_matchings = declare_parameter<bool>("bev_registration/log_all_matchings", false, param_log_matchings);
            auto param_model = rcl_interfaces::msg::ParameterDescriptor{};
            param_model.description = "Path to the model file";
            model_file = declare_parameter<std::string>("bev_registration/model_file", "models/r2d2_opt_single.pt", param_model);

            try {
                // Deserialize the ScriptModule from a file using torch::jit::load().
                RCLCPP_INFO_STREAM(get_logger(), "Loading model " << model_file);
                module = torch::jit::load(model_file);
                module.to(use_cuda ? at::kCUDA : at::kCPU);

                // model warmup
               	torch::List<torch::Tensor> list;
           	    std::vector<torch::jit::IValue> inputs(0);
                RCLCPP_INFO(get_logger(), "Model warmup...");
           	    for(int i=0; i<10; ++i){
                   	list.push_back(torch::unsqueeze(torch::rand({3, static_cast<int>(size), static_cast<int>(size)}), 0)
                   	               .to(use_cuda ? at::kCUDA : at::kCPU));
               	    inputs.emplace_back(list);
                    module(inputs);
                    list.clear();
                    inputs.clear();
                }
            }
            catch (const c10::Error& e) {
                std::cerr << "error loading the model " << e.msg() << std::endl;
            }

            auto param_correction_factor = rcl_interfaces::msg::ParameterDescriptor{};
            param_correction_factor.description = "Pose estimate correction factor";
            inlier_fac = declare_parameter<double>("bev_registration/correction_factor", 25.0, param_correction_factor);
            auto param_yaw_correction_factor = rcl_interfaces::msg::ParameterDescriptor{};
            param_yaw_correction_factor.description = "Pose estimate yaw correction factor";
            speed_fac = declare_parameter<double>("bev_registration/yaw_correction_factor", 180.0/M_PI, param_yaw_correction_factor);
            auto param_geotiff_path = rcl_interfaces::msg::ParameterDescriptor{};
            param_geotiff_path.description = "Path to the GeoTiff map";
            filename = declare_parameter<std::string>("bev_registration/geotiff_path", "GeoTiff map path argument not set", param_geotiff_path);
            RCLCPP_INFO_STREAM(get_logger(), "Reading map: " << filename);

            // Open Geotiff
            GDALAllRegister();
            dataset = GDALDatasetUniquePtr(GDALDataset::FromHandle(GDALOpen(filename.c_str(), GA_ReadOnly)));
            if(dataset == nullptr){
                RCLCPP_ERROR_STREAM(get_logger(), "Failed to open raster dataset: " << filename.c_str());
            }
            else{
                band1 = dataset->GetRasterBand(1);
                band2 = dataset->GetRasterBand(2);
                band3 = dataset->GetRasterBand(3);
                band1_data.reserve(band1->GetYSize());
                band2_data.reserve(band2->GetYSize());
                band3_data.reserve(band3->GetYSize());
                for(int i=0; i<band1->GetYSize(); ++i){
                    band1_data[i] = new float[band1->GetXSize()];
                    band2_data[i] = new float[band2->GetXSize()];
                    band3_data[i] = new float[band3->GetXSize()];
                    if(band1->RasterIO(GF_Read, 0, i, band1->GetXSize(), 1, band1_data[i],
                                       band1->GetXSize(), 1, GDT_Float32, 0, 0) == CE_Failure)
                        RCLCPP_ERROR(get_logger(), "Failed to read geotiff band1");
                    if(band2->RasterIO(GF_Read, 0, i, band2->GetXSize(), 1, band2_data[i],
                                       band2->GetXSize(), 1, GDT_Float32, 0, 0) == CE_Failure)
                        RCLCPP_ERROR(get_logger(), "Failed to read geotiff band2");
                    if(band3->RasterIO(GF_Read, 0, i, band3->GetXSize(), 1, band3_data[i],
                                       band3->GetXSize(), 1, GDT_Float32, 0, 0) == CE_Failure)
                        RCLCPP_ERROR(get_logger(), "Failed to read geotiff band3");
                }

                double transform[6];
                dataset->GetGeoTransform(transform);
                if(!GDALInvGeoTransform(transform, inv_transform)){
                    RCLCPP_ERROR(get_logger(), "Failed to invert geo transform!");
                }
            }

            positionDifferencePublisher = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/groundloc/position_difference", rclcpp::SensorDataQoS());
            posCorrectionPublisher = create_publisher<nav_msgs::msg::Odometry>("/groundloc/offset", rclcpp::SystemDefaultsQoS());

            if(visualize){
                cv::namedWindow("matches", cv::WINDOW_NORMAL | cv::WINDOW_GUI_NORMAL);
                cv::resizeWindow("matches", cv::Size(size*4,size*2));
            }
            if(log_matchings){
                log_file.open("all_matchings.csv");
                log_file << "frame_number,matching_number,x_src,y_src,x_tgt,y_tgt,descriptor_dist,transform_x,transform_y,transform_yaw" << std::endl;
            }

            RCLCPP_INFO(get_logger(), "Initialization complete.");

            // signal groundloc that we are ready
            posCorrectionPublisher->publish(nav_msgs::msg::Odometry());
    }

    QuatroNode::~QuatroNode()
    {
        for(int i=0; i<band1_data.size(); ++i){
            delete[] band1_data[i];
            delete[] band2_data[i];
            delete[] band3_data[i];
        }
        dataset->Close();
        log_file.close();
    }
    
    void QuatroNode::callbackBEV(const sensor_msgs::msg::Image::ConstSharedPtr& image){
        // inference
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
        static float avg_time = 0.0f;
        static size_t count = 0;
        static size_t received = 0;
        static geometry_msgs::msg::PoseStamped lastPose;
        ++received;
        static Quatro<PointType, PointType> quatro;
        Quatro<PointType, PointType>::Params params;
        setParams(noise_bound, noise_bound_coeff,
                  estimating_scale, num_max_iter, gnc_factor, rot_cost_diff_thr, "Quatro", params);
        quatro.reset(params);

        // Initialize image pubs
        if(visualize && received == 1){
            image_transport::ImageTransport it(shared_from_this());
            query_rela_pub = it.advertise("/groundloc/matching/reliability_query", rmw_qos_profile_sensor_data); 
            tgt_rela_pub = it.advertise("/groundloc/matching/reliability_target", rmw_qos_profile_sensor_data);
            query_repe_pub = it.advertise("/groundloc/matching/repeatability_query", rmw_qos_profile_sensor_data);
            tgt_repe_pub = it.advertise("/groundloc/matching/repeatability_target", rmw_qos_profile_sensor_data);
            query_bev_pub = it.advertise("/groundloc/matching/bev_query", rmw_qos_profile_sensor_data);
            tgt_bev_pub = it.advertise("/groundloc/matching/bev_target", rmw_qos_profile_sensor_data);
        }

        // message to opencv
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(*image.get(), image.get()->encoding);
        }
        catch (cv_bridge::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }


        // get transform between odom and map
        std::string center_frame_id = "base_link";
        geometry_msgs::msg::TransformStamped transformStamped_map, transformStamped_odom, transformStamped_utm;
        geometry_msgs::msg::PoseStamped pose_map, pose_odom, pose_utm;
        pose_map.header.frame_id = center_frame_id;
        pose_map.header.stamp = image.get()->header.stamp;
        pose_odom.header.frame_id = center_frame_id;
        pose_odom.header.stamp = image.get()->header.stamp;
        pose_utm.header.frame_id = center_frame_id;
        pose_utm.header.stamp = image.get()->header.stamp;
        pose_odom.pose.orientation.w = 1.0;
        pose_map.pose.orientation.w = 1.0;
        pose_utm.pose.orientation.w = 1.0;

        size_t idx = image.get()->header.frame_id.find("_");
        float x = std::stof(image.get()->header.frame_id.substr(0,idx-1));
        float y = std::stof(image.get()->header.frame_id.substr(idx+1));
        if(!count){
            // First pose received, initialize with it
            lastPose.pose.position.x = x;
            lastPose.pose.position.y = y;
        }
        const float x_diff = x - lastPose.pose.position.x;
        const float y_diff = y - lastPose.pose.position.y;
        const float x_diff_rotated = std::cos(-offset_yaw) * x_diff - std::sin(-offset_yaw) * y_diff;
        const float y_diff_rotated = std::sin(-offset_yaw) * x_diff + std::cos(-offset_yaw) * y_diff;
        try{
            mTfBuffer.canTransform("map", "utm", tf2::TimePointZero);
            transformStamped_utm = mTfBuffer.lookupTransform("map", "utm", tf2::TimePointZero);
        }
        catch (tf2::TransformException &ex) {
            RCLCPP_WARN(get_logger(), "Failed to get transform: %s",ex.what());
        }
        transformStamped_odom.transform.translation.x = x;
        transformStamped_odom.transform.translation.y = y;
        transformStamped_odom.transform.rotation.w = 1.0;
        transformStamped_map = transformStamped_odom; // only works iff map == odom
        tf2::Quaternion q_odom_map;
        tf2::fromMsg(transformStamped_map.transform.rotation, q_odom_map);
        tf2::Matrix3x3 m_odom_map(q_odom_map);
        double roll_odom_map, pitch_odom_map, yaw_odom_map;
        m_odom_map.getRPY(roll_odom_map, pitch_odom_map, yaw_odom_map);
        q_odom_map.setRPY(roll_odom_map, pitch_odom_map, yaw_odom_map - offset_yaw);
        transformStamped_map.transform.translation.x = lastPose.pose.position.x + x_diff_rotated - offset_x;
        transformStamped_map.transform.translation.y = lastPose.pose.position.y + y_diff_rotated - offset_y;
        tf2::doTransform(pose_map, pose_map, transformStamped_map);
        tf2::doTransform(pose_odom, pose_odom, transformStamped_odom);
        tf2::doTransform(pose_map, pose_utm, transformStamped_utm);
        tf2::Quaternion q_map, q_odom, q_utm;
        tf2::fromMsg(pose_map.pose.orientation, q_map);
        tf2::fromMsg(pose_odom.pose.orientation, q_odom);
        tf2::fromMsg(pose_utm.pose.orientation, q_utm);
        tf2::Matrix3x3 m_map(q_map);
        tf2::Matrix3x3 m_odom(q_odom);
        tf2::Matrix3x3 m_utm(q_utm);
        double roll_map, pitch_map, yaw_map;
        double roll_odom, pitch_odom, yaw_odom;
        double roll_utm, pitch_utm, yaw_utm;
        m_map.getRPY(roll_map, pitch_map, yaw_map);
        m_odom.getRPY(roll_odom, pitch_odom, yaw_odom);
        m_utm.getRPY(roll_utm, pitch_utm, yaw_utm);
        cv::Point center = cv::Point(cv_ptr->image.size().width/2, cv_ptr->image.size().height/2);
        cv::Mat M = cv::getRotationMatrix2D(center, yaw_utm * 180.0/M_PI  - offset_yaw * 180.0/M_PI - 90, 1.0f);
        cv::Mat& img = cv_ptr->image;
        cv::warpAffine(img, img, M, img.size());
        const auto crop_start_x = image->width/2 - size/2;
        const auto crop_start_y = image->height/2 - size/2;
        const auto crop_end_x = image->width/2 + size/2;
        const auto crop_end_y = image->height/2 + size/2;
        cv::Mat crop = cv_ptr->image(cv::Range(crop_start_x,crop_end_x), cv::Range(crop_start_y,crop_end_y));
        lastPose.pose.position.x = x;
        lastPose.pose.position.y = y;

        // crop geotiff
        float img2_array[3][size][size];
        x = pose_map.pose.position.x;
        y = pose_map.pose.position.y;
        offset_x -= x_diff_rotated - x_diff;
        offset_y -= y_diff_rotated - y_diff;
        int center_idx_y = static_cast<unsigned int>(floor(inv_transform[0] + inv_transform[1] * x
                                                  + inv_transform[2] * y));
        int center_idx_x = static_cast<unsigned int>(floor(inv_transform[3] + inv_transform[4] * x 
                                                  + inv_transform[5] * y));

        if(center_idx_x < size/2 || center_idx_y < size/2 || center_idx_x+size/2 > band1->GetXSize() || center_idx_y+size/2 > band1->GetYSize()){
            RCLCPP_DEBUG_STREAM(get_logger(), "requested index out of map bounds: " << center_idx_x << ", " << center_idx_y << " (bounds: " << band1->GetXSize() << ", " << band1->GetYSize() << ")");
            //return;
        }
        for(int i=0; i<size; ++i){
            for(int j=0; j<size; ++j){
                int x_idx = center_idx_x-size/2+i;
                int y_idx = center_idx_y-size/2+j;
                if(x_idx < 0 || y_idx < 0 || x_idx >= band1->GetYSize() || y_idx >= band1->GetXSize()){
                    // out of bounds access requested
                    img2_array[0][i][j] = 0.0f;
                    img2_array[1][i][j] = 0.0f;
                    img2_array[2][i][j] = 0.0f;
                }
                else{
                    img2_array[0][i][j] = band1_data[x_idx][y_idx];
                    img2_array[1][i][j] = band2_data[x_idx][y_idx];
                    img2_array[2][i][j] = band3_data[x_idx][y_idx];
                }
            }
        }

        std::vector<cv::Mat> img1_split;
        cv::split(crop, img1_split);
        cv::Mat img2_1(size, size, CV_32FC1, img2_array[0]);
        cv::Mat img2_2(size, size, CV_32FC1, img2_array[1]);
        cv::Mat img2_3(size, size, CV_32FC1, img2_array[2]);
        cv::merge(img1_split, crop);
        std::vector<cv::Mat> channels;
        channels.push_back(img2_1);
        channels.push_back(img2_2);
        channels.push_back(img2_3);
        cv::Mat img2(size, size, CV_32FC3);
        cv::merge(channels, img2);
        if(visualize){
            cv::Mat query_bev_uint, target_bev_uint;
            crop.convertTo(query_bev_uint, CV_8UC3, 255, 0);
            img2.convertTo(target_bev_uint, CV_8UC3);
            sensor_msgs::msg::Image::SharedPtr msg_query = cv_bridge::CvImage(std_msgs::msg::Header(), "8UC3", query_bev_uint).toImageMsg();
            sensor_msgs::msg::Image::SharedPtr msg_target = cv_bridge::CvImage(std_msgs::msg::Header(), "8UC3", target_bev_uint).toImageMsg();
            query_bev_pub.publish(std::move(msg_query));
            tgt_bev_pub.publish(std::move(msg_target));
        } 
        img2.convertTo(img2, CV_32FC3, 1.0/255.0, 0.0);

        if(writeDebugImgs){
            std::stringstream oss, oss2;
            oss << "/tmp/quatro/" << count << "_tgt_img.exr";
            oss2 << "/tmp/quatro/" << count << "_src_img.exr";
            cv::imwrite(oss.str(), img2);
            cv::imwrite(oss2.str(), crop);
        }

        at::Tensor result = infer_descriptors(crop);
        at::Tensor result2 = infer_descriptors(img2);
        const auto& keypoints1 = get_keypoints(result, crop, "src", border, 2);
        const auto& keypoints2 = get_keypoints(result2, img2, "tgt", border, 2);
        
        at::Tensor output_cpu = result.to(at::kCPU);
        at::Tensor output_cpu2 = result2.to(at::kCPU);
        
        // publish debug images
        if(visualize || writeDebugImgs){
            cv::Mat src_repe = cv::Mat(size, size, CV_32FC1, output_cpu.index({0,128}).data_ptr());
            cv::Mat tgt_repe = cv::Mat(size, size, CV_32FC1, output_cpu2.index({0,128}).data_ptr());
            cv::Mat src_rela = cv::Mat(size, size, CV_32FC1, output_cpu.index({0,129}).data_ptr());
            cv::Mat tgt_rela = cv::Mat(size, size, CV_32FC1, output_cpu2.index({0,129}).data_ptr());
            if(writeDebugImgs){
                cv::imwrite("/tmp/torch_repe_src.exr", src_repe);
                cv::imwrite("/tmp/torch_repe_tgt.exr", tgt_repe);
                cv::imwrite("/tmp/torch_rela_src.exr", src_rela);
                cv::imwrite("/tmp/torch_rela_tgt.exr", tgt_rela);
            }
            if(visualize){
                cv::Mat src_repe_colored, tgt_repe_colored, src_rela_colored, tgt_rela_colored;
                src_repe.convertTo(src_repe_colored, CV_8UC3, 255, 0);
                tgt_repe.convertTo(tgt_repe_colored, CV_8UC3, 255, 0);
                src_rela.convertTo(src_rela_colored, CV_8UC3, 255, 0);
                tgt_rela.convertTo(tgt_rela_colored, CV_8UC3, 255, 0);
                cv::applyColorMap(src_repe_colored, src_repe_colored, cv::COLORMAP_DEEPGREEN);
                cv::applyColorMap(tgt_repe_colored, tgt_repe_colored, cv::COLORMAP_DEEPGREEN);
                cv::applyColorMap(src_rela_colored, src_rela_colored, cv::COLORMAP_DEEPGREEN);
                cv::applyColorMap(tgt_rela_colored, tgt_rela_colored, cv::COLORMAP_DEEPGREEN);
                sensor_msgs::msg::Image::SharedPtr msg_query_repe = cv_bridge::CvImage(std_msgs::msg::Header(), "8UC3", src_repe_colored).toImageMsg();
                sensor_msgs::msg::Image::SharedPtr msg_target_repe = cv_bridge::CvImage(std_msgs::msg::Header(), "8UC3", tgt_repe_colored).toImageMsg();
                sensor_msgs::msg::Image::SharedPtr msg_query_rela = cv_bridge::CvImage(std_msgs::msg::Header(), "8UC3", src_rela_colored).toImageMsg();
                sensor_msgs::msg::Image::SharedPtr msg_target_rela = cv_bridge::CvImage(std_msgs::msg::Header(), "8UC3", tgt_rela_colored).toImageMsg();
                query_repe_pub.publish(std::move(msg_query_repe));
                tgt_repe_pub.publish(std::move(msg_target_repe));
                query_rela_pub.publish(std::move(msg_query_rela));
                tgt_rela_pub.publish(std::move(msg_target_rela));
            }
        }
        output_cpu = output_cpu.contiguous();
        output_cpu2 = output_cpu2.contiguous();
        std::vector<float> v(output_cpu.data_ptr<float>(), output_cpu.data_ptr<float>() + output_cpu.numel());
        std::vector<float> v2(output_cpu2.data_ptr<float>(), output_cpu2.data_ptr<float>() + output_cpu2.numel());

        // descriptor debug imgs
        if(writeDebugImgs || visualize){
            publish_debug_descriptor_image(v, "/tmp/desc_cpp_src.exr");
            publish_debug_descriptor_image(v2, "/tmp/desc_cpp_tgt.exr");
        }
          
        static float error_xy = 0.0f, error_yaw = 0.0f;
        static int successful = 0, total = 0;
        pcl::PointCloud<PointType>::Ptr srcMatched(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr tgtMatched(new pcl::PointCloud<PointType>);


        float* src_descs_arr = new float[130000*128];
        float* tgt_descs_arr = new float[130000*128]; // TODO: dynamic size with vectors
        const int& size_sq = size*size;

        // use only a subset of the size of max keypoints
        std::vector<std::pair<unsigned int, float>> relas_src(keypoints1.size());
        std::vector<std::pair<unsigned int, float>> relas_tgt(keypoints2.size());

        cv::Mat repe_src = cv::Mat(size, size, CV_32FC1, output_cpu.index({0,128}).data_ptr()); 
        cv::Mat repe_tgt = cv::Mat(size, size, CV_32FC1, output_cpu2.index({0,128}).data_ptr());
        cv::Mat rela_src = cv::Mat(size, size, CV_32FC1, output_cpu.index({0,129}).data_ptr()); 
        cv::Mat rela_tgt = cv::Mat(size, size, CV_32FC1, output_cpu2.index({0,129}).data_ptr());

        // order keypoints by repeatability * reliability
        for(size_t i=0; i<keypoints1.size(); ++i){
            relas_src[i] = std::make_pair(i, repe_src.at<float>(std::get<0>(keypoints1[i]), std::get<1>(keypoints1[i]))
                                          * rela_src.at<float>(std::get<0>(keypoints1[i]), std::get<1>(keypoints1[i]))); //TODO: check correct indexing
        }
        for(size_t i=0; i<keypoints2.size(); ++i){
            relas_tgt[i] = std::make_pair(i, repe_tgt.at<float>(std::get<0>(keypoints2[i]), std::get<1>(keypoints2[i]))
                                          * rela_tgt.at<float>(std::get<0>(keypoints2[i]), std::get<1>(keypoints2[i]))); //TODO: check correct indexing
        }
        std::sort(relas_src.begin(), relas_src.end(), [](auto &left, auto &right) {
            return left.second > right.second;
        });
        std::sort(relas_tgt.begin(), relas_tgt.end(), [](auto &left, auto &right) {
            return left.second > right.second;
        });

        for(size_t i=0; i<std::min(keypoints1.size(), max_keypoints); ++i){
            const auto& coord = keypoints1[relas_src[i].first];
            for(size_t j=0; j<128; ++j){
                src_descs_arr[128*i+j] = v[j*size_sq + std::get<0>(coord) * size + std::get<1>(coord)];
            }
        }
        for(size_t i=0; i<std::min(keypoints2.size(), max_keypoints); ++i){
            const auto& coord = keypoints2[relas_tgt[i].first];
            for(size_t j=0; j<128; ++j){
                tgt_descs_arr[128*i+j] = v2[j*size_sq + std::get<0>(coord) * size + std::get<1>(coord)];
            }
        }

        flann::Matrix<float> src_descs(src_descs_arr, std::min(keypoints1.size(), max_keypoints), 128);
        flann::Matrix<float> tgt_descs(tgt_descs_arr, std::min(keypoints2.size(), max_keypoints), 128);
        // No keypoints found...
        if(keypoints1.empty() || keypoints2.empty()){
            std::cout << "No keypoints found!" << std::endl;
            if(posCorrectionPublisher->get_subscription_count() > 0){
                nav_msgs::msg::Odometry corrected_odom;
                corrected_odom.pose.pose.position.x = offset_x;
                corrected_odom.pose.pose.position.y = offset_y;
                tf2::Quaternion q_offset;
                q_offset.setRPY(0, 0, offset_yaw);
                corrected_odom.pose.pose.orientation = tf2::toMsg(q_offset);

                corrected_odom.header.stamp = image.get()->header.stamp;
                corrected_odom.header.frame_id = "map";
                posCorrectionPublisher->publish(corrected_odom);
            }
            delete[] src_descs_arr;
            delete[] tgt_descs_arr;
            return;
        }


        std::chrono::system_clock::time_point before_kps = std::chrono::system_clock::now();
        flann::Index<flann::L2<float> > index(tgt_descs, flann::KDTreeIndexParams(2));
        index.buildIndex();
        int nn = 1;
        flann::Matrix<int> indices(new int[src_descs.rows*nn], src_descs.rows, nn);
        flann::Matrix<float> dists(new float[src_descs.rows*nn], src_descs.rows, nn);
        flann::SearchParams search_params(128);
        search_params.cores = 0;
        index.knnSearch(src_descs, indices, dists, nn, search_params);
        std::chrono::duration<double> keypoint_time = std::chrono::system_clock::now() - before_kps;
        int mid = size/2;
        for(int i=0; i<src_descs.rows; ++i){
            if(dists[i][0] < max_feat_dist){
                const auto& p1 = keypoints1[relas_src[i].first];
                const auto& p2 = keypoints2[relas_tgt[indices[i][0]].first];
                PointType src;
                PointType tgt;
                src.x = std::get<0>(p1)-mid;
                src.y = mid-std::get<1>(p1);
                src.z = dists[i][0]; // dist in z
                tgt.x = std::get<0>(p2)-mid;
                tgt.y = mid-std::get<1>(p2);
                tgt.z = src.z;
                if(std::hypot(src.x-tgt.x,src.y-tgt.y) > search_radius)
                    continue;
                srcMatched->points.push_back(src);
                tgtMatched->points.push_back(tgt);
            }
        }

        // No keypoints found...
        if(srcMatched->points.empty() || tgtMatched->points.empty()){
            std::cout << "No keypoints found!" << std::endl;
        }

        std::chrono::system_clock::time_point before_optim = std::chrono::system_clock::now();
        quatro.setInputSource(srcMatched);
        quatro.setInputTarget(tgtMatched);
        Eigen::Matrix4d output;
        quatro.computeTransformation(output);

        std::chrono::duration<double> sec = std::chrono::system_clock::now() - start;
        ++count;
        avg_time += (sec.count() - avg_time)/count;
        RCLCPP_DEBUG_STREAM(get_logger(), "Avg runtime " << avg_time);

        tf2::Matrix3x3 rot_matrix;
        for (int idx = 0; idx < 12; ++idx){
            int i = idx / 4;
            int j = idx % 4;
            if(j < 3)
                rot_matrix[i][j] = output(idx);
        }
        
        double roll = 0, pitch = 0, yaw = 0;
        rot_matrix.getRPY(roll, pitch, yaw);
        float yaw_deg = yaw*180.0f/M_PI;
        float cur_error_xy = std::sqrt(std::pow(output(12)*resolution, 2.0f) + std::pow(output(13)*resolution, 2.0f));
        if(std::isnan(yaw_deg))
            yaw_deg = 0.0f;
        error_xy += cur_error_xy;

        if(cur_error_xy < search_radius/2.0f * resolution){
            search_radius = std::max(search_radius - 1, static_cast<size_t>(3));
        }
        else{
            search_radius = std::min(search_radius + 3, size);
        }
        int ind_count = quatro.getFinalInliersIndices().size();
        if(writeDebugImgs){
            cv::Mat src_kpts(size, size, CV_8UC1, cv::Scalar(0));
            cv::Mat tgt_kpts(size, size, CV_8UC1, cv::Scalar(0));
            for(const auto& src_pt : srcMatched->points)
                src_kpts.at<unsigned char>(src_pt.x + mid, -src_pt.y + mid) = 255;
            for(const auto& tgt_pt : tgtMatched->points)
                tgt_kpts.at<unsigned char>(tgt_pt.x + mid, -tgt_pt.y + mid) = 255;
            cv::imwrite("/tmp/src_candidates.png", src_kpts);
            cv::imwrite("/tmp/tgt_candidates.png", tgt_kpts);
        }
        if(visualize || log_matchings){
            pcl::PointCloud<PointType> srcMaxCliques;
            pcl::PointCloud<PointType> tgtMaxCliques;
            quatro.getMaxCliques(srcMaxCliques, tgtMaxCliques);
            std::vector<cv::KeyPoint> src_kps, tgt_kps;
            std::vector<std::vector<cv::DMatch>> matches;
            for(int i=0; i<srcMaxCliques.size(); ++i){
                const PointType& src_point = srcMaxCliques[i];
                const PointType& tgt_point = tgtMaxCliques[i];
                std::vector<cv::DMatch> match;
                match.push_back(cv::DMatch(i, i, 1.0f)); 
                matches.push_back(match);
                // we multiply the coordinates because we also increase the image size for better readability
                src_kps.push_back(cv::KeyPoint((mid-src_point.y)*2, (mid+src_point.x)*2, 1));
                tgt_kps.push_back(cv::KeyPoint((mid-tgt_point.y)*2, (mid+tgt_point.x)*2, 1));
                if(log_matchings)
                    log_file << std::fixed << count << "," << i << "," << (src_point.x) * resolution + x << "," << (src_point.y) * resolution + y
                            << "," << (tgt_point.x) * resolution + x << "," << (tgt_point.y) * resolution + y << ","<< src_point.z <<
                            "," << output(13) * resolution << "," << output(12) * resolution << "," << yaw << std::endl; 
            }
            if(visualize){
                cv::Mat matches_img, crop_int, img2_int;
                crop.convertTo(crop_int, CV_8UC3, 255, 0);
                img2.convertTo(img2_int, CV_8UC3, 255, 0);
                // scale images for better readability
                cv::resize(crop_int, crop_int, cv::Size(crop_int.cols * 2,crop_int.rows * 2), 0, 0, CV_INTER_LINEAR);
                cv::resize(img2_int, img2_int, cv::Size(img2_int.cols * 2,img2_int.rows * 2), 0, 0, CV_INTER_LINEAR);
                cv::drawMatches(crop_int, src_kps, img2_int, tgt_kps, matches, matches_img);
                cv::imshow("matches", matches_img);
                cv::waitKey(1);
            }
        }
        // update offset params only if more than 3 inliers
        if(ind_count >= 3){
            const float& speed = std::hypot(x_diff, y_diff);
            const double max_correction_xy = std::max((ind_count)/inlier_fac*speed, 0.0);
            offset_x += std::max(std::min(output(13) * resolution * 0.3, max_correction_xy), -max_correction_xy);
            offset_y += std::max(std::min(output(12) * resolution * 0.3, max_correction_xy), -max_correction_xy);
            //float offset_diff = std::max(std::min(std::asin(output(1)) * 0.3, ind_count/25.0*speed/20.0), -ind_count/25.0*speed/20.0);//TODO aeva test
            float offset_diff = std::max(std::min(-yaw * 0.3, (ind_count)/inlier_fac*speed/speed_fac), -(ind_count)/inlier_fac*speed/speed_fac);
            if(!std::isnan(offset_diff)){ // sometimes quatro gives back NaN as yaw
                offset_yaw += offset_diff;
                if(offset_yaw > M_PI)
                    offset_yaw -= M_2_PI;
                if(offset_yaw < -M_PI)
                    offset_yaw += M_2_PI;
            }
        }
        if(ind_count < 30){
            if(max_keypoints < size_t(2000))
                max_keypoints = std::min(max_keypoints + 100, size_t(2000));
            else
                max_feat_dist = std::min(max_feat_dist + 0.05f, .8f);
        }
        if(ind_count >= 60){
            max_feat_dist = std::max(max_feat_dist - 0.05f, .4f);
            if(max_feat_dist <= .4f)
                max_keypoints = std::max(max_keypoints - 100, size_t(500));
        }
        
      // Publish position difference
        if(positionDifferencePublisher->get_subscription_count() > 0)
        {
            geometry_msgs::msg::Pose offset;
            geometry_msgs::msg::PoseWithCovarianceStamped ps;
            offset.position.x = output(13) * resolution;
            offset.position.y = output(12) * resolution;
            offset.orientation.z = yaw;
            std::array<double, 36> cov;
            for(int i=0; i<36; ++i) cov[i] = 0.0;
            cov[0] = 100.0/(ind_count*ind_count) + 0.33;
            cov[7] = 100.0/(ind_count*ind_count) + 0.33;
            cov[35] = 5.0/(ind_count*ind_count) + 0.02;

            if(ind_count < 1){
                offset.position.x = 0.0;
                offset.position.y = 0.0;
                offset.orientation.z = 0.0;
            }
            ps.header.stamp = image.get()->header.stamp;
            ps.header.frame_id = image.get()->header.frame_id;
            ps.pose.pose = offset;
            ps.pose.covariance = cov;
            positionDifferencePublisher->publish(ps);
        }
        if(posCorrectionPublisher->get_subscription_count() > 0){
            nav_msgs::msg::Odometry corrected_odom;
            corrected_odom.pose.pose.position.x = offset_x;
            corrected_odom.pose.pose.position.y = offset_y;
            tf2::Quaternion q_offset;
            q_offset.setRPY(0, 0, offset_yaw);
            corrected_odom.pose.pose.orientation = tf2::toMsg(q_offset);

            std::array<double, 36> cov;
            for(int i=0; i<36; ++i) cov[i] = 0.0;
            cov[0] = 100.0/(ind_count*ind_count) + .33;
            cov[7] = 100.0/(ind_count*ind_count) + .33;
            cov[35] = 5.0/(ind_count*ind_count) + 0.02;

            corrected_odom.pose.covariance = cov;
            corrected_odom.header.stamp = image.get()->header.stamp;
            corrected_odom.header.frame_id = "map";
            posCorrectionPublisher->publish(corrected_odom);
        }

        delete[] src_descs_arr;
        delete[] tgt_descs_arr; 
    }


    void QuatroNode::setParams(
            double noise_bound_of_each_measurement, double square_of_the_ratio_btw_noise_and_noise_bound,
            double estimating_scale, int num_max_iter, double control_parameter_for_gnc,
            double rot_cost_thr, const std::string& reg_type_name, Quatro<PointType, PointType>::Params &params) {

        params.noise_bound                   = noise_bound_of_each_measurement;
        params.cbar2                         = square_of_the_ratio_btw_noise_and_noise_bound;
        params.estimate_scaling              = estimating_scale;
        params.rotation_max_iterations       = num_max_iter;
        params.rotation_gnc_factor           = control_parameter_for_gnc;
        params.rotation_estimation_algorithm = Quatro<PointType, PointType>::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
        params.rotation_cost_threshold       = rot_cost_thr;

        params.reg_name                  = reg_type_name;
        if (reg_type_name == "Quatro") {
            params.inlier_selection_mode = Quatro<PointType, PointType>::INLIER_SELECTION_MODE::PMC_HEU;
        } else { params.inlier_selection_mode = Quatro<PointType, PointType>::INLIER_SELECTION_MODE::PMC_EXACT; }
    }

    void QuatroNode::publish_debug_descriptor_image(const std::vector<float>& data, const std::string& path) const{       
        cv::Mat img = cv::Mat(size, size, CV_32FC3, 0.0f);
        const int size_sq = size*size;
        for(int k=0; k<128; ++k){
            for(int i=0; i<size; ++i)
                for(int j=0; j<size; ++j){
                      if(k < 43)
                         img.at<cv::Vec<float, 3> >(i,j)[0] += data[k*size_sq + i * size + j]; 
                      else if(k < 86)
                         img.at<cv::Vec<float, 3> >(i,j)[1] += data[k*size_sq + i * size + j]; 
                      else if(k < 128)
                         img.at<cv::Vec<float, 3> >(i,j)[2] += data[k*size_sq + i * size + j]; 
                }
        }
        cv::normalize(img, img, 0.0f, 1.0f, cv::NORM_MINMAX, CV_32FC3);
        if(writeDebugImgs)
            cv::imwrite(path.c_str(), img);
    }

    at::Tensor QuatroNode::infer_descriptors(const cv::Mat& img){
        cv::Mat img_cpy = img.clone();
        cv::dnn::blobFromImage(img_cpy, img_cpy);
        std::vector<float> image_float32_vec;
        image_float32_vec.assign(img_cpy.begin<float>(), img_cpy.end<float>());
        long size_long = size;
        torch::Tensor input1 = torch::from_blob(image_float32_vec.data(), {3,size_long,size_long});
       	torch::List<torch::Tensor> list;
       	list.push_back(torch::unsqueeze(input1,0).to(use_cuda ? at::kCUDA : at::kCPU));
   	    std::vector<torch::jit::IValue> inputs(0);
   	    inputs.emplace_back(list);
        return module.forward(inputs).toTensor();
    }

    std::vector<std::tuple<int, int> > QuatroNode::get_keypoints(const at::Tensor& result, const cv::Mat& mask, const std::string& debug_image_postfix, const int border, const int add_keypoints_area){
        at::Tensor repe = torch::unsqueeze(result.index({0, 128}), 0);
        at::Tensor keypoints = repe;
        const at::Tensor& rep_idx = at::ge(result.index({0, 128}), repeatability_thresh);
        const at::Tensor& rel_idx = at::ge(result.index({0, 129}), reliability_thresh);
        keypoints *= rep_idx;
        keypoints *= rel_idx;
        keypoints = at::nonzero(keypoints);
       
        std::vector<std::tuple<int, int> > keypoints1;
        keypoints = keypoints.to(at::kCPU).contiguous();
        size_t num_keypoints = keypoints.size(0);
        keypoints1.reserve(num_keypoints);
        cv::Mat keypoints_debug_img = cv::Mat(size, size, CV_32FC1, 0.0f);
        std::vector<std::int64_t> points_vec(keypoints.data_ptr<std::int64_t>(), keypoints.data_ptr<std::int64_t>() + keypoints.numel());
        for(int i=1; i<points_vec.size(); i+=3){
            const long& x = points_vec[i];
            const long& y = points_vec[i + 1];
            double mask_sum = 0.0;
            if(x >= border && x <= size-border && y >= border && y <= size-border){
                mask_sum = mask.at<cv::Vec<float, 3> >(x, y)[0];
                    keypoints1.push_back(std::make_pair(y,x));
                    if(writeDebugImgs)
                        keypoints_debug_img.at<float>(x,y) = 1.0;
            }
        }
        if(writeDebugImgs)
            cv::imwrite("/tmp/test_keypoints_" + debug_image_postfix + ".exr", keypoints_debug_img);


        return keypoints1;
    }
}


int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto executor = std::make_shared<rclcpp::experimental::executors::EventsExecutor>();
  auto node = std::make_shared<quatro_node::QuatroNode>();
  image_transport::ImageTransport it(node);
  // the 3-channel BEV images for localization are published under the grid_map_cv_normals_z topic
  image_transport::Subscriber imageSubscriber = it.subscribe("/groundloc/groundgrid/grid_map_cv_normals_z", 10, std::bind(&quatro_node::QuatroNode::callbackBEV, node, std::placeholders::_1));
  executor->add_node(node);
  executor->spin();
  rclcpp::shutdown();
  return 0;
}
