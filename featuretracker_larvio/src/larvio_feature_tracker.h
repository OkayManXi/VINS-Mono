
#include "ORBDescriptor.h"
#include "ImuData.hpp"
#include "ImageData.hpp"
#include <fstream>
#include "parameters.h"
#include "tic_toc.h"

#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <Eigen/StdVector>

#include <boost/thread.hpp>
#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>

    
    class LarvioFeatureTracker {

    public:
        LarvioFeatureTracker();

        //void feed_monocular(double timestamp, cv::Mat &img) ;

        void feedimu(double timestamp, Eigen::Vector3d wm, Eigen::Vector3d am) ;

        template <typename T>
        void removeUnmarkedElements(
            const std::vector<T>& raw_vec,
            const std::vector<unsigned char>& markers,
            std::vector<T>& refined_vec) {
            if (raw_vec.size() != markers.size()) {
                for (int i = 0; i < raw_vec.size(); ++i)
                refined_vec.push_back(raw_vec[i]);
            return;
            }
            for (int i = 0; i < markers.size(); ++i) {
                if (markers[i] == 0) continue;
                refined_vec.push_back(raw_vec[i]);
            }
            return;
        }

        void undistortPoints(const std::vector<cv::Point2f>& pts_in,
        const cv::Vec4d& intrinsics, const cv::Vec4d& distortion_coeffs,
        std::vector<cv::Point2f>& pts_out, const cv::Matx33d &rectification_matrix = cv::Matx33d::eye(),
        const cv::Vec4d &new_intrinsics = cv::Vec4d(1,1,0,0));

        void integrateImuData(cv::Matx33f& cam_R_p2c,  std::vector<ImuData>& imu_msg_buffer);

        void predictFeatureTracking(const std::vector<cv::Point2f>& input_pts, 
        const cv::Matx33f& R_p_c, 
        const cv::Vec4d& intrinsics, 
        std::vector<cv::Point2f>& compenstated_pts);

        bool initializeFirstFrame();

        bool initializeFirstFeatures( std::vector<ImuData>& imu_msg_buffer);

        void createImagePyramids();

        void trackFeatures();

        void trackNewFeatures();

        void findNewFeaturesToBeTracked();

        typedef unsigned long int FeatureIDType;

        void publish( const sensor_msgs::ImageConstPtr &img_msg, 
        const std::vector<FeatureIDType> &curr_ids, 
        const std::vector<cv::Point2f> &curr_pt, 
        sensor_msgs::PointCloudPtr feature_points);

        const std::vector<cv::Point2f> undistort_point_brown(const std::vector<cv::Point2f> pts_in, 
        cv::Vec4d &camK, 
        cv::Vec4d &camD) 
        {
            cv::Matx33d ck;
            ck=cv::Matx33d::zeros();
            ck(0, 0) = camK[0];
            ck(0, 2) = camK[1];
            ck(1, 1) = camK[2];
            ck(1, 2) = camK[3];
            ck(2, 2) = 1;
            std::vector<cv::Point2f> pts_out;

            for(auto &pt_in : pts_in)
            {
            // Convert to opencv format
            cv::Mat mat(1, 2, CV_32F);
            mat.at<float>(0, 0) = pt_in.x;
            mat.at<float>(0, 1) = pt_in.y;
            mat = mat.reshape(2); // Nx1, 2-channel
            // Undistort it!
            cv::undistortPoints(mat, mat, ck, camD);
            // Construct our return vector
            cv::Point2f temp;
            mat = mat.reshape(1); // Nx2, 1-channel
            temp.x = mat.at<float>(0, 0);
            temp.y = mat.at<float>(0, 1);
            pts_out.push_back(temp);
            }

            return pts_out;
        }

        const std::vector<cv::Point2f> undistort_point_fisheye(const std::vector<cv::Point2f> pts_in, 
        cv::Vec4d &camK, 
        cv::Vec4d &camD) 
        {
            cv::Matx33d ck;
            ck=cv::Matx33d::zeros();
            ck(0, 0) = camK[0];
            ck(0, 2) = camK[1];
            ck(1, 1) = camK[2];
            ck(1, 2) = camK[3];
            ck(2, 2) = 1;
            std::vector<cv::Point2f> pts_out;

            for(auto &pt_in : pts_in)
            {
            // Convert to opencv format
            cv::Mat mat(1, 2, CV_32F);
            mat.at<float>(0, 0) = pt_in.x;
            mat.at<float>(0, 1) = pt_in.y;
            mat = mat.reshape(2); // Nx1, 2-channel
            // Undistort it!
            cv::fisheye::undistortPoints(mat, mat, ck, camD);
            // Construct our return vector
            cv::Point2f temp;
            mat = mat.reshape(1); // Nx2, 1-channel
            temp.x = mat.at<float>(0, 0);
            temp.y = mat.at<float>(0, 1);
            pts_out.push_back(temp);
            }

            return pts_out;
        }
        

        bool bFirstImg=true;

        double last_pub_time;
        double curr_img_time;
        double prev_img_time;
        double first_image_time;
        double last_image_time = 0;

        ImageDataPtr prev_img_ptr;
        ImageDataPtr curr_img_ptr;

        cv::Size win_size = cv::Size(15, 15);

        std::vector<ImuData> imu_msg_buffer;

        cv::Matx33f R_Prev2Curr;  
        cv::Matx33d R_cam_imu;

        std::vector<cv::Mat> prev_pyramid_;
        std::vector<cv::Mat> curr_pyramid_;

        FeatureIDType next_feature_id=0;

        int before_tracking;
        int after_tracking;
        int after_ransac;

        cv::Mat fisheye_mask;

        // Points for tracking, added by QXC
        std::vector<cv::Point2f> new_pts_;
        std::vector<cv::Point2f> prev_pts_;
        std::vector<cv::Point2f> curr_pts_;
        std::vector<FeatureIDType> pts_ids_;
        std::vector<int> pts_lifetime_;
        std::vector<cv::Point2f> init_pts_;

        eImageState image_state=FIRST_IMAGE;

        boost::shared_ptr<ORBdescriptor> prevORBDescriptor_ptr;
        boost::shared_ptr<ORBdescriptor> currORBDescriptor_ptr;
        std::vector<cv::Mat> vOrbDescriptors;


    };
