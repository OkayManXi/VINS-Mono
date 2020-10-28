#pragma once

#include <csignal>
#include <cstdio>
#include <execinfo.h>
#include <iostream>
#include <queue>

#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "ImuData.hpp"
#include "ORBDescriptor.h"
#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);
void reduceVector(cv::Mat &v, vector<uchar> status);
class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img, double _cur_time);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();

    void integrateImuData(cv::Matx33f &cam_R_p2c,
                          const std::vector<ImuData> &imu_msg_buffer);

    void predictFeatureTracking(const vector<cv::Point2f> &input_pts,
                                const cv::Matx33f &R_p_c,
                                const cv::Vec4d &intrinsics,
                                vector<cv::Point2f> &compensated_pts);

    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img, forw_img;
    vector<cv::Mat> prev_pyramid_;
    vector<cv::Mat> curr_pyramid_;
    vector<cv::Mat> forw_pyramid_;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts;
    vector<cv::Point2f> pts_velocity;
    vector<int> ids;
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;

    boost::shared_ptr<ORBdescriptor> prevORBDescriptor_ptr;
    boost::shared_ptr<ORBdescriptor> currORBDescriptor_ptr;
    boost::shared_ptr<ORBdescriptor> forwORBDescriptor_ptr;
    cv::Mat currDescriptors;
    cv::Mat forwDescriptors;
    std::vector<cv::Mat> vOrbDescriptors;
    cv::Matx33f R_Prev2Curr;
    std::vector<ImuData> imu_msg_buffer;

    int countdebug = 1;

    static int n_id;
};
