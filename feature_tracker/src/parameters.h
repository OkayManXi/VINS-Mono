#pragma once
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>

extern int ROW;
extern int COL;
extern int FOCAL_LENGTH;
const int NUM_OF_CAM = 1;


extern std::string IMAGE_TOPIC;
extern std::string IMU_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
extern int WINDOW_SIZE;
extern int FREQ;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int STEREO_TRACK;
extern int EQUALIZE;
extern int FISHEYE;
extern bool PUB_THIS_FRAME;

extern cv::Matx33d R_cam_imu;
extern cv::Vec4d cam_intrinsics;
extern cv::Vec4d cam_distortion;
extern int PYR_LEVELS;
extern int PATCH_SIZE;
extern int MAX_ITERATION;
extern int TRACK_PRECISION;
extern std::string DATASET_NAME;

extern int SHOW_FEATURE_TRACK;
extern int USE_LARVIO;

void readParameters(ros::NodeHandle &n);
