#include "parameters.h"

std::string IMAGE_TOPIC;
std::string IMU_TOPIC;
std::vector<std::string> CAM_NAMES;
std::string FISHEYE_MASK;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
int EQUALIZE;
int ROW;
int COL;
int FOCAL_LENGTH;
int FISHEYE;
bool PUB_THIS_FRAME;

cv::Matx33d R_cam_imu;
cv::Vec4d cam_intrinsics;
cv::Vec4d cam_distortion;
int pyr_levels;

//读launch文件
template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{   
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}
//读config,yaml文件
void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file");
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    std::string VINS_FOLDER_PATH = readParam<std::string>(n, "vins_folder");

    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["imu_topic"] >> IMU_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    FREQ = fsSettings["freq"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    EQUALIZE = fsSettings["equalize"];
    FISHEYE = fsSettings["fisheye"];
    if (FISHEYE == 1)
        FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";
    CAM_NAMES.push_back(config_file);

    WINDOW_SIZE = 20;
    STEREO_TRACK = false;
    FOCAL_LENGTH = 460;
    PUB_THIS_FRAME = false;

    cv::FileNode x = fsSettings["projection_parameters"];
    cam_intrinsics(0) = static_cast<double>(x["fx"]);
    cam_intrinsics(1) = static_cast<double>(x["fy"]);
    cam_intrinsics(2) = static_cast<double>(x["cx"]);
    cam_intrinsics(3) = static_cast<double>(x["cy"]);

    x = fsSettings["distortion_parameters"];
    cam_distortion(0) = static_cast<double>(x["k1"]);
    cam_distortion(1) = static_cast<double>(x["k2"]);
    cam_distortion(2) = static_cast<double>(x["p1"]);
    cam_distortion(3) = static_cast<double>(x["p1"]);

    x = fsSettings["extrinsicRotation"];
    R_cam_imu = static_cast<cv::Matx33d>(x["data"]);

    pyr_levels = 3;

    
    if (FREQ == 0)
        FREQ = 100;

    fsSettings.release();
}
