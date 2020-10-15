#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "larvio_feature_tracker.h"
/*
vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img,pub_match;
ros::Publisher pub_restart;

LarvioFeatureTracker trackerData;
int pub_count = 1;
bool init_pub = 0;

void imucallback(const sensor_msgs::Imu::ConstPtr& msg)
{    
    ROS_WARN("2222");
    //imu callback
    double imutime = msg->header.stamp.toSec();
    Eigen::Vector3d wm, am;
    wm << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
    am << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
    trackerData.feedimu(imutime, wm, am);
}

void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    ROS_WARN("3333");
    //initial
    if(trackerData.bFirstImg)
    {
        trackerData.bFirstImg = false;
        trackerData.first_image_time = img_msg->header.stamp.toSec();
        trackerData.last_image_time = img_msg->header.stamp.toSec();
        return;
    }
    if (img_msg->header.stamp.toSec() - trackerData.last_image_time > 1.0 || img_msg->header.stamp.toSec() < trackerData.last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        trackerData.bFirstImg = true; 
        trackerData.last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        //重启publish
        pub_restart.publish(restart_flag);
        return;
    }
    trackerData.last_image_time=img_msg->header.stamp.toSec();

    //freq control
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - trackerData.first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - trackerData.first_image_time) - FREQ) < 0.01 * FREQ)
        {
            trackerData.first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    //read image
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat show_img = ptr->image;

    TicToc t_r;
    if(EQUALIZE)
        cv::equalizeHist(ptr->image, ptr->image);

    //image process
    ImageDataPtr msgPtr(new ImgData);
    msgPtr->timeStampToSec = img_msg->header.stamp.toSec();
    msgPtr->image = ptr->image.clone();
    trackerData.curr_img_ptr = msgPtr;
    trackerData.curr_img_time = trackerData.curr_img_ptr->timeStampToSec;
    std::vector<cv::Point2f> good_left;
    std::vector<unsigned long int> good_ids_left;

    trackerData.createImagePyramids();
    trackerData.currORBDescriptor_ptr.reset(new ORBdescriptor(trackerData.curr_pyramid_[0], 2, pyr_levels));
    std::vector<cv::Mat> imgpyr(trackerData.curr_pyramid_);
    bool haveFeatures = false;

    if ( FIRST_IMAGE==trackerData.image_state ) {
        if (trackerData.initializeFirstFrame())
            trackerData.image_state = SECOND_IMAGE;
    } else if ( SECOND_IMAGE==trackerData.image_state ) {
        if ( !trackerData.initializeFirstFeatures(trackerData.imu_msg_buffer) ) {
            trackerData.image_state = FIRST_IMAGE;
        } else 
        {
            if ( PUB_THIS_FRAME ) 
            {
                pub_count++;
                trackerData.findNewFeaturesToBeTracked();
                /*
                std::ofstream outfile(("/home/zty/myGit/open_vins/src/open_vins/featuretrack.txt"),std::ios::app);
                outfile <<std::fixed<< std::setprecision(13) << timestamp <<" "<<prev_pts_.size()<<endl;
                outfile.close();
                */
               /*
                for(size_t i=0; i<trackerData.prev_pts_.size(); i++) {
                good_left.push_back(trackerData.prev_pts_[i]);
                good_ids_left.push_back(trackerData.pts_ids_[i]);
                }
                sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
                trackerData.publish( img_msg, good_ids_left, good_left ,feature_points );
            if (!init_pub)
            {
                init_pub = 1;
            }
            else
            {
                //ros publisher
                pub_img.publish(feature_points);
            }
                haveFeatures = true;
            }

            trackerData.image_state = OTHER_IMAGES;
        }
    } else if ( OTHER_IMAGES==trackerData.image_state ) {

        trackerData.integrateImuData(trackerData.R_Prev2Curr, trackerData.imu_msg_buffer);
        trackerData.trackFeatures();
        trackerData.trackNewFeatures();
        /*
        std::ofstream outfile(("/home/zty/myGit/open_vins/src/open_vins/featuretrack.txt"),std::ios::app);
        outfile <<std::fixed<< std::setprecision(13) << timestamp <<" "<<prev_pts_.size()<<endl;
        outfile.close();
        */
       /*
        if ( PUB_THIS_FRAME ) 
        {
            pub_count++;
            trackerData.findNewFeaturesToBeTracked();
            for(size_t i=0; i<trackerData.prev_pts_.size(); i++) {
                good_left.push_back(trackerData.prev_pts_[i]);
                good_ids_left.push_back(trackerData.pts_ids_[i]);
            }
            sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
            trackerData.publish(img_msg, good_ids_left, good_left , feature_points);
            if (!init_pub)
            {
                init_pub = 1;
            }
            else
            {
                //ros publisher
                pub_img.publish(feature_points);
            }
            haveFeatures = true;
        }
    }
    if(SHOW_TRACK)
    {
        ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
        cv::Mat stereo_img = ptr->image;
        cv::Mat tmp_img = stereo_img;
        cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);
        for (unsigned int j = 0; j < trackerData.curr_pts_.size(); j++)
        {
            double len = std::min(1.0, 1.0 * trackerData.pts_lifetime_[j] / WINDOW_SIZE);
            cv::circle(tmp_img, trackerData.curr_pts_[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
        }
        for (unsigned int j = 0; j < trackerData.new_pts_.size(); j++)
        {
            double len = std::min(1.0, 1.0 / WINDOW_SIZE);
            cv::circle(tmp_img, trackerData.new_pts_[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
        }
        pub_match.publish(ptr->toImageMsg());
    }

    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());

    //更新
    trackerData.prev_img_ptr = trackerData.curr_img_ptr;
    trackerData.prev_img_time = trackerData.curr_img_time;
    std::swap(trackerData.prev_pyramid_,trackerData.curr_pyramid_);
    trackerData.prevORBDescriptor_ptr = trackerData.currORBDescriptor_ptr;
    std::swap(trackerData.prev_pts_,trackerData.curr_pts_);
    std::vector<cv::Point2f>().swap(trackerData.curr_pts_);
    //feature vec清空
    std::vector<cv::Point2f>().swap(good_left);
    std::vector<unsigned long int>().swap(good_ids_left);


}
  */  
int main(int argc, char **argv)
{
    ros::init(argc, argv, "larvio_feature_tracker");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    ROS_WARN("1111");
    /*
    readParameters(n);
    
    if(FISHEYE)
    {
        trackerData.fisheye_mask = cv::imread(FISHEYE_MASK, 0);
        if(!trackerData.fisheye_mask.data)
        {
            ROS_INFO("load mask fail");
            ROS_BREAK();
        }
        else
            ROS_INFO("load mask success");
    }
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 5000, imucallback);
    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);

    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    ros::spin();


    return 0;

}  





