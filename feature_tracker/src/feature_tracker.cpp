#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE &&
           BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<cv::Mat> &v, vector<uchar> status)
{
    vector<cv::Mat> temp;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            temp.push_back(v[i]);
    swap(v, temp);
}

void reduceVector(cv::Mat &v, vector<uchar> status)
{
    cv::Mat temp;
    for (int i = 0; i < int(v.rows); i++)
        if (status[i])
            temp.push_back(v.row(i));
    v = temp.clone();
}

FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::setMask()
{
    if (FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    if (USE_LARVIO)
    {
        vector<pair<int, pair<cv::Point2f, pair<int, cv::Mat>>>> cnt_pts_id;

        for (unsigned int i = 0; i < forw_pts.size(); i++)
            cnt_pts_id.push_back(make_pair(
                track_cnt[i],
                make_pair(forw_pts[i], make_pair(ids[i], vOrbDescriptors[i]))));

        sort(cnt_pts_id.begin(), cnt_pts_id.end(),
             [](const pair<int, pair<cv::Point2f, pair<int, cv::Mat>>> &a,
                const pair<int, pair<cv::Point2f, pair<int, cv::Mat>>> &b) {
                 return a.first > b.first;
             });

        forw_pts.clear();
        ids.clear();
        track_cnt.clear();
        vOrbDescriptors.clear();

        for (auto &it : cnt_pts_id)
        {
            if (mask.at<uchar>(it.second.first) == 255)
            {
                forw_pts.push_back(it.second.first);
                ids.push_back(it.second.second.first);
                track_cnt.push_back(it.first);
                vOrbDescriptors.push_back(it.second.second.second);
                cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
            }
        }
    }
    else
    {
        vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

        for (unsigned int i = 0; i < forw_pts.size(); i++)
            cnt_pts_id.push_back(
                make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

        sort(cnt_pts_id.begin(), cnt_pts_id.end(),
             [](const pair<int, pair<cv::Point2f, int>> &a,
                const pair<int, pair<cv::Point2f, int>> &b) {
                 return a.first > b.first;
             });

        forw_pts.clear();
        ids.clear();
        track_cnt.clear();

        for (auto &it : cnt_pts_id)
        {
            if (mask.at<uchar>(it.second.first) == 255)
            {
                forw_pts.push_back(it.second.first);
                ids.push_back(it.second.second);
                track_cnt.push_back(it.first);
                cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
            }
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
    if (USE_LARVIO)
    {
        vector<int> levels(n_pts.size(), 0);
        forwORBDescriptor_ptr->computeDescriptors(n_pts, levels,
                                                  currDescriptors);
        for (int i = 0; i < n_pts.size(); i++)
        {
            vOrbDescriptors.push_back(currDescriptors.row(i));
        }
    }
    // std::cerr <<"vOrbDescriptors size:"<< vOrbDescriptors.size() <<
    // std::endl;
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;
    // std::cerr << "Read new image" << std::endl;
    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    if (forw_img.empty())
    {

        //将prev_img、cur_img、forw_img都赋值为img
        //相当于在第一次送image初始化时将三个image都赋值了
        prev_img = img;
        cur_img = img;
        forw_img = img;
        if (USE_LARVIO)
        {
            buildOpticalFlowPyramid(forw_img, forw_pyramid_,
                                    cv::Size(PATCH_SIZE, PATCH_SIZE),
                                    PYR_LEVELS, true, cv::BORDER_REFLECT_101,
                                    cv::BORDER_CONSTANT, false);
            buildOpticalFlowPyramid(cur_img, curr_pyramid_,
                                    cv::Size(PATCH_SIZE, PATCH_SIZE),
                                    PYR_LEVELS, true, cv::BORDER_REFLECT_101,
                                    cv::BORDER_CONSTANT, false);
            buildOpticalFlowPyramid(prev_img, prev_pyramid_,
                                    cv::Size(PATCH_SIZE, PATCH_SIZE),
                                    PYR_LEVELS, true, cv::BORDER_REFLECT_101,
                                    cv::BORDER_CONSTANT, false);
            prevORBDescriptor_ptr.reset(
                new ORBdescriptor(forw_pyramid_[0], 2, PYR_LEVELS));
            currORBDescriptor_ptr.reset(
                new ORBdescriptor(forw_pyramid_[0], 2, PYR_LEVELS));
            forwORBDescriptor_ptr.reset(
                new ORBdescriptor(forw_pyramid_[0], 2, PYR_LEVELS));
        }
    }
    else
    {

        forw_img = img;
        if (USE_LARVIO)
        {
            buildOpticalFlowPyramid(forw_img, forw_pyramid_,
                                    cv::Size(PATCH_SIZE, PATCH_SIZE),
                                    PYR_LEVELS, true, cv::BORDER_REFLECT_101,
                                    cv::BORDER_CONSTANT, false);
            forwORBDescriptor_ptr.reset(
                new ORBdescriptor(forw_pyramid_[0], 2, PYR_LEVELS));
        }
    }
    //将forw_pts清空

    forw_pts.clear();

    if (cur_pts.size() > 0)
    {

        if (USE_LARVIO)
        {
            int before_tracking;
            int after_lktrack;
            int after_rlktrack;
            int after_destrack;
            int after_ransac;

            before_tracking = cur_pts.size();
            // std::cerr <<"before_tracking:"<< before_tracking <<
            // std::endl;
            //计算imu数据
            TicToc t_i;
            integrateImuData(R_Prev2Curr, imu_msg_buffer);
            predictFeatureTracking(cur_pts, R_Prev2Curr, cam_intrinsics,
                                   forw_pts);
            ROS_DEBUG("ImuData integrate and features predict costs: %fms",
                      t_i.toc());

            //光流跟踪
            TicToc t_o;
            vector<uchar> status1;
            vector<float> err1;
            cv::calcOpticalFlowPyrLK(
                curr_pyramid_, forw_pyramid_, cur_pts, forw_pts, status1, err1,
                cv::Size(PATCH_SIZE, PATCH_SIZE), PYR_LEVELS,
                cv::TermCriteria(cv::TermCriteria::COUNT +
                                     cv::TermCriteria::EPS,
                                 MAX_ITERATION, TRACK_PRECISION),
                cv::OPTFLOW_USE_INITIAL_FLOW);
            for (int i = 0; i < int(forw_pts.size()); i++)
                if (status1[i] && !inBorder(forw_pts[i]))
                    status1[i] = 0;
            reduceVector(prev_pts, status1);
            reduceVector(cur_pts, status1);
            reduceVector(forw_pts, status1);
            reduceVector(ids, status1);
            reduceVector(cur_un_pts, status1);
            reduceVector(track_cnt, status1);
            reduceVector(vOrbDescriptors, status1);
            after_lktrack = cur_pts.size();
            // std::cerr <<"after_lktrack:"<< after_lktrack <<
            // std::endl;
            ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());

            //反光流
            TicToc t_f;
            vector<uchar> status2;
            vector<float> err2;
            vector<cv::Point2f> cur_pts_copy(cur_pts);
            cv::calcOpticalFlowPyrLK(
                forw_pyramid_, curr_pyramid_, forw_pts, cur_pts_copy, status2,
                err2, cv::Size(PATCH_SIZE, PATCH_SIZE), PYR_LEVELS,
                cv::TermCriteria(cv::TermCriteria::COUNT +
                                     cv::TermCriteria::EPS,
                                 MAX_ITERATION, TRACK_PRECISION),
                cv::OPTFLOW_USE_INITIAL_FLOW);
            for (int i = 0; i < int(cur_pts_copy.size()); i++)
                if (status2[i] && !inBorder(cur_pts_copy[i]))
                    status2[i] = 0;
            reduceVector(prev_pts, status2);
            reduceVector(cur_pts, status2);
            reduceVector(forw_pts, status2);
            reduceVector(ids, status2);
            reduceVector(cur_un_pts, status2);
            reduceVector(track_cnt, status2);
            reduceVector(vOrbDescriptors, status2);
            after_rlktrack = cur_pts.size();
            // std::cerr <<"after_rlktrack:"<< after_rlktrack <<
            // std::endl;
            ROS_DEBUG("temporal reverse optical flow costs: %fms", t_f.toc());

            // ORB描述符
            TicToc t_orb;
            vector<int> levels(cur_pts.size(), 0);
            if (!forwORBDescriptor_ptr->computeDescriptors(forw_pts, levels,
                                                           forwDescriptors))
            {
                cerr << "error happen while compute descriptors" << endl;
                vector<cv::Point2f>().swap(cur_pts);
                vector<cv::Point2f>().swap(prev_pts);
                vector<cv::Point2f>().swap(forw_pts);
                vector<cv::Point2f>().swap(cur_un_pts);
                vector<int>().swap(ids);
                vector<int>().swap(track_cnt);
                vector<cv::Mat>().swap(vOrbDescriptors);
                return;
            }
            // forwDescriptors第j行对应第j个特征点，vOrbDescriptors的第j个向量（类型为mat）对应第j
            vector<int> vDis;
            for (int j = 0; j < forwDescriptors.rows; ++j)
            {
                int dis = ORBdescriptor::computeDescriptorDistance(
                    vOrbDescriptors.at(j), forwDescriptors.row(j));
                // std::cerr << dis <<" ";
                vDis.push_back(dis);
            }
            // std::cerr << std::endl;
            //通过描述符距离判断是否局内点
            vector<unsigned char> status3(vOrbDescriptors.size(), 0);
            for (int i = 0; i < vOrbDescriptors.size(); i++)
            {
                if (vDis[i] <= 58)
                    status3[i] = 1;
            }
            reduceVector(prev_pts, status3);
            reduceVector(cur_pts, status3);
            reduceVector(forw_pts, status3);
            reduceVector(ids, status3);
            reduceVector(cur_un_pts, status3);
            reduceVector(track_cnt, status3);
            reduceVector(vOrbDescriptors, status3);
            after_destrack = cur_pts.size();
            // std::cerr <<"after_destrack:"<< after_destrack <<
            // std::endl;
            ROS_DEBUG("ORB Descriptors costs: %fms", t_orb.toc());

            // RANSAC
            rejectWithF();
            after_ransac = cur_pts.size();
            // std::cerr <<"after_ransac:"<< after_ransac << std::endl;

            if (SHOW_FEATURE_TRACK)
            {
                std::ofstream outfile(("/home/zty/myGit/VINS-Mono/"
                                       "src/VINS-Mono/results/" +
                                       DATASET_NAME + ".txt"),
                                      std::ios::app);
                outfile << std::fixed << std::setprecision(13) << cur_time
                        << " " << before_tracking << " " << after_lktrack << " "
                        << after_rlktrack << " " << after_destrack << " "
                        << after_ransac << std::endl;
                outfile.close();
            }
        }
        else
        {
            TicToc t_o;
            vector<uchar> status;
            vector<float> err;
            cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts,
                                     status, err, cv::Size(21, 21), 3);

            for (int i = 0; i < int(forw_pts.size()); i++)
                if (status[i] && !inBorder(forw_pts[i]))
                    status[i] = 0;
            reduceVector(prev_pts, status);
            reduceVector(cur_pts, status);
            reduceVector(forw_pts, status);
            reduceVector(ids, status);
            reduceVector(cur_un_pts, status);
            reduceVector(track_cnt, status);
            ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
        }
    }

    for (auto &n : track_cnt)
        n++;

    if (PUB_THIS_FRAME)
    {
        if (!USE_LARVIO)
            rejectWithF();
        countdebug = 1;
        // std::cerr <<"PUB_THIS_FRAME"<<std::endl;
        // rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if (mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(),
                                    0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        // std::cerr <<"n_pts numbers:"<< n_pts.size() << std::endl;
        TicToc t_a;
        //将n_pts加入forw_pts中，同时计算描述符
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
    //描述符交换
    if (USE_LARVIO)
    {
        prevORBDescriptor_ptr = currORBDescriptor_ptr;
        currORBDescriptor_ptr = prevORBDescriptor_ptr;
        prev_pyramid_.swap(curr_pyramid_);
        curr_pyramid_.swap(forw_pyramid_);
    }
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()),
            un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(
                Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(
                Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC,
                               F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        reduceVector(vOrbDescriptors, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(),
                  1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
        {
            ids[i] = n_id++;
        }
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera =
        CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(
                Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            // printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(),
            // b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        // cout << trackerData[0].K << endl;
        // printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        // printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 &&
            pp.at<float>(1, 0) + 300 < ROW + 600 &&
            pp.at<float>(0, 0) + 300 >= 0 &&
            pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300,
                                     pp.at<float>(0, 0) + 300) =
                cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            // ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y,
            // distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    // cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(
            make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        // printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x,
        // cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}

void FeatureTracker::integrateImuData(
    cv::Matx33f &cam_R_p2c, const std::vector<ImuData> &imu_msg_buffer)
{
    // Find the start and the end limit within the imu msg buffer.
    auto begin_iter = imu_msg_buffer.begin();
    // begin在上一帧前0.0049s以内
    while (begin_iter != imu_msg_buffer.end())
    {
        if (begin_iter->timeStampToSec - prev_time < -0.0049)
            ++begin_iter;
        else
            break;
    }

    auto end_iter = begin_iter;
    // end在当前帧后0.0049s以内
    while (end_iter != imu_msg_buffer.end())
    {
        if (end_iter->timeStampToSec - cur_time < 0.0049)
            ++end_iter;
        else
            break;
    }
    //计算角速度均值
    // Compute the mean angular velocity in the IMU frame.
    cv::Vec3f mean_ang_vel(0.0, 0.0, 0.0);
    for (auto iter = begin_iter; iter < end_iter; ++iter)
        mean_ang_vel +=
            cv::Vec3f(iter->angular_velocity[0], iter->angular_velocity[1],
                      iter->angular_velocity[2]);

    if (end_iter - begin_iter > 0)
        mean_ang_vel *= 1.0f / (end_iter - begin_iter);

    // Transform the mean angular velocity from the IMU
    // frame to the cam0 and cam1 frames.
    // imu坐标系转换到相机坐标系
    cv::Vec3f cam_mean_ang_vel = R_cam_imu.t() * mean_ang_vel;

    // Compute the relative rotation.
    double dtime = cur_time - prev_time;
    Rodrigues(cam_mean_ang_vel * dtime, cam_R_p2c);
    cam_R_p2c = cam_R_p2c.t();
    // cam_R_p2c是输出的imu计算的相机旋转矩阵
    return;
}

void FeatureTracker::predictFeatureTracking(
    const vector<cv::Point2f> &input_pts, const cv::Matx33f &R_p_c,
    const cv::Vec4d &intrinsics, vector<cv::Point2f> &compensated_pts)
{
    // Return directly if there are no input features.
    if (input_pts.size() == 0)
    {
        compensated_pts.clear();
        return;
    }
    compensated_pts.resize(input_pts.size());
    Eigen::Matrix3d R_p_c_temp;
    R_p_c_temp << R_p_c(0, 0), R_p_c(0, 1), R_p_c(0, 2), R_p_c(1, 0),
        R_p_c(1, 1), R_p_c(1, 2), R_p_c(2, 0), R_p_c(2, 1), R_p_c(2, 2);

    for (unsigned int i = 0; i < input_pts.size(); i++)
    {
        Eigen::Vector2d a(input_pts[i].x, input_pts[i].y);
        Eigen::Vector3d b;
        Eigen::Vector2d c;
        m_camera->liftProjective(a, b);
        b = R_p_c_temp * b;
        m_camera->spaceToPlane(b, c);
        compensated_pts[i].x = c[0];
        compensated_pts[i].y = c[1];
    }

    return;
}