#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include <std_msgs/Float64MultiArray.h>

#include "visualization.h"

using namespace ros;
using namespace Eigen;
ros::Publisher pub_odometry, pub_latest_odometry;
ros::Publisher pub_path, pub_relo_path;
ros::Publisher pub_point_cloud, pub_margin_cloud;
ros::Publisher pub_key_poses;
ros::Publisher pub_relo_relative_pose;
ros::Publisher pub_camera_pose;
ros::Publisher pub_camera_pose_visual;
nav_msgs::Path path, relo_path;

ros::Publisher pub_keyframe_pose;
ros::Publisher pub_keyframe_point;
ros::Publisher pub_extrinsic;

// debug
ros::Publisher pub_scale;
ros::Publisher pub_bias;
ros::Publisher pub_imu_predict;
ros::Publisher pub_optimized;
ros::Publisher pub_wheel_predict;
ros::Publisher pub_wheel_imu_predict;
ros::Publisher pub_window_info;

CameraPoseVisualization cameraposevisual(0, 1, 0, 1);
CameraPoseVisualization keyframebasevisual(0.0, 0.0, 1.0, 1.0);
static double sum_of_path = 0;
static Vector3d last_path(0.0, 0.0, 0.0);

using namespace Eigen2ROS;


PathTFPublisher::PathTFPublisher(ros::NodeHandle &nh) {
    sub_pose_graph_path_ = nh.subscribe<nav_msgs::Path>("/pose_graph/pose_graph_path", 1000,
                                 std::bind(&PathTFPublisher::on_pose_graph_path, this, std::placeholders::_1));

    pub_path_origin_o_wheel_ = nh.advertise<nav_msgs::Path>("wheel_path", 1000);
    pub_path_origin_o_wheelimu_ = nh.advertise<nav_msgs::Path>("wheel_imu_path", 1000);
    pub_path_origin_o_wheelimu3D_ = nh.advertise<nav_msgs::Path>("wheel_imu_path3D", 1000);
    pub_path_origin_o_vio_ = nh.advertise<nav_msgs::Path>("vio_base_path", 1000);
    pub_path_origin_o_loop_ = nh.advertise<nav_msgs::Path>("base_loop_path", 1000);
}

void PathTFPublisher::reset() {
    Affine3d I = Translation3d(0, 0, 0) * Matrix3d::Identity();
    T_b_c = I;
    T_b_o = I;
    T_origin_worldbase = I;
    T_worldbase_w = I;
    T_w_b = I;
    T_origin_o_wheel = I;
    T_origin_o_wheelimu = I;
    T_origin_o_wheelimu3D = I;
    T_origin_o_vio = I;
    T_origin_o_loop = I;

    header_ = {};
    path_origin_o_wheel_ = {};
    path_origin_o_wheelimu_ = {};
    path_origin_o_wheelimu3D_ = {};
    path_origin_o_vio_ = {};
    path_origin_o_loop_ = {};
}

void PathTFPublisher::on_pose_graph_path(const nav_msgs::Path::ConstPtr &msg) {
    /* rebuild path_origin_o_loop_ */
    path_origin_o_loop_.poses.clear();
    // insert path point from wheel_odom before pose_graph_path
    for (geometry_msgs::PoseStamped pose_stamped : path_origin_o_vio_.poses) {
        if (pose_stamped.header.stamp >= msg->poses.begin()->header.stamp) {
            break;
        }
        T_origin_o_loop = Pose2Affine3d(pose_stamped.pose);
        pose_stamped.header.frame_id = "world_origin_loop";
        path_origin_o_loop_.poses.emplace_back(pose_stamped);
    }
    // insert pose_graph_path point
    for (geometry_msgs::PoseStamped pose_stamped : msg->poses) {
        Affine3d T_w_b_loop = Pose2Affine3d(pose_stamped.pose);
        T_origin_o_loop = T_origin_worldbase * T_worldbase_w * T_w_b_loop * T_b_o;
        pose_stamped.header.frame_id = "world_origin_loop";
        pose_stamped.pose = Affine3d2Pose(T_origin_o_loop);
        path_origin_o_loop_.poses.emplace_back(pose_stamped);
    }
    // insert path point from vio after pose_graph_path
    if (path_origin_o_loop_.poses.empty()) {
        for (geometry_msgs::PoseStamped pose_stamped : path_origin_o_vio_.poses) {
            pose_stamped.header.frame_id = "world_origin_loop";
            path_origin_o_loop_.header = pose_stamped.header;
            path_origin_o_loop_.header.frame_id = "world_origin_loop";
            path_origin_o_loop_.poses.emplace_back(pose_stamped);
        }
    } else {
        Affine3d T_origin_oi_loop = Pose2Affine3d(path_origin_o_loop_.poses.back().pose);
        bool found_T_origin_oi_vio = false;
        Affine3d T_origin_oi_vio;
        for (geometry_msgs::PoseStamped pose_stamped : path_origin_o_vio_.poses) {
            if (pose_stamped.header.stamp <= path_origin_o_loop_.poses.back().header.stamp) {
                found_T_origin_oi_vio = true;
                T_origin_oi_vio = Pose2Affine3d(pose_stamped.pose);
            } else {
                if (found_T_origin_oi_vio) {
                    Affine3d T_origin_oj_vio = Pose2Affine3d(pose_stamped.pose);
                    Affine3d T_origin_oj_loop = T_origin_oi_loop * T_origin_oi_vio.inverse() * T_origin_oj_vio;
                    T_origin_o_loop = T_origin_oj_loop;
                    pose_stamped.header.frame_id = "world_origin_loop";
                    pose_stamped.pose = Affine3d2Pose(T_origin_o_loop);
                    path_origin_o_loop_.poses.emplace_back(pose_stamped);
                } else {
                    break;
                }
            }
        }
    }

    br_.sendTransform(
            tf::StampedTransform(Affine3d2Transform(T_origin_o_loop.inverse()), header_.stamp, "base_footprint",
                                 "world_origin_loop"));
    pub_path_origin_o_loop_.publish(path_origin_o_loop_);
}

void PathTFPublisher::on_estimator_update(const Estimator &estimator) {
    /* Update status */
    header_ = estimator.Headers[estimator.frame_count];
    T_b_c = Translation3d(estimator.tic[0]) * estimator.ric[0];
    T_b_o = Translation3d(estimator.tio) * estimator.rio;
    Vector3d p_o_b = -estimator.rio.inverse() * estimator.tio;
    Vector3d ypr_o_b = Utility::R2ypr(estimator.rio.inverse());
    ypr_o_b[1] = 0;
    ypr_o_b[2] = 0;
    T_worldbase_w = Translation3d(p_o_b) * Utility::ypr2R(ypr_o_b);
    T_origin_worldbase = Translation3d(estimator.base_integration_before_init.delta_p)
                         * estimator.base_integration_before_init.delta_q;
    if (estimator.solver_flag == Estimator::SolverFlag::INITIAL) {
        Affine3d T_origin_o = Translation3d(estimator.wheel_imu_odom.delta_p)
                              * estimator.wheel_imu_odom.delta_q;
        Affine3d T_worldbase_o = T_origin_worldbase.inverse() * T_origin_o;
        T_w_b = T_worldbase_w.inverse() * T_worldbase_o * T_b_o.inverse();
    } else {
        T_w_b = Translation3d(estimator.Ps[WINDOW_SIZE])
                * estimator.Rs[WINDOW_SIZE];
    }
    T_origin_o_wheel = estimator.wheel_only_odom.transform();
    T_origin_o_wheelimu = estimator.wheel_imu_odom.transform();
    T_origin_o_wheelimu3D = estimator.wheel_imu_odom3D.transform();
    T_origin_o_vio = T_origin_worldbase * T_worldbase_w * T_w_b * T_b_o;

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = header_;
    // wheel only odometry path
    pose_stamped.header.frame_id = "wheel_odom";
    pose_stamped.pose = Affine3d2Pose(T_origin_o_wheel);
    path_origin_o_wheel_.header = header_;
    path_origin_o_wheel_.header.frame_id = "wheel_odom";
    path_origin_o_wheel_.poses.emplace_back(pose_stamped);
    // wheel-imu-fusion odometry path
    pose_stamped.header.frame_id = "wheel_imu_odom";
    pose_stamped.pose = Affine3d2Pose(T_origin_o_wheelimu);
    path_origin_o_wheelimu_.header = header_;
    path_origin_o_wheelimu_.header.frame_id = "wheel_imu_odom";
    path_origin_o_wheelimu_.poses.emplace_back(pose_stamped);
    // wheel-imu-fusion odometry 3D path
    pose_stamped.header.frame_id = "wheel_imu_odom3D";
    pose_stamped.pose = Affine3d2Pose(T_origin_o_wheelimu3D);
    path_origin_o_wheelimu3D_.header = header_;
    path_origin_o_wheelimu3D_.header.frame_id = "wheel_imu_odom3D";
    path_origin_o_wheelimu3D_.poses.emplace_back(pose_stamped);
    // vio path
    pose_stamped.header.frame_id = "world_origin";
    pose_stamped.pose = Affine3d2Pose(T_origin_o_vio);
    path_origin_o_vio_.header = header_;
    path_origin_o_vio_.header.frame_id = "world_origin";
    path_origin_o_vio_.poses.emplace_back(pose_stamped);
    // loop path
    if (path_origin_o_loop_.poses.empty()) {
        for (geometry_msgs::PoseStamped pose_stamped : path_origin_o_vio_.poses) {
            T_origin_o_loop = Pose2Affine3d(pose_stamped.pose);
            pose_stamped.header.frame_id = "world_origin_loop";
            path_origin_o_loop_.header = pose_stamped.header;
            path_origin_o_loop_.header.frame_id = "world_origin_loop";
            path_origin_o_loop_.poses.emplace_back(pose_stamped);
        }
    } else {
        Affine3d T_origin_oi_loop = Pose2Affine3d(path_origin_o_loop_.poses.back().pose);
        bool found_T_origin_oi_vio = false;
        Affine3d T_origin_oi_vio;
        for (geometry_msgs::PoseStamped pose_stamped : path_origin_o_vio_.poses) {
            if (pose_stamped.header.stamp <= path_origin_o_loop_.poses.back().header.stamp) {
                found_T_origin_oi_vio = true;
                T_origin_oi_vio = Pose2Affine3d(pose_stamped.pose);
            } else {
                if (found_T_origin_oi_vio) {
                    Affine3d T_origin_oj_vio = Pose2Affine3d(pose_stamped.pose);
                    Affine3d T_origin_oj_loop = T_origin_oi_loop * T_origin_oi_vio.inverse() * T_origin_oj_vio;
                    T_origin_o_loop = T_origin_oj_loop;
                    pose_stamped.header.frame_id = "world_origin_loop";
                    pose_stamped.pose = Affine3d2Pose(T_origin_o_loop);
                    path_origin_o_loop_.header = pose_stamped.header;
                    path_origin_o_loop_.header.frame_id = "world_origin_loop";
                    path_origin_o_loop_.poses.emplace_back(pose_stamped);
                } else {
                    break;
                }
            }
        }
    }

    publish();
}

void PathTFPublisher::publish() {
    /* Publish transform */
    // fixed
    br_.sendTransform(tf::StampedTransform(Affine3d2Transform(T_b_c), header_.stamp, "body", "camera"));
    br_.sendTransform(tf::StampedTransform(Affine3d2Transform(T_b_o.inverse()), header_.stamp, "base_footprint", "body"));
    br_.sendTransform(
            tf::StampedTransform(Affine3d2Transform(T_worldbase_w.inverse()), header_.stamp, "world", "world_base"));
    // dynamic
    br_.sendTransform(
            tf::StampedTransform(Affine3d2Transform(T_origin_worldbase.inverse()), header_.stamp, "world_base",
                                 "world_origin"));
    br_.sendTransform(tf::StampedTransform(Affine3d2Transform(T_w_b.inverse()), header_.stamp, "body", "world"));
    br_.sendTransform(
            tf::StampedTransform(Affine3d2Transform(T_origin_o_wheel.inverse()), header_.stamp, "base_footprint",
                                 "wheel_odom"));
    br_.sendTransform(
            tf::StampedTransform(Affine3d2Transform(T_origin_o_wheelimu.inverse()), header_.stamp, "base_footprint",
                                 "wheel_imu_odom"));
    br_.sendTransform(
            tf::StampedTransform(Affine3d2Transform(T_origin_o_wheelimu3D.inverse()), header_.stamp, "base_footprint",
                                 "wheel_imu_odom3D"));
    br_.sendTransform(
            tf::StampedTransform(Affine3d2Transform(T_origin_o_loop.inverse()), header_.stamp, "base_footprint",
                                 "world_origin_loop"));

    /* Publish path */
    pub_path_origin_o_wheel_.publish(path_origin_o_wheel_);
    pub_path_origin_o_wheelimu_.publish(path_origin_o_wheelimu_);
    pub_path_origin_o_wheelimu3D_.publish(path_origin_o_wheelimu3D_);
    pub_path_origin_o_vio_.publish(path_origin_o_vio_);
    pub_path_origin_o_loop_.publish(path_origin_o_loop_);
}

void registerPub(ros::NodeHandle &n)
{
    pub_latest_odometry = n.advertise<nav_msgs::Odometry>("imu_propagate", 1000);
    pub_path = n.advertise<nav_msgs::Path>("path", 1000);
    pub_relo_path = n.advertise<nav_msgs::Path>("relocalization_path", 1000);
    pub_odometry = n.advertise<nav_msgs::Odometry>("odometry", 1000);
    pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("point_cloud", 1000);
    pub_margin_cloud = n.advertise<sensor_msgs::PointCloud>("history_cloud", 1000);
    pub_key_poses = n.advertise<visualization_msgs::Marker>("key_poses", 1000);
    pub_camera_pose = n.advertise<nav_msgs::Odometry>("camera_pose", 1000);
    pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
    pub_keyframe_pose = n.advertise<nav_msgs::Odometry>("keyframe_pose", 1000);
    pub_keyframe_point = n.advertise<sensor_msgs::PointCloud>("keyframe_point", 1000);
    pub_extrinsic = n.advertise<nav_msgs::Odometry>("extrinsic", 1000);
    pub_relo_relative_pose = n.advertise<nav_msgs::Odometry>("relo_relative_pose", 1000);

    // debug
    pub_scale = n.advertise<std_msgs::Float64MultiArray>("scale", 1000);
    pub_bias = n.advertise<sensor_msgs::Imu>("bias", 1000);
    pub_imu_predict = n.advertise<nav_msgs::Odometry>("imu_predict", 1000);
    pub_optimized = n.advertise<nav_msgs::Odometry>("optimized", 1000);
    pub_wheel_predict = n.advertise<nav_msgs::Odometry>("wheel_predict", 1000);
    pub_wheel_imu_predict = n.advertise<nav_msgs::Odometry>("wheel_imu_predict", 1000);
    pub_window_info = n.advertise<std_msgs::Float64MultiArray>("window_info", 1000);

    cameraposevisual.setScale(1);
    cameraposevisual.setLineWidth(0.05);
    keyframebasevisual.setScale(0.1);
    keyframebasevisual.setLineWidth(0.01);
}

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, const std_msgs::Header &header)
{
    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position = Vector3d2Point(P);
    odometry.pose.pose.orientation = Q2wxyz(Q);
    odometry.twist.twist.linear = Vector3d2Vector3(V);
    pub_latest_odometry.publish(odometry);
}

void printStatistics(const Estimator &estimator, double t)
{
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    printf("p:%6.3f %6.3f %6.3f v:%6.3f %6.3f %6.3f Ba:%6.3f %6.3f %6.3f Bg:%6.3f %6.3f %6.3f\r",
           estimator.Ps[WINDOW_SIZE].x(), estimator.Ps[WINDOW_SIZE].y(), estimator.Ps[WINDOW_SIZE].z(),
           estimator.Vs[WINDOW_SIZE].x(), estimator.Vs[WINDOW_SIZE].y(), estimator.Vs[WINDOW_SIZE].z(),
           estimator.Bas[WINDOW_SIZE].x(), estimator.Bas[WINDOW_SIZE].y(), estimator.Bas[WINDOW_SIZE].z(),
           estimator.Bgs[WINDOW_SIZE].x() / M_PI * 180, estimator.Bgs[WINDOW_SIZE].y() / M_PI * 180, estimator.Bgs[WINDOW_SIZE].z() / M_PI * 180);
    std::cout << std::flush;
    ROS_DEBUG_STREAM("position: " << estimator.Ps[WINDOW_SIZE].transpose());
    ROS_DEBUG_STREAM("orientation: " << estimator.Vs[WINDOW_SIZE].transpose());
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        //ROS_DEBUG("calibration result for camera %d", i);
        ROS_DEBUG_STREAM("extirnsic tic: " << estimator.tic[i].transpose());
        ROS_DEBUG_STREAM("extrinsic ric: " << Utility::R2ypr(estimator.ric[i]).transpose());
        if (ESTIMATE_EXTRINSIC)
        {
            cv::FileStorage fs(EX_CALIB_RESULT_PATH, cv::FileStorage::WRITE);
            Eigen::Matrix3d eigen_R;
            Eigen::Vector3d eigen_T;
            eigen_R = estimator.ric[i];
            eigen_T = estimator.tic[i];
            cv::Mat cv_R, cv_T;
            cv::eigen2cv(eigen_R, cv_R);
            cv::eigen2cv(eigen_T, cv_T);
            fs << "extrinsicRotation" << cv_R << "extrinsicTranslation" << cv_T;
            fs.release();
        }
    }

    static double sum_of_time = 0;
    static int sum_of_calculation = 0;
    sum_of_time += t;
    sum_of_calculation++;
    ROS_DEBUG("vo solver costs: %f ms", t);
    ROS_DEBUG("average of time %f ms", sum_of_time / sum_of_calculation);

    sum_of_path += (estimator.Ps[WINDOW_SIZE] - last_path).norm();
    last_path = estimator.Ps[WINDOW_SIZE];
    ROS_DEBUG("sum of path %f", sum_of_path);
//    if (ESTIMATE_TD)
//        ROS_INFO("td %f", estimator.td);
}

void pubOdometry(const Estimator &estimator, const std_msgs::Header &header)
{

    geometry_msgs::PoseStamped pose_stamped;

    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        nav_msgs::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.child_frame_id = "body";
        odometry.pose.pose = Rp2Pose(estimator.Rs[WINDOW_SIZE], estimator.Ps[WINDOW_SIZE]);
        odometry.twist.twist.linear = Vector3d2Vector3(estimator.Rs[WINDOW_SIZE].inverse() * estimator.Vs[WINDOW_SIZE]);
        pub_odometry.publish(odometry);

        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose = odometry.pose.pose;
        path.header = header;
        path.header.frame_id = "world";
        path.poses.push_back(pose_stamped);
        pub_path.publish(path);

        Vector3d correct_t;
        Vector3d correct_v;
        Quaterniond correct_q;
        correct_t = estimator.drift_correct_r * estimator.Ps[WINDOW_SIZE] + estimator.drift_correct_t;
        correct_q = estimator.drift_correct_r * estimator.Rs[WINDOW_SIZE];
        odometry.pose.pose = Rp2Pose(correct_q, correct_t);

        pose_stamped.pose = odometry.pose.pose;
        relo_path.header = header;
        relo_path.header.frame_id = "world";
        relo_path.poses.push_back(pose_stamped);
        pub_relo_path.publish(relo_path);


        // debug pub bias imu
        sensor_msgs::Imu msg;
        msg.header = header;
        msg.angular_velocity = Vector3d2Vector3(estimator.Bgs[WINDOW_SIZE]);
        msg.linear_acceleration = Vector3d2Vector3(estimator.Bas[WINDOW_SIZE]);
        pub_bias.publish(msg);
    }
}

void pubKeyPoses(const Estimator &estimator, const std_msgs::Header &header) {
    if (estimator.key_poses.size() == 0)
        return;
    visualization_msgs::Marker key_poses;
    key_poses.header = header;
    key_poses.header.frame_id = "world";
    key_poses.ns = "key_poses";
    key_poses.type = visualization_msgs::Marker::SPHERE_LIST;
    key_poses.action = visualization_msgs::Marker::ADD;
    key_poses.pose.orientation.w = 1.0;
    key_poses.lifetime = ros::Duration();

    //static int key_poses_id = 0;
    key_poses.id = 0; //key_poses_id++;
    key_poses.scale.x = 0.05;
    key_poses.scale.y = 0.05;
    key_poses.scale.z = 0.05;
    key_poses.color.r = 1.0;
    key_poses.color.a = 1.0;

    for (int i = 0; i <= WINDOW_SIZE; i++) {
        Vector3d correct_pose;
        correct_pose = estimator.key_poses[i];
        geometry_msgs::Point pose_marker = Vector3d2Point(correct_pose);
        key_poses.points.push_back(pose_marker);
    }
    pub_key_poses.publish(key_poses);

    // debug: print scale_vio VS scale_wheel
    if (estimator.solver_flag == estimator.NON_LINEAR) {
        std::vector<double> scales;
        // calc scale_vio
        {
            Eigen::Affine3d pose_imu_0 = Eigen::Translation3d(estimator.Ps[0]) * estimator.Rs[0];
            Eigen::Affine3d pose_imu_1 = Eigen::Translation3d(estimator.Ps[WINDOW_SIZE - 1]) * estimator.Rs[WINDOW_SIZE - 1];
            Eigen::Affine3d T_imu_base = Eigen::Translation3d(estimator.tio) * estimator.rio;
            Eigen::Affine3d pose_vio_0 = pose_imu_0 * T_imu_base;
            Eigen::Affine3d pose_vio_1 = pose_imu_1 * T_imu_base;
            Eigen::Affine3d T_vio = pose_vio_0.inverse() * pose_vio_1;
            Eigen::Vector2d t_vio = T_vio.translation().head(2);
            scales.push_back(t_vio.norm());
        }
        // calc scale_wheel
        {
            Eigen::Affine3d T_wheel = Eigen::Translation3d(Eigen::Vector3d::Zero()) * Eigen::Matrix3d::Identity();
            for (int i = 1; i < WINDOW_SIZE; i++) {
                T_wheel = T_wheel * (Eigen::Translation3d(estimator.base_integrations[i]->delta_p) * estimator.base_integrations[i]->delta_q);
            }
            Eigen::Vector3d t_wheel = T_wheel.translation();
            scales.push_back(t_wheel.norm());
        }
        {
            // calc vel_vio (latest keyframe)
            Eigen::Affine3d pose_imu_0 = Eigen::Translation3d(estimator.Ps[WINDOW_SIZE - 2]) * estimator.Rs[WINDOW_SIZE - 2];
            Eigen::Affine3d pose_imu_1 = Eigen::Translation3d(estimator.Ps[WINDOW_SIZE - 1]) * estimator.Rs[WINDOW_SIZE - 1];
            Eigen::Affine3d T_imu_base = Eigen::Translation3d(estimator.tio) * estimator.rio;
            Eigen::Affine3d pose_vio_0 = pose_imu_0 * T_imu_base;
            Eigen::Affine3d pose_vio_1 = pose_imu_1 * T_imu_base;
            Eigen::Affine3d T_vio = pose_vio_0.inverse() * pose_vio_1;
            Eigen::Vector2d t_vio = T_vio.translation().head(2);
            scales.push_back(t_vio.norm() / estimator.pre_integrations[WINDOW_SIZE - 1]->sum_dt);
            // calc vel_wheel (latest keyframe)
            scales.push_back(estimator.base_integrations[WINDOW_SIZE - 1]->delta_p.norm() / estimator.base_integrations[WINDOW_SIZE - 1]->sum_dt);
            // latest keyframe translation err between vio and wheel-odom
            // calc x error
            scales.push_back(t_vio.x() - estimator.base_integrations[WINDOW_SIZE - 1]->delta_p.x());
            // calc y error
            scales.push_back(t_vio.y() - estimator.base_integrations[WINDOW_SIZE - 1]->delta_p.y());
        }

        std_msgs::Float64MultiArray msg;
        msg.data = scales;
        pub_scale.publish(msg);
    }
}

void pubCameraPose(const Estimator &estimator, const std_msgs::Header &header)
{
    int idx2 = WINDOW_SIZE - 1;

    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        int i = idx2;
        Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[0]);

        nav_msgs::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.pose.pose = Rp2Pose(R, P);
        pub_camera_pose.publish(odometry);

        cameraposevisual.reset();
        cameraposevisual.add_pose(P, R);
        cameraposevisual.publish_by(pub_camera_pose_visual, odometry.header);
    }
}


void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header)
{
    sensor_msgs::PointCloud point_cloud, loop_point_cloud;
    point_cloud.header = header;
    loop_point_cloud.header = header;


    for (auto &it_per_id : estimator.f_manager.feature)
    {
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1)
            continue;
        int imu_i = it_per_id.start_frame;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

        geometry_msgs::Point32 p;
        p.x = w_pts_i(0);
        p.y = w_pts_i(1);
        p.z = w_pts_i(2);
        point_cloud.points.push_back(p);
    }
    pub_point_cloud.publish(point_cloud);


    // pub margined potin
    sensor_msgs::PointCloud margin_cloud;
    margin_cloud.header = header;

    for (auto &it_per_id : estimator.f_manager.feature)
    { 
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        //if (it_per_id->start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id->solve_flag != 1)
        //        continue;

        if (it_per_id.start_frame == 0 && it_per_id.feature_per_frame.size() <= 2 
            && it_per_id.solve_flag == 1 )
        {
            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
            Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

            geometry_msgs::Point32 p;
            p.x = w_pts_i(0);
            p.y = w_pts_i(1);
            p.z = w_pts_i(2);
            margin_cloud.points.push_back(p);
        }
    }
    pub_margin_cloud.publish(margin_cloud);
}

void pubTF(const Estimator &estimator, const std_msgs::Header &header)
{

    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose = Rp2Pose(estimator.ric[0], estimator.tic[0]);
    pub_extrinsic.publish(odometry);

}

void pubKeyframe(const Estimator &estimator)
{
    // pub camera pose, 2D-3D points of keyframe
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR && estimator.marginalization_flag == 0)
    {
        int i = WINDOW_SIZE - 2;
        //Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];

        nav_msgs::Odometry odometry;
        odometry.header = estimator.Headers[WINDOW_SIZE - 2];
        odometry.header.frame_id = "world";
        odometry.pose.pose = Rp2Pose(estimator.Rs[i], estimator.Ps[i]);

        //printf("time: %f t: %f %f %f r: %f %f %f %f\n", odometry.header.stamp.toSec(), P.x(), P.y(), P.z(), R.w(), R.x(), R.y(), R.z());

        pub_keyframe_pose.publish(odometry);


        sensor_msgs::PointCloud point_cloud;
        point_cloud.header = estimator.Headers[WINDOW_SIZE - 2];
        for (auto &it_per_id : estimator.f_manager.feature)
        {
            int frame_size = it_per_id.feature_per_frame.size();
            if(it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - 2 && it_per_id.solve_flag == 1)
            {

                int imu_i = it_per_id.start_frame;
                Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0])
                                      + estimator.Ps[imu_i];
                geometry_msgs::Point32 p;
                p.x = w_pts_i(0);
                p.y = w_pts_i(1);
                p.z = w_pts_i(2);
                point_cloud.points.push_back(p);

                int imu_j = WINDOW_SIZE - 2 - it_per_id.start_frame;
                sensor_msgs::ChannelFloat32 p_2d;
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.y());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.y());
                p_2d.values.push_back(it_per_id.feature_id);
                point_cloud.channels.push_back(p_2d);
            }

        }
        pub_keyframe_point.publish(point_cloud);
    }
}

void pubRelocalization(const Estimator &estimator)
{
    nav_msgs::Odometry odometry;
    odometry.header.stamp = ros::Time(estimator.relo_frame_stamp);
    odometry.header.frame_id = "world";
    odometry.pose.pose = Rp2Pose(estimator.relo_relative_q, estimator.relo_relative_t);
    odometry.twist.twist.linear.x = estimator.relo_relative_yaw;
    odometry.twist.twist.linear.y = estimator.relo_frame_index;

    pub_relo_relative_pose.publish(odometry);
}