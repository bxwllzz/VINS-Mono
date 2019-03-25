#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include <std_msgs/Float64MultiArray.h>

#include "visualization.h"

using namespace ros;
using namespace Eigen;
ros::Publisher pub_odometry, pub_latest_odometry;
ros::Publisher pub_path, pub_relo_path;
ros::Publisher pub_vio_base_path, pub_wheel_path, pub_wheel_imu_path, pub_wheel_imu_path3D;
ros::Publisher pub_point_cloud, pub_margin_cloud;
ros::Publisher pub_key_poses;
ros::Publisher pub_relo_relative_pose;
ros::Publisher pub_camera_pose;
ros::Publisher pub_camera_pose_visual;
nav_msgs::Path path, relo_path;
nav_msgs::Path vio_base_path, wheel_path, wheel_imu_path, wheel_imu_path3D;

ros::Publisher pub_keyframe_pose;
ros::Publisher pub_keyframe_point;
ros::Publisher pub_extrinsic;

// debug
ros::Publisher pub_velocity_yaw;
ros::Publisher pub_orig_odom;
ros::Publisher pub_filted_odom;
ros::Publisher pub_orig_imu;
ros::Publisher pub_filted_imu;
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

void registerPub(ros::NodeHandle &n)
{
    pub_latest_odometry = n.advertise<nav_msgs::Odometry>("imu_propagate", 1000);
    pub_path = n.advertise<nav_msgs::Path>("path", 1000);
    pub_relo_path = n.advertise<nav_msgs::Path>("relocalization_path", 1000);
    pub_vio_base_path = n.advertise<nav_msgs::Path>("vio_base_path", 1000);
    pub_wheel_path = n.advertise<nav_msgs::Path>("wheel_path", 1000);
    pub_wheel_imu_path = n.advertise<nav_msgs::Path>("wheel_imu_path", 1000);
    pub_wheel_imu_path3D = n.advertise<nav_msgs::Path>("wheel_imu_path3D", 1000);
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
    pub_velocity_yaw = n.advertise<nav_msgs::Odometry>("velocity_yaw", 1000);
    pub_orig_odom = n.advertise<nav_msgs::Odometry>("orig_odom", 1000);
    pub_filted_odom = n.advertise<nav_msgs::Odometry>("filted_odom", 1000);
    pub_orig_imu = n.advertise<sensor_msgs::Imu>("orig_imu", 1000);
    pub_filted_imu = n.advertise<sensor_msgs::Imu>("filted_imu", 1000);
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

void pubVelocityYaw(const Estimator &estimator, const std_msgs::Header& header) {

    geometry_msgs::PoseStamped pose_stamped;

    // publish wheel only odometry path
    pose_stamped.header = header;
    pose_stamped.header.frame_id = "wheel_odom";
    pose_stamped.pose.position.x = estimator.wheel_only_odom.delta_p.x();
    pose_stamped.pose.position.y = estimator.wheel_only_odom.delta_p.y();
    pose_stamped.pose.position.z = estimator.wheel_only_odom.delta_p.z();
    pose_stamped.pose.orientation.x = estimator.wheel_only_odom.delta_q.x();
    pose_stamped.pose.orientation.y = estimator.wheel_only_odom.delta_q.y();
    pose_stamped.pose.orientation.z = estimator.wheel_only_odom.delta_q.z();
    pose_stamped.pose.orientation.w = estimator.wheel_only_odom.delta_q.w();
    wheel_path.header = header;
    wheel_path.header.frame_id = "wheel_odom";
    wheel_path.poses.emplace_back(pose_stamped);
    pub_wheel_path.publish(wheel_path);

    // publish wheel-imu-fusion odometry path
    pose_stamped.header = header;
    pose_stamped.header.frame_id = "wheel_imu_odom";
    pose_stamped.pose.position.x = estimator.wheel_imu_odom.delta_p.x();
    pose_stamped.pose.position.y = estimator.wheel_imu_odom.delta_p.y();
    pose_stamped.pose.position.z = estimator.wheel_imu_odom.delta_p.z();
    pose_stamped.pose.orientation.x = estimator.wheel_imu_odom.delta_q.x();
    pose_stamped.pose.orientation.y = estimator.wheel_imu_odom.delta_q.y();
    pose_stamped.pose.orientation.z = estimator.wheel_imu_odom.delta_q.z();
    pose_stamped.pose.orientation.w = estimator.wheel_imu_odom.delta_q.w();
    wheel_imu_path.header = header;
    wheel_imu_path.header.frame_id = "wheel_imu_odom";
    wheel_imu_path.poses.emplace_back(pose_stamped);
    pub_wheel_imu_path.publish(wheel_imu_path);

    // publish wheel-imu-fusion odometry 3D path
    pose_stamped.header = header;
    pose_stamped.header.frame_id = "wheel_imu_odom3D";
    pose_stamped.pose.position.x = estimator.wheel_imu_odom3D.delta_p.x();
    pose_stamped.pose.position.y = estimator.wheel_imu_odom3D.delta_p.y();
    pose_stamped.pose.position.z = estimator.wheel_imu_odom3D.delta_p.z();
    pose_stamped.pose.orientation.x = estimator.wheel_imu_odom3D.delta_q.x();
    pose_stamped.pose.orientation.y = estimator.wheel_imu_odom3D.delta_q.y();
    pose_stamped.pose.orientation.z = estimator.wheel_imu_odom3D.delta_q.z();
    pose_stamped.pose.orientation.w = estimator.wheel_imu_odom3D.delta_q.w();
    wheel_imu_path3D.header = header;
    wheel_imu_path3D.header.frame_id = "wheel_imu_odom3D";
    wheel_imu_path3D.poses.emplace_back(pose_stamped);
    pub_wheel_imu_path3D.publish(wheel_imu_path3D);

    static tf::TransformBroadcaster br;
    tf::Transform transform;

    // broadcast base_footprint -> wheel_odom
    Affine3d T_wheelodom_base =
            Translation3d(estimator.wheel_only_odom.delta_p)
            * estimator.wheel_only_odom.delta_q;
    Affine3d T_base_wheelodom = T_wheelodom_base.inverse();
    transform.setOrigin({ T_base_wheelodom.translation().x(),
                          T_base_wheelodom.translation().y(),
                          T_base_wheelodom.translation().z() });
    Quaterniond q_base_wheelodom(T_base_wheelodom.rotation());
    transform.setRotation({ q_base_wheelodom.x(),
                            q_base_wheelodom.y(),
                            q_base_wheelodom.z(),
                            q_base_wheelodom.w()});
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "base_footprint", "wheel_odom"));

    // broadcast base_footprint -> wheel_imu_odom
    Affine3d T_wheelimuodom_base =
            Translation3d(estimator.wheel_imu_odom.delta_p)
            * estimator.wheel_imu_odom.delta_q;
    Affine3d T_base_wheelimuodom = T_wheelimuodom_base.inverse();
    transform.setOrigin({ T_base_wheelimuodom.translation().x(),
                          T_base_wheelimuodom.translation().y(),
                          T_base_wheelimuodom.translation().z() });
    Quaterniond q_base_wheelimuodom(T_base_wheelimuodom.rotation());
    transform.setRotation({ q_base_wheelimuodom.x(),
                            q_base_wheelimuodom.y(),
                            q_base_wheelimuodom.z(),
                            q_base_wheelimuodom.w()});
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "base_footprint", "wheel_imu_odom"));

    // broadcast base_footprint -> wheel_imu_odom3D
    Affine3d T_wheelimuodom3D_base = Translation3d(estimator.wheel_imu_odom3D.delta_p) * estimator.wheel_imu_odom3D.delta_q;
    Affine3d T_base_wheelimuodom3D = T_wheelimuodom3D_base.inverse();
    Quaterniond q_base_wheelimuodom3D(T_base_wheelimuodom3D.rotation());
    transform.setOrigin({ T_base_wheelimuodom3D.translation().x(),
                          T_base_wheelimuodom3D.translation().y(),
                          T_base_wheelimuodom3D.translation().z() });
    transform.setRotation({ q_base_wheelimuodom3D.x(),
                            q_base_wheelimuodom3D.y(),
                            q_base_wheelimuodom3D.z(),
                            q_base_wheelimuodom3D.w() });
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "base_footprint", "wheel_imu_odom3D"));

//    static std_msgs::Header prev_header;
//    static Vector3d prev_vel_wheelimuodom;
//    nav_msgs::Odometry msg;
//    msg.header = header;
////    msg.pose.pose.position.x = estimator.wheel_only_odom.measurements.back().velocity.second;
////    msg.pose.pose.position.y = estimator.wheel_imu_odom.measurements.back().velocity.second;
////    msg.pose.pose.orientation.x = estimator.wheel_only_odom.delta_yaw / M_PI * 180;
////    msg.pose.pose.orientation.y = estimator.wheel_imu_odom.delta_yaw / M_PI * 180;
//    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) {
//        // 世界坐标系下轮式里程计的速度
////        double yaw = Utility::R2ypr(estimator.Rs[WINDOW_SIZE])[0] / 180 * M_PI;
//        Vector3d vel_odom;
//        vel_odom << estimator.wheel_imu_odom3D.measurements.back().velocity.first.x(),
//                    estimator.wheel_imu_odom3D.measurements.back().velocity.first.y(), 0;
//        Vector3d vel_wheelimuodom = estimator.wheel_imu_odom3D.delta_q * vel_odom;
//        msg.twist.twist.linear.x = vel_wheelimuodom.x();
//        msg.twist.twist.linear.y = vel_wheelimuodom.y();
//        msg.twist.twist.linear.y = vel_wheelimuodom.z();
//
//        Quaterniond q_world_imu(estimator.Rs[WINDOW_SIZE]);
//        q_world_imu = q_world_imu * estimator.tmp_pre_integration->delta_q;
//        Quaterniond q_world_base(estimator.rib.inverse() * estimator.tmp_pre_integration->delta_q * estimator.rib);
//        Vector3d ypr = Utility::R2ypr(q_world_base.toRotationMatrix()) / 180 * M_PI;
//        double pitch = ypr[1];
//        double roll = ypr[2];
//        double yaw = ypr[0];
//        msg.twist.twist.angular.x = roll / M_PI * 180;
//        msg.twist.twist.angular.y = pitch / M_PI * 180;
//        msg.twist.twist.angular.z = yaw / M_PI * 180;
//
//        if (!prev_header.stamp.isZero()) {
//            Vector3d acc_wheelimuodom = (vel_wheelimuodom - prev_vel_wheelimuodom) / (header.stamp - prev_header.stamp).toSec();
//            msg.pose.pose.position.x = acc_wheelimuodom.x();
//            msg.pose.pose.position.y = acc_wheelimuodom.y();
//            msg.pose.pose.position.z = acc_wheelimuodom.norm();
//        }
//        prev_header = header;
//        prev_vel_wheelimuodom = vel_wheelimuodom;
//        pub_velocity_yaw.publish(msg);
//    }
}

void pubMixedOdom(const std_msgs::Header& header, const MixedOdomMeasurement& orig_m, const MixedOdomMeasurement& filted_m) {
    nav_msgs::Odometry orig_odom;
    orig_odom.header = header;
    orig_odom.header.frame_id = "base_footprint";
    orig_odom.twist.twist.linear.x = orig_m.velocity.first.x();
    orig_odom.twist.twist.linear.y = orig_m.velocity.first.y();
    orig_odom.twist.twist.angular.z = orig_m.velocity.second;
    pub_orig_odom.publish(orig_odom);

    nav_msgs::Odometry filted_odom;
    filted_odom.header = header;
    filted_odom.header.frame_id = "base_footprint";
    filted_odom.twist.twist.linear.x = filted_m.velocity.first.x();
    filted_odom.twist.twist.linear.y = filted_m.velocity.first.y();
    filted_odom.twist.twist.angular.z = filted_m.velocity.second;
    pub_filted_odom.publish(filted_odom);

    sensor_msgs::Imu orig_imu;
    orig_imu.header = header;
    orig_imu.header.frame_id = "base_footprint";
    orig_imu.linear_acceleration.x = orig_m.imu_linear_acceleration.x();
    orig_imu.linear_acceleration.y = orig_m.imu_linear_acceleration.y();
    orig_imu.linear_acceleration.z = orig_m.imu_linear_acceleration.z();
    orig_imu.angular_velocity.x = orig_m.imu_angular_velocity.x();
    orig_imu.angular_velocity.y = orig_m.imu_angular_velocity.y();
    orig_imu.angular_velocity.z = orig_m.imu_angular_velocity.z();
    pub_orig_imu.publish(orig_imu);

    sensor_msgs::Imu filted_imu;
    filted_imu.header = header;
    filted_imu.header.frame_id = "base_footprint";
    filted_imu.linear_acceleration.x = filted_m.imu_linear_acceleration.x();
    filted_imu.linear_acceleration.y = filted_m.imu_linear_acceleration.y();
    filted_imu.linear_acceleration.z = filted_m.imu_linear_acceleration.z();
    filted_imu.angular_velocity.x = filted_m.imu_angular_velocity.x();
    filted_imu.angular_velocity.y = filted_m.imu_angular_velocity.y();
    filted_imu.angular_velocity.z = filted_m.imu_angular_velocity.z();
    pub_filted_imu.publish(filted_imu);
}

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, const std_msgs::Header &header)
{
    Eigen::Quaterniond quadrotor_Q = Q ;

    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = quadrotor_Q.x();
    odometry.pose.pose.orientation.y = quadrotor_Q.y();
    odometry.pose.pose.orientation.z = quadrotor_Q.z();
    odometry.pose.pose.orientation.w = quadrotor_Q.w();
    odometry.twist.twist.linear.x = V.x();
    odometry.twist.twist.linear.y = V.y();
    odometry.twist.twist.linear.z = V.z();
    pub_latest_odometry.publish(odometry);
}

void printStatistics(const Estimator &estimator, double t)
{
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    printf("pos:%6.3f %6.3f %6.3f vel:%6.3f %6.3f %6.3f Ba:%6.3f %6.3f %6.3f Bg:%6.3f %6.3f %6.3f\r",
           estimator.Ps[WINDOW_SIZE].x(), estimator.Ps[WINDOW_SIZE].y(), estimator.Ps[WINDOW_SIZE].z(),
           estimator.Vs[WINDOW_SIZE].x(), estimator.Vs[WINDOW_SIZE].y(), estimator.Vs[WINDOW_SIZE].z(),
           estimator.Bas[WINDOW_SIZE].x(), estimator.Bas[WINDOW_SIZE].y(), estimator.Bas[WINDOW_SIZE].z(),
           estimator.Bgs[WINDOW_SIZE].x(), estimator.Bgs[WINDOW_SIZE].y(), estimator.Bgs[WINDOW_SIZE].z());
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

    //debug
    try {
        vector<double> window_info = {
                estimator.window_info.at("dt"),
                estimator.window_info.at("dp"),
                estimator.window_info.at("err_p"),
                estimator.window_info.at("err_v"),
                estimator.window_info.at("scale"),
                estimator.window_info.at("g"),
                estimator.window_info.at("gx"),
                estimator.window_info.at("gy"),
                estimator.window_info.at("gz"),
        };
        std_msgs::Float64MultiArray msg_window_info;
        msg_window_info.data = window_info;
        pub_window_info.publish(msg_window_info);
    } catch (std::out_of_range err) {

    }

    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        nav_msgs::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.child_frame_id = "body";
        Quaterniond tmp_Q;
        tmp_Q = Quaterniond(estimator.Rs[WINDOW_SIZE]);
        odometry.pose.pose.position.x = estimator.Ps[WINDOW_SIZE].x();
        odometry.pose.pose.position.y = estimator.Ps[WINDOW_SIZE].y();
        odometry.pose.pose.position.z = estimator.Ps[WINDOW_SIZE].z();
        odometry.pose.pose.orientation.x = tmp_Q.x();
        odometry.pose.pose.orientation.y = tmp_Q.y();
        odometry.pose.pose.orientation.z = tmp_Q.z();
        odometry.pose.pose.orientation.w = tmp_Q.w();
        // V^b = R^b_w * V^w
        Vector3d tmp_vel = estimator.Rs[WINDOW_SIZE].inverse() * estimator.Vs[WINDOW_SIZE];
        odometry.twist.twist.linear.x = tmp_vel.x();
        odometry.twist.twist.linear.y = tmp_vel.y();
        odometry.twist.twist.linear.z = tmp_vel.z();
        pub_odometry.publish(odometry);

        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose = odometry.pose.pose;
        path.header = header;
        path.header.frame_id = "world";
        path.poses.push_back(pose_stamped);
        pub_path.publish(path);

        // path of base_footprint
        pose_stamped.header = header;
        pose_stamped.header.frame_id = "world";
        Affine3d T_world_imu = Translation3d(estimator.Ps[WINDOW_SIZE]) * estimator.Rs[WINDOW_SIZE];
        Affine3d T_imu_base = Translation3d(estimator.tio) * estimator.rio;
        Affine3d T_world_base = T_world_imu * T_imu_base;
        Quaterniond q_world_base = Quaterniond(T_world_base.linear());
        pose_stamped.pose.position.x = T_world_base.translation().x();
        pose_stamped.pose.position.y = T_world_base.translation().y();
        pose_stamped.pose.position.z = T_world_base.translation().z();
        pose_stamped.pose.orientation.x = q_world_base.x();
        pose_stamped.pose.orientation.y = q_world_base.y();
        pose_stamped.pose.orientation.z = q_world_base.z();
        pose_stamped.pose.orientation.w = q_world_base.w();
        vio_base_path.header = header;
        vio_base_path.header.frame_id = "world";
        vio_base_path.poses.push_back(pose_stamped);
        pub_vio_base_path.publish(vio_base_path);

        Vector3d correct_t;
        Vector3d correct_v;
        Quaterniond correct_q;
        correct_t = estimator.drift_correct_r * estimator.Ps[WINDOW_SIZE] + estimator.drift_correct_t;
        correct_q = estimator.drift_correct_r * estimator.Rs[WINDOW_SIZE];
        odometry.pose.pose.position.x = correct_t.x();
        odometry.pose.pose.position.y = correct_t.y();
        odometry.pose.pose.position.z = correct_t.z();
        odometry.pose.pose.orientation.x = correct_q.x();
        odometry.pose.pose.orientation.y = correct_q.y();
        odometry.pose.pose.orientation.z = correct_q.z();
        odometry.pose.pose.orientation.w = correct_q.w();

        pose_stamped.pose = odometry.pose.pose;
        relo_path.header = header;
        relo_path.header.frame_id = "world";
        relo_path.poses.push_back(pose_stamped);
        pub_relo_path.publish(relo_path);

        // debug pub bias imu
        sensor_msgs::Imu msg;
        msg.header = header;
        msg.angular_velocity.x = estimator.Bgs[WINDOW_SIZE].x();
        msg.angular_velocity.y = estimator.Bgs[WINDOW_SIZE].y();
        msg.angular_velocity.z = estimator.Bgs[WINDOW_SIZE].z();
        msg.linear_acceleration.x = estimator.Bas[WINDOW_SIZE].x();
        msg.linear_acceleration.y = estimator.Bas[WINDOW_SIZE].y();
        msg.linear_acceleration.z = estimator.Bas[WINDOW_SIZE].z();
        pub_bias.publish(msg);
        // debug pub lastest keyframe translation
        nav_msgs::Odometry msg_imu_predict;
        msg_imu_predict.header = header;
        msg_imu_predict.pose.pose.position.x = estimator.imu_predict_P.x();
        msg_imu_predict.pose.pose.position.y = estimator.imu_predict_P.y();
        msg_imu_predict.pose.pose.position.z = estimator.imu_predict_P.z();
        pub_imu_predict.publish(msg_imu_predict);
        nav_msgs::Odometry msg_optimized;
        msg_optimized.header = header;
        msg_optimized.pose.pose.position.x = estimator.optimized_P.x();
        msg_optimized.pose.pose.position.y = estimator.optimized_P.y();
        msg_optimized.pose.pose.position.z = estimator.optimized_P.z();
        pub_optimized.publish(msg_optimized);
        nav_msgs::Odometry msg_wheel_predict;
        msg_wheel_predict.header = header;
        msg_wheel_predict.pose.pose.position.x = estimator.wheel_predict_P.x();
        msg_wheel_predict.pose.pose.position.y = estimator.wheel_predict_P.y();
        pub_wheel_predict.publish(msg_wheel_predict);
        nav_msgs::Odometry msg_wheel_imu_predict;
        msg_wheel_imu_predict.header = header;
        msg_wheel_imu_predict.pose.pose.position.x = estimator.wheel_imu_predict_P.x();
        msg_wheel_imu_predict.pose.pose.position.y = estimator.wheel_imu_predict_P.y();
        msg_wheel_imu_predict.pose.pose.position.z = estimator.wheel_imu_predict_P.z();
        pub_wheel_imu_predict.publish(msg_wheel_imu_predict);


        // write result to file
        ofstream foutC(VINS_RESULT_PATH, ios::app);
        foutC.setf(ios::fixed, ios::floatfield);
        foutC.precision(0);
        foutC << header.stamp.toSec() * 1e9 << ",";
        foutC.precision(5);
        foutC << estimator.Ps[WINDOW_SIZE].x() << ","
              << estimator.Ps[WINDOW_SIZE].y() << ","
              << estimator.Ps[WINDOW_SIZE].z() << ","
              << tmp_Q.w() << ","
              << tmp_Q.x() << ","
              << tmp_Q.y() << ","
              << tmp_Q.z() << ","
              << estimator.Vs[WINDOW_SIZE].x() << ","
              << estimator.Vs[WINDOW_SIZE].y() << ","
              << estimator.Vs[WINDOW_SIZE].z() << "," << endl;
        foutC.close();
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
        geometry_msgs::Point pose_marker;
        Vector3d correct_pose;
        correct_pose = estimator.key_poses[i];
        pose_marker.x = correct_pose.x();
        pose_marker.y = correct_pose.y();
        pose_marker.z = correct_pose.z();
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
            Eigen::Affine3d T_wheel = Eigen::Translation3d(Eigen::Vector3d::Zero()) * Eigen::Matrix3d::Zero();
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
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();

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
    static tf::TransformBroadcaster br;
    tf::Transform transform;

    tf::Quaternion q;
    // camera frame
    transform.setOrigin(tf::Vector3(estimator.tic[0].x(),
                                    estimator.tic[0].y(),
                                    estimator.tic[0].z()));
    q.setW(Quaterniond(estimator.ric[0]).w());
    q.setX(Quaterniond(estimator.ric[0]).x());
    q.setY(Quaterniond(estimator.ric[0]).y());
    q.setZ(Quaterniond(estimator.ric[0]).z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "body", "camera"));

    // base_footprint frame (T^imu_base)
    transform.setOrigin(tf::Vector3(estimator.tio.x(),
                                    estimator.tio.y(),
                                    estimator.tio.z()));
    q.setW(Quaterniond(estimator.rio).w());
    q.setX(Quaterniond(estimator.rio).x());
    q.setY(Quaterniond(estimator.rio).y());
    q.setZ(Quaterniond(estimator.rio).z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "body", "base_footprint"));

    // world_base to world_base_origin
    auto t_origin_base = estimator.base_integration_before_init.delta_p;
    auto q_origin_base = estimator.base_integration_before_init.delta_q;
    auto q_base_origin = q_origin_base.inverse();
    auto t_base_origin = q_base_origin * -t_origin_base;
    transform.setOrigin(tf::Vector3(t_base_origin.x(),
                                    t_base_origin.y(),
                                    t_base_origin.z()));
    q.setW(q_base_origin.w());
    q.setX(q_base_origin.x());
    q.setY(q_base_origin.y());
    q.setZ(q_base_origin.z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "world_base", "world_origin"));

    if( estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) {
        // body frame
        transform.setOrigin(tf::Vector3(estimator.Ps[WINDOW_SIZE](0),
                                        estimator.Ps[WINDOW_SIZE](1),
                                        estimator.Ps[WINDOW_SIZE](2)));
        Quaterniond q_world_body(estimator.Rs[WINDOW_SIZE]);
        q.setW(q_world_body.w());
        q.setX(q_world_body.x());
        q.setY(q_world_body.y());
        q.setZ(q_world_body.z());
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, header.stamp, "world", "body"));
        // world (imu world) to world_base
        auto t_world_worldbase = estimator.init_orientation * estimator.tio;
        Quaterniond q_world_worldbase = estimator.init_orientation * Quaterniond(estimator.rio);
        transform.setOrigin(tf::Vector3(t_world_worldbase.x(),
                                        t_world_worldbase.y(),
                                        t_world_worldbase.z()));
        q.setW(q_world_worldbase.w());
        q.setX(q_world_worldbase.x());
        q.setY(q_world_worldbase.y());
        q.setZ(q_world_worldbase.z());
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, header.stamp, "world", "world_base"));
    } else {
        transform.setOrigin(tf::Vector3(0, 0, 0));
        auto ypr_world_body = Utility::R2ypr(estimator.rio.inverse());
        ypr_world_body[0] = 0;
        Quaterniond q_world_body(Utility::ypr2R(ypr_world_body));
        q.setW(q_world_body.w());
        q.setX(q_world_body.x());
        q.setY(q_world_body.y());
        q.setZ(q_world_body.z());
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, header.stamp, "world", "body"));

        // world (imu world) to world_base
        auto ypr_world_worldbase = Utility::R2ypr(estimator.rio.inverse());
        ypr_world_worldbase[1] = 0;
        ypr_world_worldbase[2] = 0;
        Quaterniond q_world_worldbase(Utility::ypr2R(ypr_world_worldbase).inverse());
        auto t_world_worldbase = q_world_body * estimator.tio;
        transform.setOrigin(tf::Vector3(t_world_worldbase.x(),
                                        t_world_worldbase.y(),
                                        t_world_worldbase.z()));
        q.setW(q_world_worldbase.w());
        q.setX(q_world_worldbase.x());
        q.setY(q_world_worldbase.y());
        q.setZ(q_world_worldbase.z());
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, header.stamp, "world", "world_base"));
    }

    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = estimator.tic[0].x();
    odometry.pose.pose.position.y = estimator.tic[0].y();
    odometry.pose.pose.position.z = estimator.tic[0].z();
    Quaterniond tmp_q{estimator.ric[0]};
    odometry.pose.pose.orientation.x = tmp_q.x();
    odometry.pose.pose.orientation.y = tmp_q.y();
    odometry.pose.pose.orientation.z = tmp_q.z();
    odometry.pose.pose.orientation.w = tmp_q.w();
    pub_extrinsic.publish(odometry);

}

void pubKeyframe(const Estimator &estimator)
{
    // pub camera pose, 2D-3D points of keyframe
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR && estimator.marginalization_flag == 0)
    {
        int i = WINDOW_SIZE - 2;
        //Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Vector3d P = estimator.Ps[i];
        Quaterniond R = Quaterniond(estimator.Rs[i]);

        nav_msgs::Odometry odometry;
        odometry.header = estimator.Headers[WINDOW_SIZE - 2];
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();
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
    odometry.pose.pose.position.x = estimator.relo_relative_t.x();
    odometry.pose.pose.position.y = estimator.relo_relative_t.y();
    odometry.pose.pose.position.z = estimator.relo_relative_t.z();
    odometry.pose.pose.orientation.x = estimator.relo_relative_q.x();
    odometry.pose.pose.orientation.y = estimator.relo_relative_q.y();
    odometry.pose.pose.orientation.z = estimator.relo_relative_q.z();
    odometry.pose.pose.orientation.w = estimator.relo_relative_q.w();
    odometry.twist.twist.linear.x = estimator.relo_relative_yaw;
    odometry.twist.twist.linear.y = estimator.relo_frame_index;

    pub_relo_relative_pose.publish(odometry);
}