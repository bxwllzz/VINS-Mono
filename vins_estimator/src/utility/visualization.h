#pragma once

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>
#include "CameraPoseVisualization.h"
#include <eigen3/Eigen/Dense>
#include "../estimator.h"
#include "../parameters.h"
#include <fstream>

//extern ros::Publisher pub_odometry;
//extern ros::Publisher pub_path, pub_pose;
//extern ros::Publisher pub_vio_base_path, pub_wheel_path, pub_wheel_imu_path;
//extern ros::Publisher pub_cloud, pub_map;
//extern ros::Publisher pub_key_poses;
//extern ros::Publisher pub_ref_pose, pub_cur_pose;
//extern ros::Publisher pub_key;
//extern nav_msgs::Path path;
//extern ros::Publisher pub_pose_graph;
//extern int IMAGE_ROW, IMAGE_COL;

namespace Eigen2ROS {
static inline geometry_msgs::Vector3 Vector3d2Vector3(const Vector3d& in) {
    geometry_msgs::Vector3 out;
    out.x = in.x();
    out.y = in.y();
    out.z = in.z();
    return out;
}

static inline geometry_msgs::Point Vector3d2Point(const Vector3d& in) {
    geometry_msgs::Point out;
    out.x = in.x();
    out.y = in.y();
    out.z = in.z();
    return out;
}

static inline geometry_msgs::Quaternion Q2wxyz(const Quaterniond& in) {
    geometry_msgs::Quaternion out;
    out.w = in.w();
    out.x = in.x();
    out.y = in.y();
    out.z = in.z();
    return out;
}

static inline geometry_msgs::Quaternion R2wxyz(const Matrix3d& in) {
    Quaterniond q(in);
    geometry_msgs::Quaternion out;
    out.w = q.w();
    out.x = q.x();
    out.y = q.y();
    out.z = q.z();
    return out;
}

static inline geometry_msgs::Pose Affine3d2Pose(const Affine3d& in) {
    geometry_msgs::Pose out;
    out.position = Vector3d2Point(in.translation());
    out.orientation = Q2wxyz(Quaterniond(in.rotation()));
    return out;
}

static inline Affine3d Pose2Affine3d(const geometry_msgs::Pose &pose) {
    return Translation3d(pose.position.x,
                         pose.position.y,
                         pose.position.z)
           * Quaterniond(pose.orientation.w,
                         pose.orientation.x,
                         pose.orientation.y,
                         pose.orientation.z);
}

static inline geometry_msgs::Pose Rp2Pose(const Matrix3d& R, const Vector3d& p) {
    geometry_msgs::Pose out;
    out.position = Vector3d2Point(p);
    out.orientation = Q2wxyz(Quaterniond(R));
    return out;
}
static inline geometry_msgs::Pose Rp2Pose(const Quaterniond& q, const Vector3d& p) {
    geometry_msgs::Pose out;
    out.position = Vector3d2Point(p);
    out.orientation = Q2wxyz(q);
    return out;
}

static inline tf::Transform Affine3d2Transform(const Affine3d &T) {
    Matrix3d M = T.rotation();
    Vector3d p = T.translation();
    return tf::Transform(tf::Matrix3x3(M(0, 0), M(0, 1), M(0, 2),
                                       M(1, 0), M(1, 1), M(1, 2),
                                       M(2, 0), M(2, 1), M(2, 2)),
                         tf::Vector3(p[0], p[1], p[2]));
}

}

void registerPub(ros::NodeHandle &n);

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, const std_msgs::Header &header);

void printStatistics(const Estimator &estimator, double t);

void pubOdometry(const Estimator &estimator, const std_msgs::Header &header);

void pubInitialGuess(const Estimator &estimator, const std_msgs::Header &header);

void pubKeyPoses(const Estimator &estimator, const std_msgs::Header &header);

void pubCameraPose(const Estimator &estimator, const std_msgs::Header &header);

void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header);

void pubTF(const Estimator &estimator, const std_msgs::Header &header);

void pubKeyframe(const Estimator &estimator);

void pubRelocalization(const Estimator &estimator);

class PathTFPublisher {
public:
    Affine3d T_b_c;
    Affine3d T_b_o;
    Affine3d T_origin_worldbase;
    Affine3d T_worldbase_w;
    Affine3d T_w_b;
    Affine3d T_origin_o_wheel;
    Affine3d T_origin_o_wheelimu;
    Affine3d T_origin_o_wheelimu3D;
    Affine3d T_origin_o_vio;
    Affine3d T_origin_o_loop;

    ros::Subscriber sub_pose_graph_path_;
    tf::TransformBroadcaster br_;
    ros::Publisher pub_path_origin_o_wheel_;
    ros::Publisher pub_path_origin_o_wheelimu_;
    ros::Publisher pub_path_origin_o_wheelimu3D_;
    ros::Publisher pub_path_origin_o_vio_;
    ros::Publisher pub_path_origin_o_loop_;

    std_msgs::Header header_;
    nav_msgs::Path path_origin_o_wheel_;
    nav_msgs::Path path_origin_o_wheelimu_;
    nav_msgs::Path path_origin_o_wheelimu3D_;
    nav_msgs::Path path_origin_o_vio_;
    nav_msgs::Path path_origin_o_loop_;

public:
    explicit PathTFPublisher(ros::NodeHandle &nh);

    void on_estimator_update(const Estimator &estimator);

    void publish();

    void reset();

protected:
    void on_pose_graph_path(const nav_msgs::Path::ConstPtr &msg);
};