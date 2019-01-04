//
// Created by bxwllzz on 18-12-26.
//

#include "base_odom_integration.h"

#include "../parameters.h"
#include "../utility/utility.h"

using namespace std;
using namespace Eigen;

BaseOdometryIntegration::BaseOdometryIntegration(const Eigen::Matrix3d& _scale)
    : scale(_scale) {
    //        ROS_WARN("new BaseOdometryIntegration %lX", (uint64_t)this);
}

void BaseOdometryIntegration::push_back(const MergedOdomMeasurement& m) {
    measurements.emplace_back(m);
    propagate(measurements.back());
    //        ROS_INFO("%lX d_pos=(%f,%f) d_yaw=%f", (uint64_t)this, delta_p.x(), delta_p.y(), delta_yaw / M_PI * 180);
}

void BaseOdometryIntegration::propagate(const MergedOdomMeasurement& m) {
    MergedOdomMeasurement un_m = m;
    un_m.scale(scale);

    auto            d_pose = integration(un_m.dt, un_m.velocity);
    Eigen::Affine2d T_0    = Eigen::Translation2d(delta_p) * Eigen::Rotation2Dd(delta_yaw);
    Eigen::Affine2d T_01   = Eigen::Translation2d(d_pose.first) * Eigen::Rotation2Dd(d_pose.second);
    Eigen::Affine2d T_1    = T_0 * T_01;

    sum_dt += m.dt;
    delta_p   = T_1.translation();
    delta_yaw = Eigen::Rotation2Dd(T_1.rotation()).angle();
}

void BaseOdometryIntegration::repropagate(const Eigen::Matrix3d& _scale) {
    scale     = _scale;
    sum_dt    = 0;
    delta_p   = { 0, 0 };
    delta_yaw = 0;
    for (auto& m : measurements) {
        propagate(m);
    }
}

BaseOdometryIntegration3D::BaseOdometryIntegration3D(const Eigen::Matrix3d& _scale)
    : scale(_scale) {
    //        ROS_WARN("new BaseOdometryIntegration %lX", (uint64_t)this);
}

void BaseOdometryIntegration3D::push_back(const MergedOdomMeasurement& m) {
    measurements.emplace_back(m);
    propagate(measurements.back());
    //        ROS_INFO("%lX d_pos=(%f,%f) d_yaw=%f", (uint64_t)this, delta_p.x(), delta_p.y(), delta_yaw / M_PI * 180);
}

void BaseOdometryIntegration3D::propagate(const MergedOdomMeasurement& m) {
    MergedOdomMeasurement un_m = m;
    un_m.scale(scale);

    Quaterniond q_i_j(1,
                      un_m.imu_angular_velocity.x() * un_m.dt / 2,
                      un_m.imu_angular_velocity.y() * un_m.dt / 2,
                      un_m.imu_angular_velocity.z() * un_m.dt / 2);
    Vector3d    t_i_j(un_m.velocity.first.x() * un_m.dt,
                   un_m.velocity.first.y() * un_m.dt,
                   0);

    sum_dt += un_m.dt;
    delta_t += delta_q * t_i_j;
    delta_q = delta_q * q_i_j;
}

void BaseOdometryIntegration3D::repropagate(const Eigen::Matrix3d& _scale) {
    scale   = _scale;
    sum_dt  = 0;
    delta_t = { 0, 0, 0 };
    delta_q = { 1, 0, 0, 0 };
    for (auto& m : measurements) {
        propagate(m);
    }
}

//#define USE_CIRCULAR

pair<Eigen::Vector2d, double> BaseOdometryIntegration::differential(
    double                        dt,
    const pair<Vector2d, double>& pose_0,
    const pair<Vector2d, double>& pose_1) {
    pair<Vector2d, double> d_pose = { pose_1.first - pose_0.first, pose_1.second - pose_0.second };
    d_pose.first                  = Rotation2Dd(-pose_0.second) * d_pose.first;
    return differential(dt, d_pose);
}

// INPUT: [d_pos, d_yaw] w.r.t. base odom frame, dt
// OUTPUT: velocity of [x, y, yaw] w.r.t. base odom frame
pair<Vector2d, double> BaseOdometryIntegration::differential(double dt, const pair<Vector2d, double>& d_pose) {
    const Eigen::Vector2d& d_pos = d_pose.first;
    double                 d_yaw = std::remainder(d_pose.second, 2 * M_PI);

    Eigen::Vector2d linear_velocity;
    double          angular_velocity = d_yaw / dt;

#ifdef USE_CIRCULAR
    if (d_yaw == 0) {
        linear_velocity = d_pos / dt;
    } else {
        double radius               = d_pos.norm() / 2 / sin(d_yaw / 2);
        double linear_velocity_norm = radius * angular_velocity;
        double linear_velocity_angle;
        if (d_yaw < 0) {
            linear_velocity_angle = atan2(d_pos[1], d_pos[0]) + (M_PI - d_yaw) / 2 + M_PI / 2;
        } else {
            linear_velocity_angle = atan2(d_pos[1], d_pos[0]) + (M_PI - d_yaw) / 2 - M_PI / 2;
        }
        linear_velocity << cos(linear_velocity_angle) * linear_velocity_norm, sin(linear_velocity_angle) * linear_velocity_norm;
    }
#else
    linear_velocity = d_pos / dt;
#endif
    return std::make_pair(linear_velocity, angular_velocity);
}

pair<Vector2d, double> BaseOdometryIntegration::integration(
    double                        dt,
    const pair<Vector2d, double>& pose_0,
    const pair<Vector2d, double>& velocity) {
    auto d_pose  = integration(dt, velocity);
    d_pose.first = pose_0.first + Rotation2Dd(pose_0.second) * d_pose.first;
    d_pose.second += pose_0.second;
    return d_pose;
}

// INPUT: velocity of [x, y, yaw] w.r.t. base odom frame
// OUTPUT: [d_pos, d_yaw] w.r.t. base odom frame, dt
pair<Vector2d, double> BaseOdometryIntegration::integration(double dt, const pair<Vector2d, double>& velocity) {
    const Eigen::Vector2d& linear_velocity  = velocity.first;
    const double&          angular_velocity = velocity.second;

    Eigen::Vector2d d_pos;
    double          d_yaw = angular_velocity * dt;

#ifdef USE_CIRCULAR
    ROS_ASSERT(abs(d_yaw) < M_PI);

    if (d_yaw == 0) {
        d_pos = dt * linear_velocity;
    } else {
        double radius                = linear_velocity.norm() / angular_velocity;
        double linear_velocity_angle = atan2(linear_velocity[1], linear_velocity[0]);
        double d_pos_norm            = radius * sin(d_yaw / 2) * 2;
        double d_pos_angle;
        if (angular_velocity < 0) {
            d_pos_angle = linear_velocity_angle - M_PI / 2 - (M_PI - d_yaw) / 2;
        } else {
            d_pos_angle = linear_velocity_angle + M_PI / 2 - (M_PI - d_yaw) / 2;
        }
        d_pos << cos(d_pos_angle) * d_pos_norm, sin(d_pos_angle) * d_pos_norm;
    }
#else
    d_pos           = dt * linear_velocity;
#endif

    return std::make_pair(d_pos, d_yaw);
}
