//
// Created by bxwllzz on 18-12-26.
//

#include "base_odom_integration.h"

#include "../parameters.h"
#include "../utility/utility.h"

using namespace std;
using namespace Eigen;

void BaseOdometryIntegration::push_back(const MixedOdomMeasurement& m) {
    measurements.emplace_back(m);
    propagate(measurements.back());
    //        ROS_INFO("%lX d_pos=(%f,%f) d_yaw=%f", (uint64_t)this, delta_p.x(), delta_p.y(), delta_yaw / M_PI * 180);
}

void BaseOdometryIntegration::propagate(const MixedOdomMeasurement& m) {
    auto            d_pose = integration(m.dt, m.velocity);
    Eigen::Affine2d T_0    = Eigen::Translation2d(delta_p) * Eigen::Rotation2Dd(delta_yaw_imu);
    Eigen::Affine2d T_01   = Eigen::Translation2d(d_pose.first) * Eigen::Rotation2Dd(d_pose.second);
    Eigen::Affine2d T_1    = T_0 * T_01;

    sum_dt += m.dt;
    delta_p       = T_1.translation();
    delta_yaw_imu = Eigen::Rotation2Dd(T_1.rotation()).angle();
}

void BaseOdometryIntegration::repropagate() {
    sum_dt        = 0;
    delta_p       = { 0, 0 };
    delta_yaw_imu = 0;
    for (auto& m : measurements) {
        propagate(m);
    }
}

BaseOdometryIntegration3D::BaseOdometryIntegration3D(const Eigen::Matrix3d& _scale, const Eigen::Vector3d& _bg)
    : scale(_scale), linearized_bg(_bg) {
    covariance.block<3, 3>(0, 0) += 0.003 * 0.003 *  Eigen::Matrix3d::Identity();
    //        ROS_WARN("new BaseOdometryIntegration %lX", (uint64_t)this);
}

void BaseOdometryIntegration3D::push_back(const MixedOdomMeasurement& m) {
    if (!m.is_still()) {
        still = false;
//        cout << '!still' << std::endl;
    }
    measurements.emplace_back(m);
    propagate(measurements.back());
    //        ROS_INFO("%lX d_pos=(%f,%f) d_yaw=%f", (uint64_t)this, delta_p.x(), delta_p.y(), delta_yaw / M_PI * 180);
}

void BaseOdometryIntegration3D::propagate(const MixedOdomMeasurement& m) {
    Quaterniond q_i_j;
    q_i_j.w() = 1;
    q_i_j.vec() = RIO.transpose() * (m.imu_angular_velocity - linearized_bg) * m.dt / 2;
    Vector3d    t_i_j(m.velocity.first.x() * m.dt,
                   m.velocity.first.y() * m.dt,
                   0);

    F.setIdentity();
    F.block<3, 3>(3, 3) += -Utility::skewSymmetric(RIO.transpose() * (m.imu_angular_velocity - linearized_bg)) * m.dt;
    F.block<3, 3>(3, 6) += -RIO.transpose() * m.dt;

    G.setZero();
    G.block<3, 3>(0, 0) = delta_q.toRotationMatrix();
    G.block<3, 3>(3, 3) = RIO.transpose();
    G.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();

    sum_dt += m.dt;
    delta_p += delta_q * t_i_j;
    delta_q = delta_q * q_i_j;

    double constraint_err_noise = m.constraint_error_vel * m.dt;
    double scale_noise = t_i_j.norm() * 0.001;
    double odometry_noise = std::max(constraint_err_noise, scale_noise);
    noise.setZero();
    noise.block<3, 3>(0, 0) = (odometry_noise * odometry_noise) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(3, 3) = (GYR_N * m.dt * GYR_N * m.dt) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(6, 6) = (GYR_W * m.dt * GYR_W * m.dt) * Eigen::Matrix3d::Identity();

    jacobian = F * jacobian;
    covariance_intergral = F * covariance_intergral * F.transpose() + G * noise * G.transpose();

    covariance = covariance_intergral;
    covariance.block<3, 3>(0, 0) += 0.003 * 0.003 *  Eigen::Matrix3d::Identity();
}

void BaseOdometryIntegration3D::repropagate(const Eigen::Matrix3d& _scale, const Eigen::Vector3d& _bg) {
    scale   = _scale;
    linearized_bg = _bg;
    sum_dt  = 0;
    delta_p = { 0, 0, 0 };
    delta_q = { 1, 0, 0, 0 };
    jacobian.setIdentity();
    covariance_intergral.setZero();
    covariance.setZero();
    covariance = covariance_intergral;
    covariance.block<3, 3>(0, 0) += 0.003 * 0.003 *  Eigen::Matrix3d::Identity();
    for (auto& m : measurements) {
        propagate(m);
    }
}

Eigen::Matrix<double, 3, 1> BaseOdometryIntegration3D::evaluate(
        const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Bgi,
        const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, bool debug) {

    Eigen::Matrix<double, 3, 1> residuals;

    Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(0, 6);

    Eigen::Vector3d dbg = Bgi - linearized_bg;

    Eigen::Vector3d corrected_delta_p = delta_p + dp_dbg * dbg;

    residuals.block<3, 1>(0, 0) =
              RIO.transpose() * Qi.inverse() * (Pj - Pi)
            + RIO.transpose() * Qi.inverse() * Qj * TIO
            - RIO.transpose() * TIO
            - corrected_delta_p;

    if (debug) {
        Eigen::Vector3d P_Oi_Oj = RIO.transpose() * Qi.inverse() * (Pj - Pi)
                       + RIO.transpose() * Qi.inverse() * Qj * TIO
                       - RIO.transpose() * TIO;
        Eigen::Vector3d ypr_Oi_Oj = Utility::R2ypr(RIO.transpose() * Qi.inverse() * Qj * RIO);
        Eigen::Vector3d ypr_delta_q = Utility::R2ypr(delta_q.toRotationMatrix());
        ROS_INFO("=========");
        ROS_INFO("T_Oi_Oj: %8.3f m %8.3f deg (z %8.3f m) YPR %8.3f %8.3f %8.3f",
                 P_Oi_Oj.segment<2>(0).norm(), atan2(P_Oi_Oj.y(), P_Oi_Oj.x()) / M_PI * 180, P_Oi_Oj.z(),
                 ypr_Oi_Oj.x(), ypr_Oi_Oj.y(), ypr_Oi_Oj.z());
        ROS_INFO("delta_T: %8.3f m %8.3f deg (z %8.3f m) YPR %8.3f %8.3f %8.3f",
                 delta_p.segment<2>(0).norm(), atan2(delta_p.y(), delta_p.x()) / M_PI * 180, delta_p.z(),
                 ypr_delta_q.x(), ypr_delta_q.y(), ypr_delta_q.z());
        ROS_INFO("residual: %8.3f m std %8.3f", residuals.norm(), sqrt(covariance(0, 0)));
    }

    return residuals;
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
