//
// Created by bxwllzz on 18-11-28.
//

#ifndef VINS_ESTIMATOR_BASE_ODOM_INTEGRATION_H
#define VINS_ESTIMATOR_BASE_ODOM_INTEGRATION_H

#include "../parameters.h"
#include "../utility/utility.h"

#include <ceres/ceres.h>

using namespace Eigen;

class BaseOdometryIntegration {
public:
    struct Measurement {
        double          dt               = 0;
        Eigen::Vector2d linear_velocity  = {};
        double          angular_velocity = 0;

        Measurement(double _dt, const Eigen::Vector2d& _linear_velocity, double _angular_velocity)
            : dt(_dt), linear_velocity(_linear_velocity), angular_velocity(_angular_velocity)
        {}

        Measurement& scale(const Eigen::Matrix3d& scale) {
            Eigen::Vector3d result = scale * Eigen::Vector3d(linear_velocity[0], linear_velocity[1], angular_velocity);
            linear_velocity        = result.head<2>();
            angular_velocity       = result[2];
            return *this;
        }
    };

    BaseOdometryIntegration(const Eigen::Vector2d& _pos_0, double _yaw_0, const Eigen::Matrix3d& _scale = Eigen::Matrix3d::Identity())
        : pos_0(_pos_0)
        , yaw_0(_yaw_0)
        , scale(_scale)
        , sum_dt(0.0)
        , delta_p(Eigen::Vector2d::Zero())
        , delta_yaw(0.0) {
    }

    void push_back(double dt, const Eigen::Vector2d& pos_1, double yaw_1) {
        Eigen::Affine2d T_0      = Eigen::Translation2d(pos_0) * Eigen::Rotation2Dd(yaw_0);
        Eigen::Affine2d T_1      = Eigen::Translation2d(pos_1) * Eigen::Rotation2Dd(yaw_1);
        Eigen::Affine2d T_01     = T_0.inverse() * T_1;
        auto            velocity = differential(dt, std::make_pair(T_01.translation(), yaw_1 - yaw_0));
        measurements.emplace_back(dt, velocity.first, velocity.second);
        propagate(measurements.back());
        pos_0 = pos_1;
        yaw_0 = yaw_1;
    }

    void propagate(const Measurement& m) {
        Measurement un_m = m;
        un_m.scale(scale);
        auto            d_pose = integration(un_m.dt, { un_m.linear_velocity, un_m.angular_velocity });
        Eigen::Affine2d T_0    = Eigen::Translation2d(delta_p) * Eigen::Rotation2Dd(delta_yaw);
        Eigen::Affine2d T_01   = Eigen::Translation2d(d_pose.first) * Eigen::Rotation2Dd(d_pose.second);
        Eigen::Affine2d T_1    = T_0 * T_01;

        sum_dt += m.dt;
        delta_p   = T_1.translation();
        delta_yaw = Eigen::Rotation2Dd(T_1.rotation()).angle();
    }

    void repropagate(const Eigen::Matrix3d& _scale) {
        scale     = _scale;
        sum_dt    = 0;
        delta_p   = { 0, 0 };
        delta_yaw = 0;
        for (auto& m : measurements) {
            propagate(m);
        }
    }

public:
    std::vector<Measurement> measurements;

    Eigen::Vector2d pos_0; // previous position
    double          yaw_0; // previous yaw

    Eigen::Matrix3d scale;

    Eigen::Matrix<double, 15, 15> jacobian, covariance;
    Eigen::Matrix<double, 18, 18> noise;

    double          sum_dt;
    Eigen::Vector2d delta_p;
    double          delta_yaw;

public:
    // INPUT: [d_pos, d_yaw] w.r.t. base odom frame, dt
    // OUTPUT: velocity of [x, y, yaw] w.r.t. base odom frame
    static inline std::pair<Eigen::Vector2d, double> differential(double dt, const std::pair<Eigen::Vector2d, double>& d_pose) {
        const Eigen::Vector2d& d_pos = d_pose.first;
        double                 d_yaw = std::remainder(d_pose.second, 2 * M_PI);

        Eigen::Vector2d linear_velocity;
        double          angular_velocity = d_yaw / dt;

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
        return std::make_pair(linear_velocity, angular_velocity);
    }

    // INPUT: velocity of [x, y, yaw] w.r.t. base odom frame
    // OUTPUT: [d_pos, d_yaw] w.r.t. base odom frame, dt
    static inline std::pair<Eigen::Vector2d, double> integration(double dt, const std::pair<Eigen::Vector2d, double>& velocity) {
        const Eigen::Vector2d& linear_velocity  = velocity.first;
        const double&          angular_velocity = velocity.second;

        Eigen::Vector2d d_pos;
        double          d_yaw = angular_velocity * dt;
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

        return std::make_pair(d_pos, d_yaw);
    }
};

#endif //VINS_ESTIMATOR_BASE_ODOM_INTEGRATION_H
