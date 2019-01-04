//
// Created by bxwllzz on 18-11-28.
//

#ifndef VINS_ESTIMATOR_BASE_ODOM_INTEGRATION_H
#define VINS_ESTIMATOR_BASE_ODOM_INTEGRATION_H

#include <Eigen/Dense>
#include <std_msgs/Header.h>

static inline std::ostream& operator<<(std::ostream&                             output,
                                       const std::pair<Eigen::Vector2d, double>& xyw) {
    output << "{(" << xyw.first.x() << ", " << xyw.first.y() << "), " << xyw.second << "}";
    return output;
}

struct MergedOdomMeasurement {
    double                             dt = 0;
    std::pair<Eigen::Vector2d, double> velocity = {{0, 0}, 0};
    Eigen::Vector3d                    imu_angular_velocity = {0, 0, 0};

    MergedOdomMeasurement() = default;

    MergedOdomMeasurement(double                             dt_,
                          std::pair<Eigen::Vector2d, double> base_odom_velocity)
        : dt(dt_)
        , velocity(std::move(base_odom_velocity))
        , imu_angular_velocity(0, 0, base_odom_velocity.second) {}

    MergedOdomMeasurement(double                                    dt_,
                          const std::pair<Eigen::Vector2d, double>& base_odom_velocity,
                          const Eigen::Vector3d&                    imu_angular_velocity,
                          const Eigen::Matrix3d&                    r_base_imu)
        : dt(dt_)
        , velocity({ base_odom_velocity.first, (r_base_imu * imu_angular_velocity).z() })
        , imu_angular_velocity(r_base_imu * imu_angular_velocity) {}

    MergedOdomMeasurement& scale(const Eigen::Matrix3d& scale) {
        Eigen::Vector3d result = scale * Eigen::Vector3d(velocity.first[0], velocity.first[1], velocity.second);
        velocity.first         = result.head<2>();
        velocity.second        = result[2];
        return *this;
    }
};

class BaseOdometryIntegration {
public:
    explicit BaseOdometryIntegration(const Eigen::Matrix3d& _scale = Eigen::Matrix3d::Identity());

    void push_back(const MergedOdomMeasurement& m);

    void propagate(const MergedOdomMeasurement& m);

    void repropagate(const Eigen::Matrix3d& _scale);

public:
    std::vector<MergedOdomMeasurement> measurements;

    Eigen::Matrix3d scale;

    Eigen::Matrix<double, 15, 15> jacobian, covariance;
    Eigen::Matrix<double, 18, 18> noise;

    double          sum_dt    = 0;
    Eigen::Vector2d delta_p   = { 0, 0 };
    double          delta_yaw = 0;

public:
    static std::pair<Eigen::Vector2d, double> differential(double dt, const std::pair<Eigen::Vector2d, double>& pose_0, const std::pair<Eigen::Vector2d, double>& pose_1);

    // INPUT: [d_pos, d_yaw] w.r.t. base odom frame, dt
    // OUTPUT: velocity of [x, y, yaw] w.r.t. base odom frame
    static std::pair<Eigen::Vector2d, double> differential(double dt, const std::pair<Eigen::Vector2d, double>& d_pose);

    static std::pair<Eigen::Vector2d, double> integration(double dt, const std::pair<Eigen::Vector2d, double>& pose_0, const std::pair<Eigen::Vector2d, double>& velocity);

    // INPUT: velocity of [x, y, yaw] w.r.t. base odom frame
    // OUTPUT: [d_pos, d_yaw] w.r.t. base odom frame, dt
    static std::pair<Eigen::Vector2d, double> integration(double dt, const std::pair<Eigen::Vector2d, double>& velocity);
};

class BaseOdometryIntegration3D {
public:
    explicit BaseOdometryIntegration3D(const Eigen::Matrix3d& _scale = Eigen::Matrix3d::Identity());

    void push_back(const MergedOdomMeasurement& m);

    void propagate(const MergedOdomMeasurement& m);

    void repropagate(const Eigen::Matrix3d& _scale);

public:
    std::vector<MergedOdomMeasurement> measurements;

    Eigen::Matrix3d scale;

    double             sum_dt  = 0;
    Eigen::Vector3d    delta_t = { 0, 0, 0 };
    Eigen::Quaterniond delta_q = { 1, 0, 0, 0 };
};

#endif //VINS_ESTIMATOR_BASE_ODOM_INTEGRATION_H
