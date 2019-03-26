//
// Created by bxwllzz on 18-11-28.
//

#ifndef VINS_ESTIMATOR_BASE_ODOM_INTEGRATION_H
#define VINS_ESTIMATOR_BASE_ODOM_INTEGRATION_H

#include <Eigen/Dense>
#include <deque>
#include <exception>
#include <std_msgs/Header.h>

static inline std::ostream& operator<<(std::ostream&                             output,
                                       const std::pair<Eigen::Vector2d, double>& xyw) {
    output << "{(" << xyw.first.x() << ", " << xyw.first.y() << "), " << xyw.second << "}";
    return output;
}

struct MixedOdomMeasurement {
    double                             dt                      = 0;
    std::pair<Eigen::Vector2d, double> velocity                = { { 0, 0 }, 0 };   // base odom frame
    double constraint_error_vel                                = 0;
    Eigen::Vector3d                    imu_angular_velocity    = { 0, 0, 0 };       // angular_velocity in IMU frame
    Eigen::Vector3d                    imu_linear_acceleration = { 0, 0, 0 };       // linear_acceleration in IMU frame
//    Eigen::Vector2d                    noise                   = { 1e-3, 1e-3 };

    MixedOdomMeasurement() = default;

    // only wheel odometry is known
//    MixedOdomMeasurement(double                             dt_,
//                         std::pair<Eigen::Vector2d, double> base_odom_velocity,
//                         double                             constraint_error_vel_)
//        : dt(dt_)
//        , velocity(std::move(base_odom_velocity))
//        , constraint_error_vel(constraint_error_vel_)
//        , imu_angular_velocity(0, 0, base_odom_velocity.second) {}

    // wheel odometry and imu is known
    MixedOdomMeasurement(double                             dt_,
                         std::pair<Eigen::Vector2d, double> base_odom_velocity,
                         double                             constraint_error_vel_,
                         Eigen::Vector3d                    imu_angular_velocity_,
                         Eigen::Vector3d                    imu_linear_acceleration_)
        : dt(dt_)
        , velocity(std::move(base_odom_velocity))
        , constraint_error_vel(constraint_error_vel_)
        , imu_angular_velocity(std::move(imu_angular_velocity_))
        , imu_linear_acceleration(std::move(imu_linear_acceleration_)) {}

    //    MixedOdomMeasurement& scale(const Eigen::Matrix3d& scale) {
    //        Eigen::Vector3d result = scale * Eigen::Vector3d(velocity.first[0], velocity.first[1], velocity.second);
    //        velocity.first         = result.head<2>();
    //        velocity.second        = result[2];
    //        return *this;
    //    }

    bool is_still() const {
        return velocity.first.norm() < 0.001 && velocity.second < 0.003;
    }
};

class BaseOdometryIntegration {
//public:
//    BaseOdometryIntegration() {}
//
//    void push_back(const MixedOdomMeasurement& m);
//
//    void propagate(const MixedOdomMeasurement& m);
//
//    void repropagate();
//
//public:
//    std::vector<MixedOdomMeasurement> measurements;
////    Eigen::Vector2d                   noise; // std of displacement and yaw angle of wheel-odom, guese from gyro and accel
//    double                            sum_dt          = 0;
//    Eigen::Vector2d                   delta_p         = { 0, 0 }; // fused displacement from wheel-odom linear velocity and gyro angular velocity
//    double                            delta_yaw_imu   = 0;        // yaw angle integral from gyro
//    double                            delta_yaw_wheel = 0;        // yaw angle integral from wheel-odom
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
    explicit BaseOdometryIntegration3D(const Eigen::Vector3d& _bg = Eigen::Vector3d::Zero());

    void push_back(const MixedOdomMeasurement& m);

    void propagate(const MixedOdomMeasurement& m);

    void repropagate(const Eigen::Vector3d& _bg);

    Eigen::Matrix<double, 3, 1> evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Bgi,
                                          const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, bool debug = false);

public:
    std::vector<MixedOdomMeasurement> measurements;

    // V_measure=scale*V_true
    Eigen::Matrix3d scale;
    Eigen::Vector3d linearized_bg = {0, 0, 0};

    bool               still   = true;
    double             sum_dt  = 0;
    Eigen::Vector3d    delta_p = { 0, 0, 0 };
    Eigen::Quaterniond delta_q = { 1, 0, 0, 0 };

    Eigen::Matrix<double, 9, 9> covariance_intergral = Eigen::Matrix<double, 9, 9>::Zero();
    Eigen::Matrix<double, 9, 9> covariance = Eigen::Matrix<double, 9, 9>::Zero();
    Eigen::Matrix<double, 9, 9> jacobian = Eigen::Matrix<double, 9, 9>::Identity();

    Eigen::Matrix<double, 9, 9> F, G, noise;
};

//template <typename T = double>
//class FIR {
//private:
//    const std::vector<double> coefs;
//    const T                   zero;
//    std::deque<T>             history;
//
//public:
//    FIR(std::vector<double> _coefs, T _zero)
//        : coefs(std::move(_coefs))
//        , zero(_zero) {
//        history.resize(coefs.size(), zero);
//    }
//    void reset() {
//        history.clear();
//        history.resize(coefs.size(), zero);
//    }
//    T filt(T input) {
//        T output = zero;
//        history.push_back(input);
//        history.pop_front();
//        for (int i = 0; i < coefs.size(); i++) {
//            output += coefs[i] * history[i];
//        }
//        return output;
//    }
//};

//class WheelOdometryNoiseAnalyser {
//public:
//    WheelOdometryNoiseAnalyser(double max_history)
//        : max_history_(max_history) {}
//
//    void reset() {
//        filter_vxy_.reset();
//        filter_vw_.reset();
//        filter_accel_.reset();
//        filter_gyro_.reset();
//        sum_dt_ = 0;
//        filted_.clear();
//    }
//
//    Eigen::Vector2d update(MixedOdomMeasurement m) {
//        sum_dt_ += m.dt;
//        m.velocity.first          = filter_vxy_.filt(m.velocity.first);
//        m.velocity.second         = filter_vw_.filt(m.velocity.second);
//        m.imu_linear_acceleration = filter_accel_.filt(m.imu_linear_acceleration);
//        m.imu_angular_velocity    = filter_gyro_.filt(m.imu_angular_velocity);
//        filted_.emplace_back(std::move(m));
//
//        while (sum_dt_ - filted_.front().dt >= max_history_) {
//            sum_dt_ -= filted_.front().dt;
//            filted_.pop_front();
//        }
//
//        return { 0, 0 };
//    }
//
//    std::pair<double, double> std() {
//        return { 0, 0 };
//    }
//
//public:
//    // lowpass filter Fpass=0.1(1dB) Fstop=0.2(15dB)
//    //    const std::vector<double> filter_coefs = { { -0.08355278612542, 0.0993111148786, 0.2135570173485, 0.305500065286,
//    //                                                       0.305500065286, 0.2135570173485, 0.0993111148786, -0.08355278612542 } };
//    // lowpass filter Fpass=0.025(1dB) Fstop=0.05(15dB) delay:
//    const std::vector<double> filter_coefs = { { -0.09404098392552, 0.007971715531977, 0.01039641248071, 0.01441145436184,
//                                                 0.01981611642775, 0.0262899203103, 0.03359681499331, 0.04134697091977,
//                                                 0.04910822475612, 0.05683906377275, 0.06379674860745, 0.06982392711811,
//                                                 0.07462764407637, 0.07791435172247, 0.07963951765543, 0.07963951765543,
//                                                 0.07791435172247, 0.07462764407637, 0.06982392711811, 0.06379674860745,
//                                                 0.05683906377275, 0.04910822475612, 0.04134697091977, 0.03359681499331,
//                                                 0.0262899203103, 0.01981611642775, 0.01441145436184, 0.01039641248071,
//                                                 0.007971715531977, -0.09404098392552 } };
//    const double              max_history_;
//
//    FIR<Eigen::Vector2d> filter_vxy_   = { filter_coefs, Eigen::Vector2d::Zero() };
//    FIR<double>          filter_vw_    = { filter_coefs, 0 };
//    FIR<Eigen::Vector3d> filter_accel_ = { filter_coefs, Eigen::Vector3d::Zero() };
//    FIR<Eigen::Vector3d> filter_gyro_  = { filter_coefs, Eigen::Vector3d::Zero() };
//
//    double                           sum_dt_ = 0;
//    std::deque<MixedOdomMeasurement> filted_;
//};

#endif //VINS_ESTIMATOR_BASE_ODOM_INTEGRATION_H
