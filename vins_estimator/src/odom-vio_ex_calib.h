//
// Created by bxwllzz on 19-3-22.
//

#ifndef VINS_ESTIMATOR_ODOM_VIO_EX_CALIB_H
#define VINS_ESTIMATOR_ODOM_VIO_EX_CALIB_H

#include <stdexcept>
#include <map>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <ros/ros.h>
#include <nav_msgs/Path.h>

#include "utility/utility.h"

class WheelOdomVIOAlignment {
public:
    Eigen::Affine3d guese_T_o_b_;
    std::map<double, Eigen::Affine3d> m_wo_;
    std::map<double, Eigen::Affine3d> m_vio_;
    std::vector<std::pair<double, double>> ts_;

    ros::Time start_t_;

    Eigen::Vector3d g_b_;       // gracity in body frame
    Eigen::Vector3d ypr_o_b_;
    Eigen::Vector3d p_o_b;
    Eigen::Vector2d scale_;

public:
    WheelOdomVIOAlignment(Eigen::Affine3d guese_T_o_b)
        : guese_T_o_b_(guese_T_o_b) {
    }

    void process_msg_path_wo(const nav_msgs::Path &path, double skip = 0) {
        ros::Time t0;
        for (const auto &m : path.poses) {
            if (start_t_.isZero())
                start_t_ = m.header.stamp;
            double t = (m.header.stamp - start_t_).toSec();
            if (t0.isZero()) {
                t0 = m.header.stamp;
            }
            if ((m.header.stamp - t0).toSec() < skip)
                continue;
            Eigen::Affine3d tf = Eigen::Translation3d(m.pose.position.x,
                                                      m.pose.position.y,
                                                      m.pose.position.z)
                                 * Eigen::Quaterniond(m.pose.orientation.w,
                                                      m.pose.orientation.x,
                                                      m.pose.orientation.y,
                                                      m.pose.orientation.z);
            if (m_wo_.find(t) != m_wo_.end() && m_wo_[t].matrix() != tf.matrix()) {
                ROS_WARN("Same t with diff transform");
            }
            m_wo_[t] = tf;
        }
        std::cout << "Wheel odom path: " << m_wo_.size() << std::endl;
        update_ts();
    }

    void process_msg_path_vio(const nav_msgs::Path &path, double skip = 0) {
        ros::Time t0;
        for (const auto &m : path.poses) {
            if (start_t_.isZero())
                start_t_ = m.header.stamp;
            double t = (m.header.stamp - start_t_).toSec();
            if (t0.isZero()) {
                t0 = m.header.stamp;
            }
            if ((m.header.stamp - t0).toSec() < skip)
                continue;
            Eigen::Affine3d tf = Eigen::Translation3d(m.pose.position.x,
                                                      m.pose.position.y,
                                                      m.pose.position.z)
                                 * Eigen::Quaterniond(m.pose.orientation.w,
                                                      m.pose.orientation.x,
                                                      m.pose.orientation.y,
                                                      m.pose.orientation.z);
            if (m_vio_.find(t) != m_vio_.end() && m_vio_[t].matrix() != tf.matrix()) {
                ROS_WARN("Same t with diff transform");
            }
            m_vio_[t] = tf;
        }
        std::cout << "Visual odom path: " << m_vio_.size() << std::endl;
        update_ts();
    }

    void update_ts() {
        double max_dt = 0.02;
        ts_.clear();
        if (m_wo_.empty() || m_vio_.empty()) {
            return;
        }

        auto it = m_wo_.begin();
        for (const auto& vio : m_vio_) {
            while (it != m_wo_.end() && it->first < vio.first)
                it++;
            auto tmp_it = it;
            double t_prev = (it == m_wo_.begin()) ? it->first : (--tmp_it)->first;
            double t = vio.first;
            double t_after = (it == m_wo_.end()) ? (--tmp_it)->first : it->first;
            if (std::min(std::abs(t_prev - t), std::abs(t_after - t)) > max_dt)
                continue;
            if (std::abs(t_prev - t) < std::abs(t_after - t)) {
                ts_.push_back({t_prev, t});
            } else {
                ts_.push_back({t_after, t});
            }
//            printf("%d: %lf-%lf=%lf\n", ts_.size(), ts_.back().first, ts_.back().second, std::abs(ts_.back().first - ts_.back().second));
        }
    }

    // Principal Component Analysis to solve main_rot_axis
    Eigen::Vector3d vio_main_rot_axis() {
        if (m_vio_.empty())
            throw std::out_of_range("m_vio_.empty()");

        Eigen::Matrix<double, Eigen::Dynamic, 3> X(m_vio_.size() * 2, 3);
        auto it_i = m_vio_.begin();
        auto it_j = ++m_vio_.begin();
        for (int i = 0; it_j != m_vio_.end(); it_i++, it_j++, i++) {
            Eigen::Matrix3d R_w_i = it_i->second.linear();
            Eigen::Matrix3d R_w_j = it_j->second.linear();
            Eigen::AngleAxisd R_i_j(R_w_i.transpose() * R_w_j);
            Eigen::Vector3d axis_i_j(R_i_j.angle() * R_i_j.axis());
            X.row(i) = axis_i_j.transpose();
            // duplicated with reverse to ensure mean=0
            X.row(i + m_vio_.size()) = -axis_i_j.transpose();
        }

        // covariance
        Eigen::Matrix3d cov = (X.adjoint() * X) / X.rows();

        // eigenvector corresponde to max eigenvalue
        Eigen::EigenSolver<Eigen::Matrix3d> eigen_solver(cov);
        int max_eigenvalue_index = 0;
        for (int i = 0; i < eigen_solver.eigenvalues().rows(); i++) {
            if (eigen_solver.eigenvalues()[i].real() > eigen_solver.eigenvalues()[max_eigenvalue_index].real()) {
                max_eigenvalue_index = i;
            }
        }
        Eigen::Matrix<std::complex<double>, 3, 1> max_eigenvector = eigen_solver.eigenvectors().col(
                max_eigenvalue_index);

        Eigen::Vector3d main_rot_axis;
        main_rot_axis << max_eigenvector[0].real(),
                max_eigenvector[1].real(),
                max_eigenvector[2].real();
        ROS_INFO_STREAM("VIO mean rot axis: " << main_rot_axis.transpose());

        if ((guese_T_o_b_.linear() * main_rot_axis).z() < 0) {
            // check polarity
            main_rot_axis = -main_rot_axis;
        }

        g_b_ = main_rot_axis;
        return g_b_;
    }

    Eigen::Vector2d solve_pitch_roll() {
        Eigen::Quaterniond R_w_b;
        R_w_b.setFromTwoVectors(g_b_, Eigen::Vector3d(0, 0, 1));
        Eigen::Vector3d ypr = Utility::R2ypr(R_w_b.toRotationMatrix()) * M_PI / 180;
        ROS_INFO("pitch: %f, roll: %f", ypr(1) * 180 / M_PI, ypr(2) * 180 / M_PI);
        ypr_o_b_.segment<2>(1) = ypr.segment<2>(1);
        return ypr.segment<2>(1);
    }

    double solve_yaw_xy_scale(double max_rotation = 10.0 / 180 * M_PI);

    Eigen::Affine3d T_o_b() {
        return Eigen::Translation3d(p_o_b) * Utility::ypr2R(ypr_o_b_ * 180 / M_PI);
    }

};


#include <ceres/ceres.h>

class ExCalibFactor : public ceres::SizedCostFunction<2, 2, 1, 1, 1> {
public:
    Eigen::Vector2d p_Oi_Oj_;
    Eigen::Rotation2Dd R_Oi_Oj_;
    Eigen::Vector2d p_Fi_Fj_;
    double radius_p_Fi_Fj_;
    double angle_p_Fi_Fj_;
public:
    ExCalibFactor() = delete;

    ExCalibFactor(Eigen::Vector2d p_Oi_Oj, double rot_Oi_Oj, Eigen::Vector2d p_Fi_Fj)
            : p_Oi_Oj_(p_Oi_Oj), R_Oi_Oj_(rot_Oi_Oj), p_Fi_Fj_(p_Fi_Fj)
            , radius_p_Fi_Fj_(p_Fi_Fj_.norm())
            , angle_p_Fi_Fj_(atan2(p_Fi_Fj_.y(), p_Fi_Fj_.x())) {
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

        Eigen::Vector2d p_O_F(parameters[0][0], parameters[0][1]);
        double angle_R_O_F = parameters[1][0];
        Eigen::Rotation2Dd R_O_F(angle_R_O_F);
        double sx = parameters[2][0];
        double sy = parameters[3][0];
        Eigen::Matrix2d scale; scale << sx, 0, 0, sy;

        if (residuals) {
            Eigen::Map<Eigen::Matrix<double, 2, 1>> residual(residuals);
            residual = p_O_F + R_O_F * p_Fi_Fj_ - R_Oi_Oj_ * p_O_F - scale * p_Oi_Oj_;
        }

        if (jacobians) {
            if (jacobians[0]) {
                // dres_dp
                Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> jacobian_p_O_F(jacobians[0]);
                jacobian_p_O_F = Eigen::Matrix2d::Identity() - R_Oi_Oj_.toRotationMatrix();
            }
            if (jacobians[1]) {
                // dres_dR
                Eigen::Map<Eigen::Matrix<double, 2, 1>> jacobian_R_O_F(jacobians[1]);
                jacobian_R_O_F << radius_p_Fi_Fj_ * cos(angle_R_O_F + angle_p_Fi_Fj_ + M_PI / 2),
                        radius_p_Fi_Fj_ * sin(angle_R_O_F + angle_p_Fi_Fj_ + M_PI / 2);
            }
            if (jacobians[2]) {
                // dres_dsx
                Eigen::Map<Eigen::Matrix<double, 2, 1>> jacobian_sx(jacobians[2]);
                Eigen::Matrix2d J; J << -1, 0, 0, 0;
                jacobian_sx = J * p_Oi_Oj_;
            }
            if (jacobians[3]) {
                // dres_dsy
                Eigen::Map<Eigen::Matrix<double, 2, 1>> jacobian_sy(jacobians[3]);
                Eigen::Matrix2d J; J << 0, 0, 0, -1;
                jacobian_sy = J * p_Oi_Oj_;
            }
        }

        return true;
    }
};

double WheelOdomVIOAlignment::solve_yaw_xy_scale(double max_rotation) {
    Eigen::Matrix3d R_f_b = Utility::ypr2R(Eigen::Vector3d(0, ypr_o_b_(1) / M_PI * 180, ypr_o_b_(2) / M_PI * 180));
    Eigen::Affine3d T_f_b = Eigen::Translation3d(0, 0, guese_T_o_b_.translation().z()) * R_f_b;

    // initial x = guese value
    Eigen::Matrix<double, 2, 1> p_o_f = guese_T_o_b_.translation().segment<2>(0);
    double angle_R_o_f = Utility::R2ypr(guese_T_o_b_.linear())[0] / 180 * M_PI;
    double sx = 1;
    double sy = 1;

    ceres::Problem problem;
    problem.AddParameterBlock(p_o_f.data(), 2);
    problem.AddParameterBlock(&angle_R_o_f, 1);
    problem.AddParameterBlock(&sx, 1);
    problem.AddParameterBlock(&sy, 1);

    // bound x,y to [-1, 1], bound yaw to [-pi, pi]
    problem.SetParameterLowerBound(p_o_f.data(), 0, p_o_f.x() - 1);
    problem.SetParameterUpperBound(p_o_f.data(), 0, p_o_f.x() + 1);
    problem.SetParameterLowerBound(p_o_f.data(), 1, p_o_f.y() - 1);
    problem.SetParameterUpperBound(p_o_f.data(), 1, p_o_f.y() + 1);
    problem.SetParameterLowerBound(&angle_R_o_f, 0, angle_R_o_f - M_PI);
    problem.SetParameterUpperBound(&angle_R_o_f, 0, angle_R_o_f + M_PI);
    problem.SetParameterLowerBound(&sx, 0, sx * 0.8);
    problem.SetParameterUpperBound(&sx, 0, sx * 1.2);
    problem.SetParameterLowerBound(&sy, 0, sy * 0.8);
    problem.SetParameterUpperBound(&sy, 0, sy * 1.2);

    auto *loss_function = new ceres::HuberLoss(0.01);

    // generate factors
    std::vector<int> segments;
    segments.push_back(0);
    for (int i = 1; i < ts_.size(); i++) {
        auto t_i = ts_[segments.back()];
        auto t_j = ts_[i];
        Eigen::Affine3d T_oi_oj = m_wo_[t_i.first].inverse() * m_wo_[t_j.first];
        Eigen::Vector2d p_oi_oj = T_oi_oj.translation().segment<2>(0);
        double yaw_oi_oj = Utility::R2ypr(T_oi_oj.linear())[0] / 180 * M_PI;
        if (p_oi_oj.norm() >= 0.1 || std::abs(yaw_oi_oj) >= 30.0 / 180 * M_PI) {
            segments.push_back(i);
        }
    }

    std::list<std::pair<ExCalibFactor*, ceres::ResidualBlockId>> residual_blocks;
    for (int i = 0; i + 1 < segments.size(); i++) {
        auto t_i = ts_[segments[i]];
        auto t_j = ts_[segments[i + 1]];
        Eigen::Affine3d T_bi_bj = m_vio_[t_i.second].inverse() * m_vio_[t_j.second];
        Eigen::Affine3d T_fi_fj = T_f_b * T_bi_bj * T_f_b.inverse();
        Eigen::Affine3d T_oi_oj = m_wo_[t_i.first].inverse() * m_wo_[t_j.first];

        Eigen::Vector2d p_fi_fj = T_fi_fj.translation().segment<2>(0);
        double yaw_fi_fj = Utility::R2ypr(T_fi_fj.linear())[0] / 180 * M_PI;

        Eigen::Vector2d p_oi_oj = T_oi_oj.translation().segment<2>(0);
        double yaw_oi_oj = Utility::R2ypr(T_oi_oj.linear())[0] / 180 * M_PI;

        auto *factor = new ExCalibFactor(p_oi_oj, yaw_oi_oj, p_fi_fj);
        auto block_id = problem.AddResidualBlock(factor, loss_function, p_o_f.data(), &angle_R_o_f, &sx, &sy);
        residual_blocks.emplace_back(factor, block_id);
    }

    // solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.num_threads = 4;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    ROS_INFO_STREAM(summary.BriefReport());
    ROS_INFO_STREAM("x,y: " << p_o_f.transpose());
    ROS_INFO_STREAM("yaw: " << angle_R_o_f / M_PI * 180);
    ROS_INFO_STREAM("sx,sy: " << sx << " " << sy);

    // reject outliers
    double *x[] = {p_o_f.data(), &angle_R_o_f, &sx, &sy};
    std::vector<typeof(residual_blocks.begin())> need_remove;
    for (auto it = residual_blocks.begin(); it != residual_blocks.end(); it++) {
        ExCalibFactor* factor = it->first;
        Eigen::Vector2d residual;
        factor->Evaluate(x, residual.data(), nullptr);
        printf("T_oi_oj: %8.3lf (%8.3lf) ", factor->p_Oi_Oj_.norm(), factor->R_Oi_Oj_.angle() * 180 / M_PI);
        printf("T_fi_fj: %8.3lf (%8.3lf) ", factor->p_Fi_Fj_.norm(), factor->R_Oi_Oj_.angle() * 180 / M_PI);
        printf("r: %8.3lf ", residual.norm());
        if (residual.norm() > (factor->p_Fi_Fj_.norm() + std::abs(factor->R_Oi_Oj_.angle()) * 0.3) * 0.1 && residual.norm() > 0.01) {
            need_remove.emplace_back(it);
            printf("remove");
        }
        printf("\n");
    }
    for (auto& it : need_remove) {
        problem.RemoveResidualBlock(it->second);
        residual_blocks.erase(it);
    }

    // solve again
    Solve(options, &problem, &summary);
    ROS_INFO_STREAM(summary.BriefReport());
    ROS_INFO_STREAM("x,y: " << p_o_f.transpose());
    ROS_INFO_STREAM("yaw: " << angle_R_o_f / M_PI * 180);
    ROS_INFO_STREAM("sx,sy: " << sx << " " << sy);

    p_o_b.segment<2>(0) = p_o_f;
    p_o_b[2] = guese_T_o_b_.translation().z();
    ypr_o_b_[0] = angle_R_o_f;
    scale_ << sx, sy;
}

#endif //VINS_ESTIMATOR_ODOM_VIO_EX_CALIB_H
