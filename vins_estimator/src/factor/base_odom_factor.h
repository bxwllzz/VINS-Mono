//
// Created by bxwllzz on 18-12-17.
//

#ifndef VINS_ESTIMATOR_BASE_ODOM_FACTOR_H
#define VINS_ESTIMATOR_BASE_ODOM_FACTOR_H

#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"
#include "../parameters.h"
#include "base_odom_integration.h"

#include <ceres/ceres.h>

// size of residuals = 3 (displacement[3])
// size of param[0]  = 7 (p_i[3], q_i[4])
// size of param[1]  = 9 (v_i[3], ba_i[3], bg_i[3])
// size of param[2]  = 7 (p_j[3], q_j[4])
class BaseOdomFactor : public ceres::SizedCostFunction<3, 7, 9, 7>
{
public:
    BaseOdomFactor() = delete;
    BaseOdomFactor(std::shared_ptr<BaseOdometryIntegration3D> _pre_integration): pre_integration(_pre_integration)
    {
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {

        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
        Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

        Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

        Eigen::Map<Eigen::Matrix<double, 3, 1>> residual(residuals);
        residual = pre_integration->evaluate(Pi, Qi, Bgi,
                                             Pj, Qj);

        Eigen::Matrix<double, 3, 3> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 3, 3>>(pre_integration->covariance.block<3, 3>(0, 0).inverse()).matrixL().transpose();
        //sqrt_info.setIdentity();
        residual = sqrt_info * residual;

        if (jacobians)
        {
            double sum_dt = pre_integration->sum_dt;
            Eigen::Matrix3d dp_dbg = pre_integration->jacobian.block<3, 3>(0, 6);

            if (pre_integration->jacobian.maxCoeff() > 1e8 || pre_integration->jacobian.minCoeff() < -1e8)
            {
                ROS_WARN("numerical unstable in wheel preintegration");
                //std::cout << pre_integration->jacobian << std::endl;
///                ROS_BREAK();
            }

            if (jacobians[0])
            {
                // delta residual w.r.t. Pi and Qi
                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();
                // dp_dpi
                jacobian_pose_i.block<3, 3>(0, O_P) = -RIO.transpose() * Qi.inverse().toRotationMatrix();
                // dp_dqi
                jacobian_pose_i.block<3, 3>(0, O_R) = -RIO.transpose() * Utility::skewSymmetric(Qi.inverse() * (Pj - Pi + Qj * TIO));

                jacobian_pose_i = sqrt_info * jacobian_pose_i;

                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in preintegration");
                    //std::cout << sqrt_info << std::endl;
                    //ROS_BREAK();
                }
            }
            if (jacobians[1])
            {
                // delta w.r.t. Vi, Bai and Bgi
                Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
                jacobian_speedbias_i.setZero();
                // dp_dbgi
                jacobian_speedbias_i.block<3, 3>(0, O_BG - O_V) = -dp_dbg;

                jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;

                //ROS_ASSERT(fabs(jacobian_speedbias_i.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_speedbias_i.minCoeff()) < 1e8);
            }
            if (jacobians[2])
            {
                // delta w.r.t. Pj, Qj
                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);
                jacobian_pose_j.setZero();
                // dp_dpj
                jacobian_pose_j.block<3, 3>(0, O_P) = RIO.transpose() * Qi.inverse().toRotationMatrix();
                // dp_dqj
                jacobian_pose_j.block<3, 3>(0, O_R) = -RIO.transpose() * Qi.inverse() * Qj * Utility::skewSymmetric(TIO);

                jacobian_pose_j = sqrt_info * jacobian_pose_j;
                //ROS_ASSERT(fabs(jacobian_pose_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_pose_j.minCoeff()) < 1e8);
            }
        }

        return true;
    }

    //bool Evaluate_Direct(double const *const *parameters, Eigen::Matrix<double, 15, 1> &residuals, Eigen::Matrix<double, 15, 30> &jacobians);

    //void checkCorrection();
    //void checkTransition();
    //void checkJacobian(double **parameters);
    std::shared_ptr<BaseOdometryIntegration3D> pre_integration;

};

#endif //VINS_ESTIMATOR_BASE_ODOM_FACTOR_H
