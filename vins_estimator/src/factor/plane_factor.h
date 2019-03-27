//
// Created by bxwllzz on 19-3-28.
//

#ifndef VINS_ESTIMATOR_PLANE_FACTOR_H
#define VINS_ESTIMATOR_PLANE_FACTOR_H


#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"
#include "../parameters.h"

#include <ceres/ceres.h>

// size of residuals = 1 (displacement[3])
// size of param[0]  = 7 (p_i[3], q_i[4])
class GlobalPlaneFactor : public ceres::SizedCostFunction<1, 7> {
public:
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        double& residual = *residuals;
        residual = Pi.z();

        double sqrt_info = 1 / 0.01;
        residual = sqrt_info * residual;

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 1, 7>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();
                jacobian_pose_i(2) = 1;
                jacobian_pose_i = sqrt_info * jacobian_pose_i;
            }
        }
        return true;
    }
};

#endif //VINS_ESTIMATOR_PLANE_FACTOR_H
