#pragma once
#include <map>
#include <vector>
#include <memory>
using std::map;
using std::vector;
using std::pair;

#include <Eigen/Core>
using namespace Eigen;

#include "../factor/integration_base.h"
#include "../factor/base_odom_integration.h"

class ImageFrame
{
    public:
        ImageFrame(){};
        ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& _points, double _t):t{_t},is_key_frame{false}
        {
            points = _points;
        };
        map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>> > > points;

        double t;
        Matrix3d R;
        Vector3d T;
        std::shared_ptr<IntegrationBase> pre_integration;
        std::shared_ptr<BaseOdometryIntegration3D> base_integration;
        bool is_key_frame;
};

std::vector<double> GetStillFrames(const map<double, ImageFrame> &all_image_frame, int protect_frames);

void base_imu_alignment(const vector<pair<std::shared_ptr<IntegrationBase>, std::shared_ptr<BaseOdometryIntegration3D>>> &pre_integrations,
                        const Matrix3d& R_imu_base, const Vector3d& t_imu_base,
                        VectorXd &x, Vector3d &g, double &s, double &avg_err_p, double &avg_err_v);

void base_imu_alignment_fixed_scale(const vector<pair<std::shared_ptr<IntegrationBase>, std::shared_ptr<BaseOdometryIntegration3D>>> &pre_integrations,
                                    const Matrix3d& R_imu_base, const Vector3d& t_imu_base,
                                    VectorXd &x, Vector3d &g, double &avg_err_p, double &avg_err_v);

void base_imu_alignment_fixed_scale_g(const vector<pair<std::shared_ptr<IntegrationBase>, std::shared_ptr<BaseOdometryIntegration3D>>> &pre_integrations,
                                    const Matrix3d& R_imu_base, const Vector3d& t_imu_base,
                                    VectorXd &x, const Vector3d &g, vector<Vector3d>& err_p, vector<Vector3d>& err_v);

bool WheelOdomIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x);
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x);