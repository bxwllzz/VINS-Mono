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
        std::shared_ptr<BaseOdometryIntegration> base_integration;
        bool is_key_frame;
};

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x);