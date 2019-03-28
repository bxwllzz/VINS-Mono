#pragma once

#include <vector>
#include <map>
#include <utility>
#include <Eigen/Dense>
#include <std_msgs/Header.h>

#include "parameters.h"
#include "feature_manager.h"
#include "factor/base_odom_integration.h"
#include "factor/marginalization_factor.h"
#include "initial/solve_5pts.h"
#include "initial/initial_ex_rotation.h"
#include "initial/initial_alignment.h"

using namespace Eigen;
using std::vector;
using std::map;
using std::pair;

class Estimator
{
public:

    Estimator();

    void setParameter();

    // interface
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processOdometry(double dt, const pair<Vector2d, double>& velocity, double constraint_error_vel, Vector3d imu_linear_acceleration, Vector3d imu_angular_velocity);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header);
    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);

    // internal
    void clearState();
    bool initialStructure();
    bool wheelOdomInitialAlign();
    bool visualInitialAlign();
    bool baseOdomAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void solveOdometry();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();


    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    std::vector<std::pair<string, double>> status;
    void status_log_p(const string& name, const Affine3d& value) {
        status_log_p(name, value.translation());
    }
    void status_log_p(const string& name, const Vector3d& value) {
        status.emplace_back(name + "_x", value.x());
        status.emplace_back(name + "_y", value.y());
        status.emplace_back(name + "_z", value.z());
    }
    void status_log_ypr(const string& name, const Affine3d& value) {
        status_log_ypr(name, Utility::R2ypr(value.linear()));
    }
    void status_log_ypr(const string& name, const Quaterniond& value) {
        status_log_ypr(name, value.toRotationMatrix());
    }
    void status_log_ypr(const string& name, const Matrix3d& value) {
        status_log_ypr(name, Utility::R2ypr(value));
    }
    void status_log_ypr(const string& name, const Vector3d& value) {
        status.emplace_back(name + "_yaw", value.x());
        status.emplace_back(name + "_pitch", value.y());
        status.emplace_back(name + "_roll", value.z());
    }
    std::vector<string> history_index;
    std::map<string, std::vector<double>> history_status;
    void log_status();
    void save_history(string path);

    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g;                 // gravity in first cam frame (when INITIAL) OR in world frame (when NON_LINEAR)
    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;

    Matrix3d ric[NUM_OF_CAM];   // rotation of cam frame w.r.t. imu frame, R^i_c
    Vector3d tic[NUM_OF_CAM];   // translation of cam frame w.r.t. imu frame, t^i_c

    // TOTAL WINDOW SIZE = WINDOW_SIZE + 1, last frame is the newest frame
    Vector3d Ps[(WINDOW_SIZE + 1)];     // Positions of body frame w.r.t. world frame in sliding window, p^w_b
    Vector3d Vs[(WINDOW_SIZE + 1)];     // Velocities of body frame w.r.t. world framein sliding window, v^w_b
    Matrix3d Rs[(WINDOW_SIZE + 1)];     // Rotations of body frame w.r.t. world frame in sliding window, R^w_b
    Vector3d Bas[(WINDOW_SIZE + 1)];    // Bias of Acceleration in sliding window, un_acc = raw_acc - Ba
    Vector3d Bgs[(WINDOW_SIZE + 1)];    // Bias of Gyro in sliding window, un_gyro = raw_gyro - Bg
    double td;                          // time delay between cam and imu, t_cam + td = t_imu

    Matrix3d rio;
    Vector3d tio;
    double td_bo;
    std::shared_ptr<BaseOdometryIntegration3D> base_integrations[WINDOW_SIZE + 1];

    // debug
//    Vector3d imu_predict_P = Vector3d::Zero();
//    Matrix3d imu_predict_R = Matrix3d::Identity();
//    Vector3d imu_predict_V = Vector3d::Zero();
//    Vector3d optimized_P = Vector3d::Zero();
//    Matrix3d optimized_R = Matrix3d::Identity();
//    Vector3d optimized_V = Vector3d::Zero();
//    Vector3d wheel_predict_P = Vector3d::Zero();
//    Matrix3d wheel_predict_R = Matrix3d::Zero();
//    double wheel_predict_dt = 0;
//    Vector3d wheel_imu_P = Vector3d::Zero();
//    Vector3d wheel_imu_V = Vector3d::Zero();
//    Vector3d wheel_imu_predict_P = Vector3d::Zero();
//    Vector3d wheel_imu_predict_V = Vector3d::Zero();

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    std_msgs::Header Headers[(WINDOW_SIZE + 1)];

    std::map<std::string, double> window_info;
    std::shared_ptr<IntegrationBase> pre_integrations[(WINDOW_SIZE + 1)];
    Vector3d acc_0, gyr_0;  // lastest acc and gyro

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;

    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info = NULL;
    vector<double *> last_marginalization_parameter_blocks;

    /* Initialization */
    // all frame, including non-keyframe (frame older than oldest keyframe will be removed)
    map<double, ImageFrame> all_image_frame;
    // imu and base odom integration for current frame
    std::shared_ptr<IntegrationBase> tmp_pre_integration;
    std::shared_ptr<BaseOdometryIntegration3D> tmp_base_integration;

    // check wheel slip
    std::shared_ptr<BaseOdometryIntegration3D> wheel_imu_predict;
    std::list<pair<double, double>> wheel_slip_periods;

    BaseOdometryIntegration3D base_integration_before_init;

    // wheel only odometry
    BaseOdometryIntegration3D wheel_only_odom;
    BaseOdometryIntegration3D wheel_imu_odom;
    BaseOdometryIntegration3D wheel_imu_odom3D;
//    WheelOdometryNoiseAnalyser wheel_odom_niose_analyser = { 10 };

    // wheel imu fusion window
//    std::deque<std::shared_ptr<IntegrationBase>> wi_imu_integrations = {std::shared_ptr<IntegrationBase>()};
//    std::deque<std::shared_ptr<BaseOdometryIntegration3D>> wi_base_integrations = {std::shared_ptr<BaseOdometryIntegration3D>()};

    //relocalization variable
    bool relocalization_info;
    double relo_frame_stamp;
    double relo_frame_index;
    int relo_frame_local_index;
    vector<Vector3d> match_points;
    double relo_Pose[SIZE_POSE];
    Matrix3d drift_correct_r;
    Vector3d drift_correct_t;
    Vector3d prev_relo_t;
    Matrix3d prev_relo_r;
    Vector3d relo_relative_t;
    Quaterniond relo_relative_q;
    double relo_relative_yaw;
};
