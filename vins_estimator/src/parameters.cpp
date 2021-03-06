#include <fstream>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string IMU_TOPIC;
double ROW, COL;
double TD, TR;

std::string ODOM_TOPIC;
int ESTIMATE_EXTRINSIC_ODOM;
Eigen::Matrix3d RIO;
Eigen::Vector3d TIO;
int ESTIMATE_TD_ODOM;
double TD_ODOM;
Eigen::Matrix3d WHEEL_SCALE;
double WHEEL_MIN_N;
double WHEEL_N;
int USE_ODOM;
int INIT_USE_ODOM;

int USE_PLANE_FACTOR;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file");
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["imu_topic"] >> IMU_TOPIC;

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    std::string OUTPUT_PATH;
    fsSettings["output_path"] >> OUTPUT_PATH;
    VINS_RESULT_PATH = OUTPUT_PATH + "/vins_result_no_loop.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    ACC_N = fsSettings["acc_n"];
    ACC_W = fsSettings["acc_w"];
    GYR_N = fsSettings["gyr_n"];
    GYR_W = fsSettings["gyr_w"];
    G.z() = fsSettings["g_norm"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %f COL: %f ", ROW, COL);

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";

    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_R, cv_T;
        fsSettings["extrinsicRotation"] >> cv_R;
        fsSettings["extrinsicTranslation"] >> cv_T;
        Eigen::Matrix3d eigen_R;
        Eigen::Vector3d eigen_T;
        cv::cv2eigen(cv_R, eigen_R);
        cv::cv2eigen(cv_T, eigen_T);
        Eigen::Quaterniond Q(eigen_R);
        eigen_R = Q.normalized();
        RIC.push_back(eigen_R);
        TIC.push_back(eigen_T);
        ROS_INFO_STREAM("Extrinsic_R : " << std::endl << RIC[0]);
        ROS_INFO_STREAM("Extrinsic_T : " << std::endl << TIC[0].transpose());
        
    } 

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROLLING_SHUTTER = fsSettings["rolling_shutter"];
    if (ROLLING_SHUTTER)
    {
        TR = fsSettings["rolling_shutter_tr"];
        ROS_INFO_STREAM("rolling shutter camera, read out time per line: " << TR);
    }
    else
    {
        TR = 0;
    }

    fsSettings["odom_topic"] >> ODOM_TOPIC;

    fsSettings["extimate_extrinsic_odom"] >> ESTIMATE_EXTRINSIC_ODOM;
    if (ESTIMATE_EXTRINSIC_ODOM == 2) {
        ROS_WARN("have no prior about extrinsic param between imu and base, calibrate extrinsic param");
        RIO = Eigen::Matrix3d::Identity();
        TIO = Eigen::Vector3d::Zero();
    } else {
        if (ESTIMATE_EXTRINSIC == 1) {
            ROS_WARN(" Optimize extrinsic param between imu and base around initial guess!");
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param between imu and base");

        cv::Mat cv_R, cv_T;
        fsSettings["extrinsicOdomRotation"] >> cv_R;
        fsSettings["extrinsicOdomTranslation"] >> cv_T;
        Eigen::Matrix3d eigen_R;
        Eigen::Vector3d eigen_T;
        cv::cv2eigen(cv_R, eigen_R);
        cv::cv2eigen(cv_T, eigen_T);
        Eigen::Quaterniond Q(eigen_R);
        eigen_R = Q.normalized();
        RIO = eigen_R;
        TIO = eigen_T;
        ROS_INFO_STREAM("Extrinsic_Odom_R : " << std::endl << RIO);
        ROS_INFO_STREAM("Extrinsic_Odom_T : " << std::endl << TIO.transpose());
    }

    TD_ODOM = fsSettings["odom_td"];
    ESTIMATE_TD_ODOM = fsSettings["estimate_odom_td"];
    if (ESTIMATE_TD_ODOM)
        ROS_INFO_STREAM("Unsynchronized odom sensors, online estimate time offset, initial td: " << TD_ODOM);
    else
        ROS_INFO_STREAM("Synchronized odom sensors, fix time offset: " << TD_ODOM);

    cv::Mat cv_WHEEL_SCALE;
    fsSettings["wheel_odom_scale"] >> cv_WHEEL_SCALE;
    cv::cv2eigen(cv_WHEEL_SCALE, WHEEL_SCALE);
    fsSettings["wheel_min_n"] >> WHEEL_MIN_N;
    fsSettings["wheel_n"] >> WHEEL_N;
    fsSettings["use_odom"] >> USE_ODOM;
    switch (USE_ODOM) {
    case 1:
        ROS_INFO_STREAM("Use only inter-frame odometry factor");
        break;
    case 2:
        ROS_INFO_STREAM("Use only begin-end-frame odometry factor");
        break;
    case 3:
        ROS_INFO_STREAM("Use both inter-frame odom factor and begin-end-frame odom factor");
        break;
    default:
        ROS_INFO_STREAM("Not use wheel odometry fator");
        USE_ODOM = 0;
        break;
    }
    fsSettings["init_use_odom"] >> INIT_USE_ODOM;

    fsSettings["use_plane_factor"] >> USE_PLANE_FACTOR;

    fsSettings.release();
}
