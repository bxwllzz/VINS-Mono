//
// Created by bxwllzz on 19-4-6.
//

#ifndef VINS_ESTIMATOR_IMUUTILITY_H
#define VINS_ESTIMATOR_IMUUTILITY_H

#include <utility>
#include <deque>
#include <Eigen/Core>
#include <sensor_msgs/Imu.h>

class ImuUtility {
public:
    static std::pair<Eigen::Vector3d, Eigen::Vector3d> msg2data(const sensor_msgs::ImuConstPtr &msg);

    static sensor_msgs::ImuPtr data2msg(const ros::Time &t,
                                        std::pair<Eigen::Vector3d, Eigen::Vector3d> data,
                                        const sensor_msgs::ImuConstPtr &template_msg = {});

    // 获取任意时间的IMUC测量
    static std::pair<Eigen::Vector3d, Eigen::Vector3d>
    interplote(std::deque<sensor_msgs::ImuConstPtr> msgs, const ros::Time &t);

    // 求一段时间内的IMU平均测量值
    static std::pair<Eigen::Vector3d, Eigen::Vector3d> average(std::deque<sensor_msgs::ImuConstPtr> msgs,
                                                               const ros::Time &begin_t, const ros::Time &end_t);

    static void midpoint_integration(double dt, const Eigen::Vector3d &g,
                                     const Eigen::Vector3d &acc_0, const Eigen::Vector3d &gyr_0,
                                     const Eigen::Vector3d &acc_1, const Eigen::Vector3d &gyr_1,
                                     Eigen::Vector3d &P, Eigen::Quaterniond &Q, Eigen::Vector3d &V);
};

#endif //VINS_ESTIMATOR_IMUUTILITY_H
