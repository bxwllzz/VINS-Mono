//
// Created by bxwllzz on 19-4-6.
//

#include <Eigen/Dense>
#include "utility.h"
#include "ImuUtility.h"

using namespace std;
using namespace Eigen;

pair<Vector3d, Vector3d> ImuUtility::msg2data(const sensor_msgs::ImuConstPtr &msg) {
    pair<Vector3d, Vector3d> res;
    res.first.x() = msg->linear_acceleration.x;
    res.first.y() = msg->linear_acceleration.y;
    res.first.z() = msg->linear_acceleration.z;
    res.second.x() = msg->angular_velocity.x;
    res.second.y() = msg->angular_velocity.y;
    res.second.z() = msg->angular_velocity.z;
    return res;
}

sensor_msgs::ImuPtr
ImuUtility::data2msg(const ros::Time &t, pair<Vector3d, Vector3d> data, const sensor_msgs::ImuConstPtr &template_msg) {
    sensor_msgs::ImuPtr msg;
    if (template_msg) {
        msg.reset(new sensor_msgs::Imu(*template_msg));
    } else {
        msg.reset(new sensor_msgs::Imu());
    }
    msg->header.stamp = t;
    msg->linear_acceleration.x = data.first.x();
    msg->linear_acceleration.y = data.first.y();
    msg->linear_acceleration.z = data.first.z();
    msg->angular_velocity.x = data.second.x();
    msg->angular_velocity.y = data.second.y();
    msg->angular_velocity.z = data.second.z();
    return msg;
}

// 获取任意时间的IMUC测量
pair<Vector3d, Vector3d> ImuUtility::interplote(deque<sensor_msgs::ImuConstPtr> msgs, const ros::Time &t) {
    auto it = find_if(msgs.begin(), msgs.end(), [&](const sensor_msgs::ImuConstPtr &msg) -> bool {
        return msg->header.stamp >= t;
    });
    if (it == msgs.end()) {
        throw std::range_error("ImuUtility::interplote(): time is too new");
    }
    pair<Vector3d, Vector3d> res;
    if ((*it)->header.stamp == t) {
        res = msg2data(*it);
    } else {
        if (it == msgs.begin()) {
            throw std::range_error("ImuUtility::interplote(): time is too old");
        }
        const auto &imu_i = **(it - 1);
        const auto &imu_j = **it;
        double dt_1 = (t - imu_i.header.stamp).toSec();
        double dt_2 = (imu_j.header.stamp - t).toSec();
        double w1 = dt_2 / (dt_1 + dt_2);
        double w2 = dt_1 / (dt_1 + dt_2);
        res.first.x() = imu_i.linear_acceleration.x * w1 + imu_j.linear_acceleration.x * w2;
        res.first.y() = imu_i.linear_acceleration.y * w1 + imu_j.linear_acceleration.y * w2;
        res.first.z() = imu_i.linear_acceleration.z * w1 + imu_j.linear_acceleration.z * w2;
        res.second.x() = imu_i.angular_velocity.x * w1 + imu_j.angular_velocity.x * w2;
        res.second.y() = imu_i.angular_velocity.y * w1 + imu_j.angular_velocity.y * w2;
        res.second.z() = imu_i.angular_velocity.z * w1 + imu_j.angular_velocity.z * w2;
    }

    return res;
}

// 求一段时间内的IMU平均测量值
pair<Vector3d, Vector3d>
ImuUtility::average(deque<sensor_msgs::ImuConstPtr> msgs, const ros::Time &begin_t, const ros::Time &end_t) {
    pair<Vector3d, Vector3d> imu_begin = interplote(msgs, begin_t);
    pair<Vector3d, Vector3d> imu_end = interplote(msgs, end_t);

    auto it = find_if(msgs.begin(), msgs.end(), [&](const sensor_msgs::ImuConstPtr &msg) -> bool {
        return msg->header.stamp > begin_t;
    });

    pair<Vector3d, Matrix3d> res;
    res.second.setIdentity();
    pair<Vector3d, Vector3d> prev_data = imu_begin;
    ros::Time prev_t = begin_t;
    while ((*it)->header.stamp < end_t) {
        pair<Vector3d, Vector3d> data = msg2data(*it);
        res.first += (data.first + prev_data.first) * 0.5 * ((*it)->header.stamp - prev_t).toSec();
        Vector3d gyr = (data.second + prev_data.second) * 0.5 * ((*it)->header.stamp - prev_t).toSec();
        res.second = res.second * Eigen::AngleAxisd(gyr.norm(), gyr.normalized());

        prev_data = data;
        prev_t = (*it)->header.stamp;
        it++;
    }

    res.first += (imu_end.first + prev_data.first) * 0.5 * (end_t - prev_t).toSec();
    Vector3d gyr = (imu_end.second + prev_data.second) * 0.5 * (end_t - prev_t).toSec();
    res.second = res.second * Eigen::AngleAxisd(gyr.norm(), gyr.normalized());

    res.first /= (end_t - begin_t).toSec();
    Eigen::AngleAxisd rot(res.second);
    rot.angle() /= (end_t - begin_t).toSec();
    return {res.first, rot.axis() * rot.angle()};
}

void ImuUtility::midpoint_integration(double dt, const Vector3d &g,
                                      const Vector3d &acc_0, const Vector3d &gyr_0,
                                      const Vector3d &acc_1, const Vector3d &gyr_1,
                                      Vector3d &P, Quaterniond &Q, Vector3d &V) {
    Vector3d un_acc_0 = Q * acc_0 - g;

    Vector3d un_gyr = 0.5 * (gyr_0 + gyr_1);
    Q = Q * Utility::deltaQ(un_gyr * dt);

    Vector3d un_acc_1 = Q * acc_1 - g;

    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    P = P + dt * V + 0.5 * dt * dt * un_acc;
    V = V + dt * un_acc;
}