//
// Created by bxwllzz on 19-3-25.
//

#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <nav_msgs/Path.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include "utility/utility.h"
#include "odom-vio_ex_calib.h"

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: %s bag_file_path topic_visual_odom_path topic_wheel_odom_path\nType of topic: nav_msgs/Path\n",
               argv[0]);
        return -1;
    }

    string bag_file_path = argv[1];
    string topic_visual_odom_path = argv[2];
    string topic_wheel_odom_path = argv[3];

    cout << "Openning rosbag " << bag_file_path << "...\t" << flush;
    rosbag::Bag bag(bag_file_path);
    cout << "OK" << endl;

    cout << "Load visual odometry path " << topic_visual_odom_path << "...\t" << flush;
    nav_msgs::Path::ConstPtr path_vo;
    for (rosbag::MessageInstance const m : rosbag::View(bag, rosbag::TopicQuery({topic_visual_odom_path}))) {
        nav_msgs::Path::ConstPtr path = m.instantiate<nav_msgs::Path>();
        if (path != nullptr) {
            if (path_vo == nullptr || path->poses.size() > path_vo->poses.size()) {
                path_vo = path;
            }
        }
    }
    if (path_vo == nullptr) {
        cout << "Failed" << endl;
        return -2;
    }
    cout << path_vo->poses.size() << " poses" << endl;

    cout << "Load wheel odometry path " << topic_wheel_odom_path << "...\t" << flush;
    nav_msgs::Path::ConstPtr path_wo;
    for (rosbag::MessageInstance const m : rosbag::View(bag, rosbag::TopicQuery({topic_wheel_odom_path}))) {
        nav_msgs::Path::ConstPtr path = m.instantiate<nav_msgs::Path>();
        if (path != nullptr) {
            if (path_wo == nullptr || path->poses.size() > path_wo->poses.size()) {
                path_wo = path;
            }
        }
    }
    if (path_vo == nullptr) {
        cout << "Failed" << endl;
        return -2;
    }
    cout << path_wo->poses.size() << " poses" << endl;

    bag.close();

    Eigen::Affine3d guest_T_wheelodom_visualodom =
            Eigen::Translation3d(0, 0, 0.575) * Utility::ypr2R(Vector3d{0, 0, -90});

    cout << "Initial T^wheelodom_visualodom: " << endl;
    cout << "    translation: " << guest_T_wheelodom_visualodom.translation().transpose() << endl;
    cout << "    YPR:         " << Utility::R2ypr(guest_T_wheelodom_visualodom.linear()).transpose() << endl;
    WheelOdomVIOAlignment alignment(guest_T_wheelodom_visualodom);

    alignment.process_msg_path_vio(*path_vo);
    alignment.process_msg_path_wo(*path_wo);

    alignment.vio_main_rot_axis();
    alignment.solve_pitch_roll();
    alignment.solve_yaw_xy_scale();

    Matrix2d scale; scale << alignment.scale_.x(), 0, 0, alignment.scale_.y();
    cout << "Calibrated wheelodom scale (V_true=scale*V_measure):" << endl;
    cout << scale << endl;
    Affine3d T_wheelodom_visualodom = alignment.T_o_b();
    cout << "Calibrated T^wheelodom_visualodom:" << endl;
    cout << "    translation: " << T_wheelodom_visualodom.translation().transpose() << endl;
    cout << "    YPR:         " << Utility::R2ypr(T_wheelodom_visualodom.linear()).transpose() << endl;
    cout << T_wheelodom_visualodom.matrix() << endl;
    cout << "Calibrated T^visualodom_wheelodom:" << endl;
    cout << T_wheelodom_visualodom.inverse().matrix() << endl;
    return 0;
}