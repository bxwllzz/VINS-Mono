//
// Created by bxwllzz on 19-3-27.
//


#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/imgcodecs.hpp>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Odometry.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <cv_bridge/cv_bridge.h>

using namespace std;
using namespace Eigen;

static bool str_endswith(const string &full_str, const string &end_str) {
    if (full_str.size() < end_str.size())
        return false;
    return full_str.compare(full_str.size() - end_str.size(), end_str.size(), end_str) == 0;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        printf("Usage: %s bag_in bag_out blind|kidnap|decompress_image|compress_image time_begin time_end\n",
               argv[0]);
        return -1;
    }

    string bag_in_path = argv[1];
    string bag_out_path = argv[2];
    string op_type = argv[3];
    double time_begin = stod(argv[4]);
    double time_end = stod(argv[5]);

    cout << "Operation      : " << op_type << endl;
    cout << "Operation begin: " << time_begin << " sec" << endl;
    cout << "Operation end  : " << time_end << " sec" << endl;

    cout << "Open input bag : " << bag_in_path << "...\t" << flush;
    rosbag::Bag bag_in(bag_in_path);
    rosbag::View view(bag_in);
    cout << "OK" << endl;
    cout << "Bag begin time : " << view.getBeginTime() << endl;
    cout << "Bag end time   : " << view.getEndTime() << endl;
    cout << "Duration       : " << (view.getEndTime() - view.getBeginTime()).toSec() << " sec" << endl;
    cout << "Topics:" << endl;
    auto info = view.getConnections();
    map<string, string> topic_types;
    for (auto &topic : info) {
        topic_types[topic->topic] = topic->datatype;
    }
    for (auto &kv : topic_types) {
        cout << "    " << kv.first << ": " << kv.second << endl;
    }

    cout << "Openning output bag " << bag_out_path << "...\t" << flush;
    rosbag::Bag bag_out(bag_out_path, rosbag::BagMode::Write);
    cout << "OK" << endl;
    if (op_type == "blind") {
        for (const auto &m : view) {
            double t = (m.getTime() - view.getBeginTime()).toSec();
            if (t >= time_begin && t <= time_end && m.isType<sensor_msgs::Image>()) {
                sensor_msgs::Image msg = *m.instantiate<sensor_msgs::Image>();
                for (auto &c : msg.data)
                    c = 0;
                bag_out.write(m.getTopic(), m.getTime(), msg, m.getConnectionHeader());
//                cout << "Set " << t << " " << m.getTopic() << " (" << msg.width << "x" << msg.height << ")" << " to blank" << endl;
            } else {
                bag_out.write(m.getTopic(), m.getTime(), m, m.getConnectionHeader());
            }
            cout << t / (view.getEndTime() - view.getBeginTime()).toSec() * 100 << "        %\r" << flush;
        }
    } else if (op_type == "kidnap") {
        Affine3d T_before;
        double err_before = 0;
        bool reach_after = false;
        Affine3d T_after;
        double err_after = 0;
        for (const auto &m : view) {
            double t = (m.getTime() - view.getBeginTime()).toSec();
            if (m.isType<nav_msgs::Odometry>()) {
                nav_msgs::Odometry msg = *m.instantiate<nav_msgs::Odometry>();
                Affine3d T = Translation3d(msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z)
                             * Quaterniond(msg.pose.pose.orientation.w, msg.pose.pose.orientation.x,
                                           msg.pose.pose.orientation.y, msg.pose.pose.orientation.z);
                double constraint_error = msg.pose.covariance[1];

                if (t < time_begin) {
                    T_before = T;
                    err_before = constraint_error;
                } else if (t <= time_end) {
                    T = T_before;
                    constraint_error = err_before;
                    msg.pose.pose.position.x = T.translation().x();
                    msg.pose.pose.position.y = T.translation().y();
                    msg.pose.pose.position.z = T.translation().z();
                    Quaterniond q(T.linear());
                    msg.pose.pose.orientation.w = q.w();
                    msg.pose.pose.orientation.x = q.x();
                    msg.pose.pose.orientation.y = q.y();
                    msg.pose.pose.orientation.z = q.z();
                    msg.pose.covariance[1] = constraint_error;
                    msg.twist.twist.linear.x = 0;
                    msg.twist.twist.linear.y = 0;
                    msg.twist.twist.linear.z = 0;
                    msg.twist.twist.angular.x = 0;
                    msg.twist.twist.angular.y = 0;
                    msg.twist.twist.angular.z = 0;
                } else {
                    if (!reach_after) {
                        reach_after = true;
                        T_after = T;
                        err_after = constraint_error;
                    }
                    T = T_before * (T_after.inverse() * T);
                    constraint_error = err_before + (constraint_error - err_after);
                    msg.pose.pose.position.x = T.translation().x();
                    msg.pose.pose.position.y = T.translation().y();
                    msg.pose.pose.position.z = T.translation().z();
                    Quaterniond q(T.linear());
                    msg.pose.pose.orientation.w = q.w();
                    msg.pose.pose.orientation.x = q.x();
                    msg.pose.pose.orientation.y = q.y();
                    msg.pose.pose.orientation.z = q.z();
                    msg.pose.covariance[1] = constraint_error;
                }

                bag_out.write(m.getTopic(), m.getTime(), msg, m.getConnectionHeader());
            } else {
                bag_out.write(m.getTopic(), m.getTime(), m, m.getConnectionHeader());
            }
            cout << t / (view.getEndTime() - view.getBeginTime()).toSec() * 100 << "%        \r" << flush;
        }
    } else if (op_type == "compress_image") {
        for (const auto &m : view) {
            double t = (m.getTime() - view.getBeginTime()).toSec();
            if (m.isType<sensor_msgs::Image>()) {
                sensor_msgs::Image msg = *m.instantiate<sensor_msgs::Image>();
                if (msg.encoding == sensor_msgs::image_encodings::TYPE_8UC1) {
                    msg.encoding = sensor_msgs::image_encodings::MONO8;
                }
                cv_bridge::CvImagePtr cv_img = cv_bridge::toCvCopy(msg);
                sensor_msgs::CompressedImagePtr compress_msg = cv_img->toCompressedImageMsg(cv_bridge::JPG);
                if (str_endswith(m.getTopic(), "/compressed")) {
                    bag_out.write(m.getTopic(), m.getTime(), compress_msg);
                } else {
                    bag_out.write(m.getTopic() + "/compressed", m.getTime(), compress_msg);
                }
            } else {
                bag_out.write(m.getTopic(), m.getTime(), m, m.getConnectionHeader());
            }
            cout << t / (view.getEndTime() - view.getBeginTime()).toSec() * 100 << "%        \r" << flush;
        }
    } else if (op_type == "decompress_image") {
        for (const auto &m : view) {
            double t = (m.getTime() - view.getBeginTime()).toSec();
            if (m.isType<sensor_msgs::CompressedImage>()) {
                sensor_msgs::CompressedImageConstPtr compress_msg = m.instantiate<sensor_msgs::CompressedImage>();
                auto cv_img = cv_bridge::toCvCopy(compress_msg, compress_msg->format);
                auto msg = cv_img->toImageMsg();
                if (str_endswith(m.getTopic(), "/compressed")) {
                    bag_out.write(m.getTopic().substr(0, m.getTopic().size() - strlen("/compressed")), m.getTime(), msg);
                } else {
                    bag_out.write(m.getTopic(), m.getTime(), msg);
                }
            } else {
                bag_out.write(m.getTopic(), m.getTime(), m, m.getConnectionHeader());
            }
            cout << t / (view.getEndTime() - view.getBeginTime()).toSec() * 100 << "%        \r" << flush;
        }
    } else {
        cout << "Unkonwn operation: " << op_type << endl;
    }
    cout << endl;

    bag_in.close();
    bag_out.close();
    return 0;
}