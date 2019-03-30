//
// Created by bxwllzz on 19-3-27.
//


#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s bag_in\n",
               argv[0]);
        return -1;
    }

    string bag_in_path = argv[1];

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
    map<string, ros::Time> topic_prev_t;
    map<string, ros::Duration> topic_max_interval;
    for (auto& topic : info) {
        topic_types[topic->topic] = topic->datatype;
        topic_prev_t[topic->topic] = {};
        topic_max_interval[topic->topic] = {};
    }

    for (const auto& m : view) {
        std_msgs::Header header;
        auto& topic = m.getTopic();
        if (m.getDataType() == "nav_msgs/Odometry") {
            header = m.instantiate<nav_msgs::Odometry>()->header;
        } else if (m.getDataType() == "sensor_msgs/Image") {
            header = m.instantiate<sensor_msgs::Image>()->header;
        } else if (m.getDataType() == "sensor_msgs/Imu") {
            header = m.instantiate<sensor_msgs::Imu>()->header;
        }
        if (header.stamp - topic_prev_t[topic] > topic_max_interval[topic]
            && !topic_prev_t[topic].isZero()) {
            topic_max_interval[topic] = header.stamp - topic_prev_t[topic];
            cout << topic << " " << topic_max_interval[topic] << endl;
        }
        topic_prev_t[topic] = header.stamp;
        cout << (m.getTime() - view.getBeginTime()).toSec() / (view.getEndTime() - view.getBeginTime()).toSec() * 100 << "%\r" << flush;
    }

    for (auto& kv : topic_types) {
        cout << "    " << kv.first << " (" << kv.second << "): max_interval=" << topic_max_interval[kv.first].toSec() << endl;
    }

    bag_in.close();
    return 0;
}