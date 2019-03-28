#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"
#include "utility/tic_toc.h"

struct ImuMeasurement {
    std_msgs::Header header;
    double dt;
    Vector3d linear_acceleration;
    Vector3d angular_velocity;

    ImuMeasurement(const ros::Time& t, const ros::Duration& dt_, pair<Vector3d, Vector3d> data, const sensor_msgs::ImuConstPtr& template_msg = {})
    : dt(dt_.toSec()), linear_acceleration(data.first), angular_velocity(data.second) {
        if (template_msg) {
            header = template_msg->header;
        }
        header.stamp = t;
    }
};

struct OdomMeasurement {
    std_msgs::Header header;
    double dt;
    std::pair<Eigen::Vector2d, double> velocity;
    double constraint_error_vel;

    OdomMeasurement& scale(const Eigen::Matrix3d& scale) {
        Eigen::Vector3d result = scale * Eigen::Vector3d(velocity.first[0], velocity.first[1], velocity.second);
        velocity.first         = result.head<2>();
        velocity.second        = result[2];
        return *this;
    }
};

struct OdomPoseMeasurement {
    std_msgs::Header header;
    std::pair<Eigen::Vector2d, double> pose;
    double constraint_error;
};

struct Measurement {
    vector<ImuMeasurement> imu_msgs;
    vector<pair<OdomMeasurement, pair<Vector3d, Vector3d>>> odom_aligned_msgs;
    sensor_msgs::PointCloudConstPtr img_msg;
};

class ImuUtility {
public:
    static pair<Vector3d, Vector3d> msg2data(const sensor_msgs::ImuConstPtr& msg) {
        pair<Vector3d, Vector3d> res;
        res.first.x() = msg->linear_acceleration.x;
        res.first.y() = msg->linear_acceleration.y;
        res.first.z() = msg->linear_acceleration.z;
        res.second.x() = msg->angular_velocity.x;
        res.second.y() = msg->angular_velocity.y;
        res.second.z() = msg->angular_velocity.z;
        return res;
    }

    static sensor_msgs::ImuPtr data2msg(const ros::Time& t, pair<Vector3d, Vector3d> data, const sensor_msgs::ImuConstPtr& template_msg = {}) {
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
    static pair<Vector3d, Vector3d> interplote(deque<sensor_msgs::ImuConstPtr> msgs, const ros::Time& t) {
        auto it = find_if(msgs.begin(), msgs.end(), [&](const sensor_msgs::ImuConstPtr& msg) -> bool {
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
            const auto& imu_i = **(it - 1);
            const auto& imu_j = **it;
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
    static pair<Vector3d, Vector3d> average(deque<sensor_msgs::ImuConstPtr> msgs, const ros::Time& begin_t, const ros::Time& end_t) {
        pair<Vector3d, Vector3d> imu_begin = interplote(msgs, begin_t);
        pair<Vector3d, Vector3d> imu_end = interplote(msgs, end_t);

        auto it = find_if(msgs.begin(), msgs.end(), [&](const sensor_msgs::ImuConstPtr& msg) -> bool {
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
};

class OdomUtility {
public:
    static OdomPoseMeasurement interplote(const deque<OdomPoseMeasurement>& msgs, const ros::Time& t) {
        auto it = find_if(msgs.begin(), msgs.end(), [&](const OdomPoseMeasurement& msg) -> bool {
            return msg.header.stamp >= t;
        });
        if (it == msgs.end()) {
            throw std::range_error("OdomUtility::interplote_pose(): time is too new");
        }
        OdomPoseMeasurement res;
        if ((*it).header.stamp == t) {
            res.pose = (*it).pose;
            res.constraint_error = (*it).constraint_error;
        } else {
            if (it == msgs.begin()) {
                throw std::range_error("OdomUtility::interplote_pose(): time is too old");
            }
            const auto& odom_i = *(it - 1);
            const auto& odom_j = *it;
            double dt_1 = (t - odom_i.header.stamp).toSec();
            double dt_2 = (odom_j.header.stamp - t).toSec();
            res.header = odom_j.header;
            res.header.seq = 0;
            res.header.stamp = t;
            auto vel = BaseOdometryIntegration::differential(dt_1 + dt_2, odom_i.pose, odom_j.pose);
            res.pose = BaseOdometryIntegration::integration(dt_1, odom_i.pose, vel);
            double err_1 = odom_i.constraint_error;
            double err_2 = odom_j.constraint_error;
            res.constraint_error = (dt_1 * err_2 + dt_2 * err_1) / (dt_1 + dt_2);
        }

        return res;
    }
};

class IMUPredict {
public:
    ros::Time latest_time;
    Vector3d tmp_P;
    Quaterniond tmp_Q;
    Vector3d tmp_V;
    Vector3d tmp_Ba;
    Vector3d tmp_Bg;
    Vector3d tmp_g;
    Vector3d acc_0;
    Vector3d gyr_0;

public:
    IMUPredict(const Estimator& estimator, const ros::Time& t)
    :
        latest_time(t),
        tmp_P(estimator.Ps[WINDOW_SIZE]),
        tmp_Q(estimator.Rs[WINDOW_SIZE]),
        tmp_V(estimator.Vs[WINDOW_SIZE]),
        tmp_Ba(estimator.Bas[WINDOW_SIZE]),
        tmp_Bg(estimator.Bgs[WINDOW_SIZE]),
        tmp_g(estimator.g),
        acc_0(estimator.acc_0),
        gyr_0(estimator.gyr_0) {
    }

    bool predict(const sensor_msgs::ImuConstPtr& imu_msg) {
        auto t = imu_msg->header.stamp;
//        if (latest_time.isZero()) {
            // no prev measurement
//            latest_time = t;
//            tie(acc_0, gyr_0) = ImuUtility::msg2data(imu_msg);
//            return false;
//        } else
        if (t <= latest_time) {
            // msg is older than prev
            return false;
        } else {
            // do predict
            double dt = (t - latest_time).toSec();
            Vector3d acc_1, gyr_1;
            tie(acc_1, gyr_1) = ImuUtility::msg2data(imu_msg);

            Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - tmp_g;

            Vector3d un_gyr = 0.5 * (gyr_0 + gyr_1) - tmp_Bg;
            tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

            Vector3d un_acc_1 = tmp_Q * (acc_1 - tmp_Ba) - tmp_g;

            Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

            tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
            tmp_V = tmp_V + dt * un_acc;

            latest_time = t;
            acc_0 = acc_1;
            gyr_0 = gyr_1;
            return true;
        }

    }

    void publish(std_msgs::Header header) {
        header.frame_id = "world";
        pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
};

class DataPreProcess {
private:
    bool first_feature = true;

    deque<sensor_msgs::ImuConstPtr> imu_buf;
    deque<OdomPoseMeasurement> odom_buf;
    deque<sensor_msgs::PointCloudConstPtr> feature_buf;
    deque<sensor_msgs::PointCloudConstPtr> relo_buf;
    bool first_good_img = true;
    mutex m_buf;
    condition_variable cv;

    Estimator estimator;
    mutex m_estimator;

    shared_ptr<IMUPredict> predict;
    mutex m_predict;

    ros::NodeHandle& nh;
    ros::Subscriber sub_imu;
    ros::Subscriber sub_odom;
    ros::Subscriber sub_image;
    ros::Subscriber sub_relo_points;
    std::thread process_thread;

public:
    explicit DataPreProcess(ros::NodeHandle& _nh) : nh(_nh) {
        estimator.setParameter();

        sub_imu = nh.subscribe(IMU_TOPIC, 2000, &DataPreProcess::on_imu, this, ros::TransportHints().tcpNoDelay());
        sub_odom = nh.subscribe(ODOM_TOPIC, 2000, &DataPreProcess::on_odom, this, ros::TransportHints().tcpNoDelay());
        sub_image = nh.subscribe("/feature_tracker/feature", 2000, &DataPreProcess::on_feature, this);
        sub_relo_points = nh.subscribe("/pose_graph/match_points", 2000, &DataPreProcess::on_relocalization, this);

        process_thread = std::thread(bind(&DataPreProcess::process, this));
    }

    ~DataPreProcess() {
        process_thread.join();

        sub_imu.shutdown();
        sub_odom.shutdown();
        sub_image.shutdown();
        sub_relo_points.shutdown();
    }

    void restart() {
        std::lock(m_predict, m_buf, m_estimator);
        lock_guard<mutex> lk1(m_predict, adopt_lock);
        lock_guard<mutex> lk2(m_buf, adopt_lock);
        lock_guard<mutex> lk3(m_estimator, adopt_lock);

        first_good_img = true;
        imu_buf.clear();
        odom_buf.clear();
        feature_buf.clear();
        relo_buf.clear();
        estimator.clearState();
        estimator.setParameter();
    }

    void on_imu(const sensor_msgs::ImuConstPtr& imu_msg) {
        {
            lock_guard<mutex> lk(m_buf);
            if (!imu_buf.empty() && imu_msg->header.stamp.toSec() <= imu_buf.back()->header.stamp.toSec()) {
                ROS_WARN("imu message in disorder!");
                return;
            }
            imu_buf.push_back(imu_msg);
        }
        cv.notify_one();

        {
            lock_guard<mutex> lk(m_predict);
            if (predict) {
                predict->predict(imu_msg);
                predict->publish(imu_msg->header);
            }
        }
    }

    void on_odom(const nav_msgs::OdometryConstPtr& odom_msg) {
        double px = odom_msg->pose.pose.position.x;
        double py = odom_msg->pose.pose.position.y;
        double pz = odom_msg->pose.pose.position.z;
        double qx = odom_msg->pose.pose.orientation.x;
        double qy = odom_msg->pose.pose.orientation.y;
        double qz = odom_msg->pose.pose.orientation.z;
        double qw = odom_msg->pose.pose.orientation.w;
        ROS_ASSERT(pz == 0 && qx == 0 && qy == 0);
        Eigen::Vector3d ypr = Utility::R2ypr(Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix());
        ROS_ASSERT(ypr[1] == 0 && ypr[2] == 0);
        double yaw = ypr[0] / 180.0 * M_PI;
        OdomPoseMeasurement m = {
                .header=odom_msg->header,
                .pose={{px, py}, yaw},
                .constraint_error=odom_msg->pose.covariance[1]};

        {
            lock_guard<mutex> lk(m_buf);
            if (!odom_buf.empty() && odom_msg->header.stamp <= odom_buf.back().header.stamp) {
                ROS_WARN_STREAM("odom message in disorder!");
                return;
            }
            odom_buf.push_back(m);
        }
        cv.notify_one();
    }

    void on_feature(const sensor_msgs::PointCloudConstPtr& msg) {
        if (first_feature) {
            //skip the first detected feature, which doesn't contain optical flow speed
            first_feature = false;
            return;
        }
        {
            lock_guard<mutex> lk(m_buf);
            feature_buf.push_back(msg);
        }
        cv.notify_one();
    }

    void on_relocalization(const sensor_msgs::PointCloudConstPtr& points_msg) {
        lock_guard<mutex> lk(m_buf);
        relo_buf.push_back(points_msg);
    }

    Measurement get_measurement(const ros::Duration& td_cam, const ros::Duration& td_odom) {
        sensor_msgs::PointCloudConstPtr img_msg;
        if (first_good_img) {
            // wait for imu before odom
            if (imu_buf.empty()) {
                throw std::range_error("wait for imu before odom");
            }
            // wait for odom before cam
            if (odom_buf.empty() || odom_buf.back().header.stamp + td_odom < imu_buf.front()->header.stamp) {
                throw std::range_error("wait for odom before cam");
            }
            const auto& first_odom = *find_if(odom_buf.begin(), odom_buf.end(), [&](const OdomPoseMeasurement& odom_msg) {
                return odom_msg.header.stamp + td_odom >= imu_buf.front()->header.stamp;
            });
            // wait for cam
            if (feature_buf.empty() || feature_buf.back()->header.stamp + td_cam < first_odom.header.stamp + td_odom) {
                feature_buf.clear();
                throw std::range_error("wait for cam");
            }
            while (feature_buf.front()->header.stamp + td_cam < first_odom.header.stamp + td_odom) {
                feature_buf.pop_front();
            }
            img_msg = feature_buf.front();
            first_good_img = false;
        } else {
            // wait for new cam
            if (feature_buf.size() <= 1) {
                throw std::range_error("wait for new cam");
            }
            img_msg = feature_buf[1];
        }
        // wait for imu after cam
        if (imu_buf.back()->header.stamp < img_msg->header.stamp + td_cam) {
            throw std::range_error("wait for imu after cam");
        }
        // wait for odom after cam
        if (odom_buf.back().header.stamp + td_odom < img_msg->header.stamp + td_cam) {
            throw std::range_error("wait for odom after cam");
        }

        // generate
        Measurement m;
        m.img_msg = img_msg;

        ros::Time imu_begin_t, imu_end_t, odom_begin_t, odom_end_t;
        if (feature_buf.size() == 1) {
            // first measurement
            imu_begin_t = imu_buf.front()->header.stamp;
            imu_end_t = m.img_msg->header.stamp + td_cam;
            odom_begin_t = odom_buf.front().header.stamp;
            odom_end_t = m.img_msg->header.stamp + td_cam - td_odom;
            ROS_INFO("First measure: odom %lf sec, imu %lf sec",
                     (odom_end_t - odom_begin_t).toSec(),
                     (imu_end_t - imu_begin_t).toSec());
        } else {
            // other measurement
            const auto &prev_img_t = feature_buf.front()->header.stamp;
            imu_begin_t = prev_img_t + td_cam;
            imu_end_t = m.img_msg->header.stamp + td_cam;
            odom_begin_t = prev_img_t + td_cam - td_odom;
            odom_end_t = m.img_msg->header.stamp + td_cam - td_odom;
        }

        // insert IMU to measurement
        auto prev_imu_t = imu_begin_t;
        for (const auto& imu_msg : imu_buf) {
            if (imu_msg->header.stamp > imu_begin_t && imu_msg->header.stamp < imu_end_t) {
                auto imu_t = imu_msg->header.stamp;
                auto imu_data = ImuUtility::msg2data(imu_msg);
                m.imu_msgs.emplace_back(
                        imu_msg->header.stamp,
                        imu_msg->header.stamp - prev_imu_t,
                        imu_data,
                        imu_buf.back());
                prev_imu_t = imu_t;
            }
        }
        m.imu_msgs.emplace_back(
                imu_end_t,
                imu_end_t - prev_imu_t,
                ImuUtility::interplote(imu_buf, imu_end_t),
                imu_buf.back());

        // insert odom to measurement
        auto odom_begin_pose = OdomUtility::interplote(odom_buf, odom_begin_t);
        auto odom_end_pose = OdomUtility::interplote(odom_buf, odom_end_t);
        auto prev_odom_t = odom_begin_t;
        auto prev_odom_pose = odom_begin_pose;
        for (const auto& odom_msg : odom_buf) {
            ros::Time odom_t;
            OdomPoseMeasurement odom_pose;
            if (odom_msg.header.stamp > odom_begin_t && odom_msg.header.stamp < odom_end_t) {
                odom_t = odom_msg.header.stamp;
                odom_pose = odom_msg;
            } else if (odom_msg.header.stamp >= odom_end_t) {
                odom_t = odom_end_t;
                odom_pose = odom_end_pose;
            } else {
                continue;
            }

            OdomMeasurement odom_m;
            odom_m.header = odom_buf.back().header;
            odom_m.header.stamp = odom_t;
            odom_m.dt = (odom_t - prev_odom_t).toSec();
            odom_m.velocity = BaseOdometryIntegration::differential(odom_m.dt, prev_odom_pose.pose, odom_pose.pose);
            odom_m.constraint_error_vel = (odom_pose.constraint_error - prev_odom_pose.constraint_error) / odom_m.dt;
//                ROS_INFO_STREAM(prev_odom_pose << " -> " << odom_pose << " = " << odom_m.velocity);

            auto acc_gyro = ImuUtility::average(imu_buf, prev_odom_t, odom_t);
            m.odom_aligned_msgs.emplace_back(odom_m, acc_gyro);

            prev_odom_t = odom_t;
            prev_odom_pose = odom_pose;
            if (odom_msg.header.stamp >= odom_end_t) {
                break;
            }
        }

        // delete outdated
        const auto& odom_before_img_t = (*find_if(odom_buf.rbegin(), odom_buf.rend(), [&](const OdomPoseMeasurement& odom_msg) {
            return odom_msg.header.stamp + td_odom <= img_msg->header.stamp + td_cam;
        })).header.stamp;
        while (odom_buf.size() > 2 && odom_buf[2].header.stamp <= odom_before_img_t) {
//            ROS_INFO_STREAM("pop odom_buf " << odom_buf.front().header.stamp);
            odom_buf.pop_front();
        }
        while (imu_buf.size() > 2 && imu_buf[2]->header.stamp <= odom_before_img_t + td_odom) {
//            ROS_INFO_STREAM("pop imu_buf " << imu_buf.front()->header.stamp);
            imu_buf.pop_front();
        }
        while (feature_buf.size() > 1 && feature_buf[1]->header.stamp <= img_msg->header.stamp) {
//            ROS_INFO_STREAM("pop feature_buf " << feature_buf.front()->header.stamp);
            feature_buf.pop_front();
        }

        return m;
    }

    void process() {
        do {
            unique_lock<mutex> lk(m_buf);
            if (cv.wait_for(lk, chrono::milliseconds(100)) == std::cv_status::timeout)
                continue;
            Measurement measurement;
            try {
                measurement = get_measurement(ros::Duration(estimator.td), ros::Duration(estimator.td_bo));
            } catch (const std::range_error& e) {
//                ROS_INFO_STREAM("No measure: " << e.what());
                continue;
            }
            sensor_msgs::PointCloudConstPtr relo_msg;
            if (!relo_buf.empty()) {
                // set relocalization frame
                relo_msg = relo_buf.front();
                relo_buf.clear();
            }
            m_buf.unlock();

//            ROS_INFO_STREAM("New measure: " << measurement.img_msg->header.stamp << ", " << measurement.imu_msgs.size() << " imu & " << measurement.odom_aligned_msgs.size() << " odom");

            {
                lock_guard<mutex> lk2(m_estimator);

#if 0
                // process measurement by time order [optional]
                auto it_imu = measurement.imu_msgs.begin();
                auto it_odom = measurement.odom_aligned_msgs.begin();
                while (it_imu != measurement.imu_msgs.end() && it_odom != measurement.odom_aligned_msgs.end()) {
                    if (it_imu != measurement.imu_msgs.end()
                        && (it_odom == measurement.odom_aligned_msgs.end()
                            || it_imu->header.stamp <= it_odom->first.header.stamp + ros::Duration(estimator.td_bo)))
                    {
                        auto m = *it_imu;
                        estimator.processIMU(m.dt, m.linear_acceleration, m.angular_velocity);
                        it_imu++;
                    } else {
                        auto m = *it_odom;
                        estimator.processOdometry(m.first.dt, m.first.velocity, m.second);
                        pubVelocityYaw(estimator, m.first.header);
                        it_odom++;
                    }
                }
#else
                // process measurement
                for (const auto& m : measurement.imu_msgs) {
                    estimator.processIMU(m.dt, m.linear_acceleration, m.angular_velocity);
                }
                for (const auto& m : measurement.odom_aligned_msgs) {
                    estimator.processOdometry(m.first.dt, m.first.velocity, m.first.constraint_error_vel, m.second.first, m.second.second);
//                    pubMixedOdom(m.first.header, estimator.wheel_imu_odom.measurements.back(), estimator.wheel_odom_niose_analyser.filted_.back());
                }
#endif

                if (relo_msg != nullptr)
                {
                    vector<Vector3d> match_points;
                    double frame_stamp = relo_msg->header.stamp.toSec();
                    for (const auto point : relo_msg->points)
                    {
                        Vector3d u_v_id;
                        u_v_id.x() = point.x;
                        u_v_id.y() = point.y;
                        u_v_id.z() = point.z;
                        match_points.push_back(u_v_id);
                    }
                    Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                    Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                    Matrix3d relo_r = relo_q.toRotationMatrix();
                    int frame_index;
                    frame_index = relo_msg->channels[0].values[7];
                    estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
                }

                auto img_msg = measurement.img_msg;
                ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

                TicToc t_s;
                map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
                for (unsigned int i = 0; i < img_msg->points.size(); i++)
                {
                    int v = img_msg->channels[0].values[i] + 0.5;
                    int feature_id = v / NUM_OF_CAM;
                    int camera_id = v % NUM_OF_CAM;
                    double x = img_msg->points[i].x; // unified planar x, unit: m
                    double y = img_msg->points[i].y; // unified planar y, unit: m
                    double z = img_msg->points[i].z; // 1.0 m
                    double p_u = img_msg->channels[1].values[i]; // image planar, unit: pixel
                    double p_v = img_msg->channels[2].values[i]; // image planar, unit: pixel
                    double velocity_x = img_msg->channels[3].values[i]; // velocity of point in image planar, unit: pixel/s
                    double velocity_y = img_msg->channels[4].values[i]; // velocity of point in image planar, unit: pixel/s
                    ROS_ASSERT(z == 1);
                    Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                    xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                    image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
                }
                estimator.processImage(image, img_msg->header);

                double whole_t = t_s.toc();
                printStatistics(estimator, whole_t);
                std_msgs::Header header = img_msg->header;
                header.frame_id = "world";

                pubWheelOdomPathTF(estimator, header);
                pubOdometry(estimator, header);
                pubKeyPoses(estimator, header);
                pubCameraPose(estimator, header);
                pubPointCloud(estimator, header);
                pubTF(estimator, header);
                pubKeyframe(estimator);
                if (relo_msg != nullptr)
                    pubRelocalization(estimator);
                //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());

                if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) {
                    std::lock(m_buf, m_predict);
                    lock_guard<mutex> lk1(m_buf, std::adopt_lock);
                    lock_guard<mutex> lk2(m_predict, std::adopt_lock);

                    predict = make_shared<IMUPredict>(estimator, measurement.imu_msgs.back().header.stamp);
                    for (const auto& imu_msg : imu_buf) {
                        predict->predict(imu_msg);
                    }
                }
                ROS_DEBUG("processed vision data with stamp %f \n", img_msg->header.stamp.toSec());
            }
        } while (nh.ok());
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    readParameters(nh);
    DataPreProcess data_pre_process(nh);

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_INFO("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image, odom and imu...");

    registerPub(nh);

    ros::Subscriber sub_restart = nh.subscribe("/feature_tracker/restart", 2000, boost::function<void(const std_msgs::BoolConstPtr&)>([&](const std_msgs::BoolConstPtr& restart_msg) {
        if (restart_msg->data != 0) {
            ROS_WARN("restart the estimator!");
            data_pre_process.restart();
        }
    }));

    ros::spin();

    return 0;
}
